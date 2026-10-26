// Whisper voice input tests: WAV writing, transcript cleanup, transcriber
// detection and the worker against fake commands. Nothing here opens a
// microphone, contacts the network or sends input anywhere.
#include "framekeyboard/whisper.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

using namespace framekeyboard;
namespace fs = std::filesystem;
namespace {
void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
struct TemporaryDirectory {
    fs::path path;
    TemporaryDirectory() {
        std::string name = "/tmp/framekeyboard-whisper-tests-XXXXXX";
        if (!mkdtemp(name.data())) {
            throw std::runtime_error("mkdtemp failed");
        }
        path = name;
    }
    ~TemporaryDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

fs::path write_script(const fs::path& path, const std::string& body) {
    std::ofstream file(path);
    file << body;
    file.close();
    chmod(path.c_str(), 0755);
    return path;
}

std::vector<std::int16_t> tone(double seconds) {
    std::vector<std::int16_t> samples(static_cast<std::size_t>(seconds * 16000));
    for (std::size_t i = 0; i < samples.size(); ++i) {
        samples[i] = static_cast<std::int16_t>(8000 * ((i / 32) % 2 ? 1 : -1));
    }
    return samples;
}

void test_wav() {
    const auto samples = std::vector<std::int16_t>{1, -1, 2, 3};
    const auto wav = make_wav(samples);
    require(wav.size() == 44 + samples.size() * 2, "wav length");
    require(std::string(wav.data(), 4) == "RIFF", "RIFF magic");
    require(std::string(wav.data() + 8, 8) == "WAVEfmt ", "WAVE fmt");
    const auto* bytes = reinterpret_cast<const unsigned char*>(wav.data());
    const std::uint32_t data_at =
        bytes[40] | (bytes[41] << 8) | (bytes[42] << 16) | (std::uint32_t(bytes[43]) << 24);
    require(data_at == samples.size() * 2, "data chunk size");
    require(std::memcmp(wav.data() + 44, samples.data(), samples.size() * 2) == 0, "samples");
    std::cout << "ok wav\n";
}

void test_cleanup() {
    {
        const auto check = clean_transcript("hello  [Music]   world\n", 4.0);
        require(check.error.empty(), "valid phrase has no error: " + check.error);
        require(check.text == "hello world", "annotation and whitespace cleanup");
    }
    {
        const auto check = clean_transcript("[BLANK_AUDIO]", 4.0);
        require(check.error == "No speech was detected.", "silence annotation yields nothing");
    }
    {
        const auto check = clean_transcript("laugh (music) loud", 4.0);
        require(check.text == "laugh loud", "round annotation removed");
    }
    {
        const auto check = clean_transcript("keep (the music plays) inside", 4.0);
        require(check.text == "keep (the music plays) inside", "dictated parentheses survive");
    }
    {
        const auto check = clean_transcript("the the the the the end", 4.0);
        require(check.text == "the end", "decoder loop collapses");
    }
    {
        const auto check = clean_transcript("wait wait end", 4.0);
        require(check.text == "wait wait end", "short dictated repeats stay");
    }
    {
        const auto check = clean_transcript("a\tb\nc\r\nd", 4.0);
        require(check.text == "a b c d", "control characters become spaces");
    }
    {
        const auto check = clean_transcript(std::string("\xff\xfe bad utf8", 12), 4.0);
        require(check.error == "The transcription was not valid text.", "invalid UTF-8 rejected");
    }
    {
        std::string long_text;
        for (int i = 0; i < 40; ++i) {
            long_text += "w" + std::to_string(i) + " ";
        }
        const auto check = clean_transcript(long_text, 1.0);
        require(check.error == "The result was rejected as unreliable.", "implausible length");
    }
    std::cout << "ok cleanup\n";
}

// Detection reads only the environment and these fixtures, never the system.
void test_detection(const fs::path& home) {
    const auto saved = [](const char* name) {
        const char* value = std::getenv(name);
        return value ? std::string(value) : std::string();
    };
    const auto restore = [](const char* name, const std::string& value) {
        if (value.empty()) {
            unsetenv(name);
        } else {
            setenv(name, value.c_str(), 1);
        }
    };
    const auto saved_home = saved("HOME");
    const auto saved_data = saved("XDG_DATA_HOME");
    const auto saved_cli = saved("FRAMEKEYBOARD_WHISPER_CLI");
    const auto saved_model = saved("FRAMEKEYBOARD_WHISPER_MODEL");

    const auto models = home / "data/framekeyboard/models";
    fs::create_directories(models);
    {
        std::ofstream(models / "ggml-tiny.bin"), std::ofstream(models / "ggml-base.bin");
    }
    // The helpers the package installs beside the keyboard binary.
    const auto app_dir = home / "release/bin";
    fs::create_directories(app_dir);
    write_script(app_dir / "whisper-cli", "#!/bin/sh\nexit 0\n");
    write_script(app_dir / "whisper-server", "#!/bin/sh\nexit 0\n");
    setenv("HOME", home.c_str(), 1);
    setenv("XDG_DATA_HOME", (home / "data").c_str(), 1);
    unsetenv("FRAMEKEYBOARD_WHISPER_CLI");
    unsetenv("FRAMEKEYBOARD_WHISPER_MODEL");
    unsetenv("FRAMEKEYBOARD_WHISPER_SERVER");
    auto backend = detect_whisper_backend(app_dir);
    require(backend.has_value(), "bundled helpers plus a model are detected");
    require(backend->command.front() == (app_dir / "whisper-cli").string(), "bundled command");
    require(backend->server_command.empty(),
            "the default runs the one-shot CLI; no resident GPU context next to VR");
    require(backend->model.filename() == "ggml-base.bin", "small before base before tiny");
    require(backend->language == "auto" && backend->threads == 4, "defaults");

    // Opt-in warm server: the variable names the command.
    setenv("FRAMEKEYBOARD_WHISPER_SERVER", (app_dir / "whisper-server").string().c_str(), 1);
    backend = detect_whisper_backend(app_dir);
    require(backend->server_command.front() == (app_dir / "whisper-server").string(),
            "an explicit server command is used");
    unsetenv("FRAMEKEYBOARD_WHISPER_SERVER");

    setenv("FRAMEKEYBOARD_WHISPER_SERVER", "", 1);
    backend = detect_whisper_backend(app_dir);
    require(backend.has_value() && backend->server_command.empty(),
            "an explicitly empty server value keeps the CLI-only path");
    unsetenv("FRAMEKEYBOARD_WHISPER_SERVER");

    setenv("FRAMEKEYBOARD_WHISPER_MODEL", (models / "ggml-tiny.bin").c_str(), 1);
    setenv("FRAMEKEYBOARD_WHISPER_LANGUAGE", "de", 1);
    setenv("FRAMEKEYBOARD_WHISPER_THREADS", "2", 1);
    backend = detect_whisper_backend(app_dir);
    require(backend.has_value(), "explicit model");
    require(backend->model.filename() == "ggml-tiny.bin", "explicit model wins");
    require(backend->language == "de" && backend->threads == 2, "language and threads");

    setenv("FRAMEKEYBOARD_WHISPER_MODEL", (models / "missing.bin").c_str(), 1);
    require(!detect_whisper_backend(app_dir).has_value(), "a missing explicit model means unavailable");

    unsetenv("FRAMEKEYBOARD_WHISPER_MODEL");
    // No helpers beside this directory and no override: nothing to run.
    require(!detect_whisper_backend(home / "empty-release/bin").has_value(),
            "no bundled command means unavailable");

    restore("HOME", saved_home);
    restore("XDG_DATA_HOME", saved_data);
    restore("FRAMEKEYBOARD_WHISPER_CLI", saved_cli);
    restore("FRAMEKEYBOARD_WHISPER_MODEL", saved_model);
    std::cout << "ok detection\n";
}

// Waits for the worker's result; the fake commands finish far inside this.
// The wider budget covers the warm server's start tick and port readiness.
std::optional<WhisperInput::Result> wait_result(WhisperInput& input) {
    for (int attempt = 0; attempt < 1500; ++attempt) {
        if (const auto result = input.take_result()) {
            return result;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return {};
}

void test_worker(const fs::path& home) {
    fs::create_directories(home);
    WhisperBackend backend;
    backend.model = home / "ggml-fake.bin";
    {
        std::ofstream(backend.model) << "fake";
    }
    const auto ok = write_script(home / "fake-cli", R"sh(#!/bin/sh
out=""; in=""; model=""
while [ $# -gt 0 ]; do
  case "$1" in
    -of) out="$2"; shift 2 ;;
    -f)  in="$2";  shift 2 ;;
    -m)  model="$2"; shift 2 ;;
    *) shift ;;
  esac
done
[ -f "$model" ] || exit 3
head -c 4 "$in" | grep -q RIFF || exit 4
printf ' hello [Music]  world\n' > "$out.txt"
)sh");
    const auto failure = write_script(home / "failing-cli", "#!/bin/sh\nexit 2\n");
    const auto empty = write_script(home / "empty-cli", R"sh(#!/bin/sh
out=""
while [ $# -gt 0 ]; do
  case "$1" in
    -of) out="$2"; shift 2 ;;
    *) shift ;;
  esac
done
printf ' [Silence] \n' > "$out.txt"
)sh");

    backend.command = {ok.string()};
    WhisperInput input(backend);
    input.submit(tone(0.5));
    const auto result = wait_result(input);
    require(result.has_value(), "a fake CLI result arrives");
    require(result->error.empty(), "success: " + result->error);
    require(result->text == "hello world", "worker output cleaned");

    // The working files are removed after every run, success or failure.
    backend.command = {failure.string()};
    WhisperInput broken(backend);
    broken.submit(tone(0.5));
    const auto failed = wait_result(broken);
    require(failed && failed->error == "Whisper failed to transcribe the recording.",
            "a failing CLI reports a fixed error");

    backend.command = {empty.string()};
    WhisperInput quiet(backend);
    quiet.submit(tone(0.5));
    const auto nothing = wait_result(quiet);
    require(nothing && nothing->error == "No speech was detected.", "annotation-only text fails");

    WhisperInput short_(backend);
    short_.submit(tone(0.1));
    const auto early = wait_result(short_);
    require(early && early->error == "The recording was too short.", "too-short clips fail early");

    std::error_code ec;
    std::size_t leftovers = 0;
    // The worker writes clips under $XDG_CACHE_HOME/framekeyboard, which main
    // points inside this temporary directory.
    const auto cache = fs::path(std::getenv("XDG_CACHE_HOME")) / "framekeyboard";
    for (const auto& item : fs::directory_iterator(cache, ec)) {
        if (item.path().filename().string().starts_with("whisper-")) {
            ++leftovers;
        }
    }
    require(leftovers == 0, "working files are removed");
    std::cout << "ok worker\n";
}

// The resident-server path: a fake python server stands in for whisper-server
// (same arguments and /inference JSON contract). Python 3 is a build-tool
// dependency already, and this opens no ports beyond localhost.
void test_server(const fs::path& home) {
    fs::create_directories(home);
    const auto server = write_script(home / "fake-server", R"py(#!/usr/bin/env python3
import http.server, json, sys
port = int(sys.argv[sys.argv.index("--port") + 1])
class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        assert b"RIFF" in body, "the clip must be the WAV bytes"
        payload = json.dumps({"text": "  hello world [Music]  "}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
    def log_message(self, *args):
        pass
http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
)py");
    WhisperBackend backend;
    backend.model = home / "ggml-fake.bin";
    {
        std::ofstream(backend.model) << "fake";
    }
    backend.server_command = {server.string()};
    backend.server_idle_seconds = 0; // Hidden means stopped at once.
    WhisperInput input(backend);
    input.submit(tone(0.5));
    const auto result = wait_result(input);
    require(result.has_value(), "a server result arrives");
    require(result->error.empty(), "server success: " + result->error);
    require(result->text == "hello world", "server output cleaned");
    require(input.server_running(), "the server stays warm while visible");
    // A second clip reuses the loaded server.
    input.submit(tone(0.5));
    const auto second = wait_result(input);
    require(second && second->text == "hello world", "the warm server answers again");
    input.set_visible(false);
    for (int attempt = 0; attempt < 500 && input.server_running(); ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    require(!input.server_running(), "hiding the keyboard stops the warm server");
    std::cout << "ok server\n";
}

// A server that dies at once must fall back to the one-shot CLI silently.
void test_server_fallback(const fs::path& home) {
    fs::create_directories(home);
    const auto dying = write_script(home / "dying-server", "#!/bin/sh\nexit 0\n");
    const auto ok = write_script(home / "fallback-cli", R"sh(#!/bin/sh
out=""; in=""
while [ $# -gt 0 ]; do
  case "$1" in
    -of) out="$2"; shift 2 ;;
    -f)  in="$2";  shift 2 ;;
    *) shift ;;
  esac
done
head -c 4 "$in" | grep -q RIFF || exit 4
printf ' hello world\n' > "$out.txt"
)sh");
    WhisperBackend backend;
    backend.model = home / "ggml-fake.bin";
    {
        std::ofstream(backend.model) << "fake";
    }
    backend.command = {ok.string()};
    backend.server_command = {dying.string()};
    WhisperInput input(backend);
    input.submit(tone(0.5));
    const auto result = wait_result(input);
    require(result && result->error.empty() && result->text == "hello world",
            "a dead server falls back to the CLI");
    std::cout << "ok server fallback\n";
}
} // namespace

int main() {
    const auto saved_cache = [] {
        const char* value = std::getenv("XDG_CACHE_HOME");
        return value ? std::string(value) : std::string();
    }();
    TemporaryDirectory temporary;
    setenv("XDG_CACHE_HOME", (temporary.path / "cache").c_str(), 1);
    try {
        test_wav();
        test_cleanup();
        test_detection(temporary.path / "home");
        test_worker(temporary.path / "worker");
        test_server(temporary.path / "server");
        test_server_fallback(temporary.path / "fallback");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    if (saved_cache.empty()) {
        unsetenv("XDG_CACHE_HOME");
    } else {
        setenv("XDG_CACHE_HOME", saved_cache.c_str(), 1);
    }
    std::cout << "whisper tests passed\n";
    return 0;
}
