// Whisper voice input: SDL captures the headset microphone into a buffer on
// the audio thread, one worker process turns the buffer into text with an
// external whisper.cpp command, and App types the result through its normal
// output queue. Nothing about speech is logged, and audio never touches disk
// outside one private, immediately removed WAV file.
#include "framekeyboard/whisper.hpp"

#include <SDL.h>
#include <algorithm>
#include <arpa/inet.h>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <glib.h>
#include <json-c/json.h>
#include <netinet/in.h>
#include <signal.h>
#include <sstream>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace framekeyboard {
namespace fs = std::filesystem;
namespace {
constexpr int capture_rate = 16000; // Whisper's input rate; no resampling here.
constexpr double max_capture_seconds = 300.0;
constexpr double min_capture_seconds = 0.3;
constexpr std::size_t max_transcript_bytes = 256 * 1024;
constexpr int cli_timeout_seconds = 900; // Small models spend tens of seconds
                                         // loading per run; long clips then
                                         // need minutes of CPU inference.
// The resident server loads large models for tens of seconds; a request then
// runs at most as long as the one-shot CLI may.
constexpr gint64 server_ready_timeout_us = 300ll * 1000 * 1000;
constexpr int server_request_timeout_seconds = 900;

fs::path data_home() {
    if (const char* base = std::getenv("XDG_DATA_HOME"); base && *base && fs::path(base).is_absolute()) {
        return base;
    }
    if (const char* home = std::getenv("HOME")) {
        return fs::path(home) / ".local/share";
    }
    return {};
}

fs::path cache_home() {
    if (const char* base = std::getenv("XDG_CACHE_HOME");
        base && *base && fs::path(base).is_absolute()) {
        return base;
    }
    if (const char* home = std::getenv("HOME")) {
        return fs::path(home) / ".cache";
    }
    return {};
}

bool executable(const fs::path& path) {
    std::error_code ec;
    return !path.empty() && fs::is_regular_file(path, ec) && access(path.c_str(), X_OK) == 0;
}

std::optional<fs::path> find_on_path(const std::string& name) {
    if (const char* path = std::getenv("PATH")) {
        std::istringstream stream(path);
        std::string directory;
        while (std::getline(stream, directory, ':')) {
            if (directory.empty()) {
                continue;
            }
            if (auto candidate = fs::path(directory) / name; executable(candidate)) {
                return candidate;
            }
        }
    }
    return {};
}

// Splits an environment-supplied command; the first word resolves on PATH
// when it is not an absolute path.
std::vector<std::string> tokenize_command(const char* value) {
    std::vector<std::string> argv;
    std::istringstream stream(value);
    std::string token;
    while (stream >> token) {
        argv.push_back(token);
    }
    if (!argv.empty() && !fs::path(argv.front()).is_absolute()) {
        if (auto found = find_on_path(argv.front())) {
            argv.front() = found->string();
        }
    }
    return argv;
}

// The directory of the running executable; /proc/self/exe covers the
// installed release and the development tree alike.
fs::path executable_dir() {
    std::array<char, 4096> buffer{};
    const ssize_t len = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (len <= 0) {
        return {};
    }
    return fs::path(std::string(buffer.data(), static_cast<std::size_t>(len))).parent_path();
}

// The shipped whisper-cli sits beside the keyboard binary;
// FRAMEKEYBOARD_WHISPER_CLI overrides it for development and tests.
std::vector<std::string> detect_command(const fs::path& app_dir) {
    if (const char* cli = std::getenv("FRAMEKEYBOARD_WHISPER_CLI"); cli && *cli) {
        return tokenize_command(cli);
    }
    if (auto bundled = app_dir / "whisper-cli"; executable(bundled)) {
        return {bundled.string()};
    }
    return {};
}

// whisper-server keeps one model loaded for many clips, so dictation stops
// paying the model load per press. It is opt-in: a resident helper holds a
// Vulkan context and its GPU buffers for the whole keyboard appearance, and
// while a VR session was rendering, a resident server's request faulted the
// Frame's GPU out from under the keyboard (VK_ERROR_DEVICE_LOST, session
// restart; observed 2026-10-26). The one-shot CLI loads the model per clip
// for about a second on the GPU and frees everything when it exits, so it is
// the default. FRAMEKEYBOARD_WHISPER_SERVER names the warm-server command;
// the bundled binary is used only when this variable explicitly asks.
std::vector<std::string> detect_server_command(const std::vector<std::string>& cli,
                                               const fs::path& app_dir) {
    if (const char* server = std::getenv("FRAMEKEYBOARD_WHISPER_SERVER")) {
        if (!*server) {
            return {}; // Explicitly set empty: CLI per clip, no server.
        }
        return tokenize_command(server);
    }
    // Unset: no warm server. (The variable's name is the opt-in.)
    (void)cli;
    (void)app_dir;
    return {};
}

// Higher is the better on-device default from docs/dictation.md's
// measurements: small is the agreed balance, then base, then tiny.
int model_rank(const std::string& filename) {
    auto has = [&filename](const char* needle) { return filename.find(needle) != std::string::npos; };
    if (has("small")) {
        return 4;
    }
    if (has("base")) {
        return 3;
    }
    if (has("tiny")) {
        return 2;
    }
    return 1;
}

std::optional<fs::path> detect_model() {
    if (const char* model = std::getenv("FRAMEKEYBOARD_WHISPER_MODEL"); model && *model) {
        std::error_code ec;
        if (fs::is_regular_file(model, ec)) {
            return fs::path(model);
        }
        return {};
    }
    const auto directory = data_home().empty() ? fs::path() : data_home() / "framekeyboard/models";
    std::error_code ec;
    if (directory.empty() || !fs::is_directory(directory, ec)) {
        return {};
    }
    std::optional<fs::path> best;
    int best_rank = 0;
    for (const auto& item : fs::directory_iterator(directory, ec)) {
        const auto name = item.path().filename().string();
        if (!item.is_regular_file(ec) || !name.starts_with("ggml-") || !name.ends_with(".bin")) {
            continue;
        }
        const int rank = model_rank(name);
        if (!best || rank > best_rank ||
            (rank == best_rank && item.path().filename() < best->filename())) {
            best = item.path();
            best_rank = rank;
        }
    }
    return best;
}

void append_char(std::string& out, char32_t c) {
    if (g_unichar_isspace(c)) {
        out += ' ';
        return;
    }
    if (c < 0x20 || (c >= 0x7f && c < 0xa0)) {
        return; // Control characters never reach the target.
    }
    std::array<char, 8> buffer{};
    const int size = g_unichar_to_utf8(c, buffer.data());
    out.append(buffer.data(), static_cast<std::size_t>(size));
}

// Whisper annotates non-speech instead of staying quiet. Only bracket spans
// made entirely of known annotation words are removed, so dictated
// parentheses survive.
bool annotation_content(const std::string& content) {
    static const std::vector<std::string> words = {
        // Must stay sorted for binary_search.
        "audio",       "applause",     "applauding", "background", "bell",           "bells",
        "blank_audio", "buzzer",       "cheer",      "cheers",     "crosstalk",      "end",
        "inaudible",   "instrumental", "laugh",      "laughing",   "laughter",       "music",
        "noise",       "quiet",        "silence",    "silent",     "singing",        "song",
        "sound",       "sounds",       "static",     "talking",    "unintelligible",
    };
    std::string word;
    bool any = false;
    auto flush = [&]() {
        if (word.empty()) {
            return true;
        }
        any = true;
        const bool known = std::binary_search(words.begin(), words.end(), word);
        word.clear();
        return known;
    };
    for (char c : content) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
            word += static_cast<char>(g_ascii_tolower(c));
            continue;
        }
        if (!flush()) {
            return false;
        }
    }
    // An all-word span ("[Music]") flushes only now; empty content is not an
    // annotation.
    return flush() && any;
}

std::string strip_annotations(const std::string& text) {
    std::string out;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char open = text[i];
        if (open != '[' && open != '(' && open != '{') {
            out += open;
            continue;
        }
        const char close = open == '[' ? ']' : open == '(' ? ')' : '}';
        const auto end = text.find(close, i + 1);
        if (end == std::string::npos || end - i > 40) {
            out += open;
            continue;
        }
        const auto content = text.substr(i + 1, end - i - 1);
        if (annotation_content(content)) {
            out += ' ';
        } else {
            out += open;
            out += content;
            out += close;
        }
        i = end;
    }
    return out;
}

// Whisper decoder loops repeat a run of words; four or more copies collapse to
// one, as docs/dictation.md's cleanup requires. Shorter runs of two or three
// are ordinary dictated repetitions and stay.
std::string fold_repeats(const std::string& text) {
    std::istringstream in(text);
    std::vector<std::string> words;
    for (std::string word; in >> word;) {
        words.push_back(word);
    }
    std::string out;
    auto append = [&](const std::string& word) {
        if (!out.empty()) {
            out += ' ';
        }
        out += word;
    };
    std::size_t i = 0;
    while (i < words.size()) {
        std::size_t run = 1;
        while (i + run < words.size() &&
               g_ascii_strcasecmp(words[i + run].c_str(), words[i].c_str()) == 0) {
            ++run;
        }
        if (run >= 4) {
            append(words[i]); // A decoder loop leaves one copy behind.
        } else {
            for (std::size_t k = 0; k < run; ++k) {
                append(words[i + k]);
            }
        }
        i += run;
    }
    return out;
}

// Full-scale fraction (0..1) of the loudest sample in a clip. Room tone sits
// far below normal speech; this separates "the user made a sound" from
// "the clip was silence", which decides whether Whisper's text can be
// trusted at any speed.
double peak_amplitude(const std::vector<std::int16_t>& samples) {
    std::int32_t peak = 0;
    for (const auto sample : samples) {
        const std::int32_t value =
            sample == INT16_MIN ? 32768 : std::abs(static_cast<std::int32_t>(sample));
        peak = std::max(peak, value);
    }
    return static_cast<double>(peak) / 32768.0;
}
constexpr double audible_peak = 0.02; // 2 % of full scale: above room tone.

// Whisper writes typographic punctuation (curly quotes, en/em dashes,
// ellipsis) that no physical key can produce and that pushes a phrase off
// the ordinary-keymap typing path. Fold those to the ASCII the keyboard
// types; spoken words are never altered, only their punctuation marks.
std::string fold_typography(const std::string& text) {
    std::string out;
    for (const char* it = text.c_str(); *it;) {
        const auto c = static_cast<char32_t>(g_utf8_get_char(it));
        it = g_utf8_next_char(it);
        switch (c) {
        case U'\u2018':
        case U'\u2019':
        case U'\u201A':
            out += '\'';
            break;
        case U'\u201C':
        case U'\u201D':
            out += '"';
            break;
        case U'\u2013':
        case U'\u2014':
            out += '-';
            break;
        case U'\u2026':
            out += "...";
            break;
        case U'\u00A0':
            out += ' ';
            break;
        default:
            append_char(out, c);
        }
    }
    return out;
}
} // namespace

// Gamescope's text commit carries at most 32 characters at once, so longer
// results are delivered as chunks the output queue types in order. Each word
// keeps its trailing space, so the chunks concatenate to exactly the input;
// splits fall on word boundaries, and words longer than the limit split at
// character boundaries.
std::vector<std::string> split_commits(const std::string& text, std::size_t limit) {
    std::vector<std::string> chunks;
    std::string chunk;
    std::size_t characters = 0;
    const auto flush = [&] {
        if (!chunk.empty()) {
            chunks.push_back(chunk);
            chunk.clear();
            characters = 0;
        }
    };
    for (const char* it = text.c_str(); *it;) {
        const char* word_end = it;
        std::size_t word_characters = 0;
        while (*word_end && *word_end != ' ') {
            word_end = g_utf8_next_char(word_end);
            ++word_characters;
        }
        const bool trailing_space = *word_end == ' ';
        std::string token(it, static_cast<std::size_t>(word_end - it));
        if (trailing_space) {
            token += ' ';
        }
        const std::size_t token_characters = word_characters + (trailing_space ? 1 : 0);
        if (token_characters <= limit && characters + token_characters > limit) {
            flush(); // Start a new commit; the token, spaces included, moves whole.
        }
        // Append, force-splitting an over-limit token at the limit.
        for (const char* piece = token.c_str(); *piece;) {
            const char* next = g_utf8_next_char(piece);
            if (characters == limit) {
                flush();
            }
            chunk.append(piece, static_cast<std::size_t>(next - piece));
            ++characters;
            piece = next;
        }
        it = trailing_space ? word_end + 1 : word_end;
    }
    flush();
    return chunks;
}

TranscriptCheck clean_transcript(const std::string& raw, double seconds, double peak) {
    if (!g_utf8_validate(raw.c_str(), -1, nullptr)) {
        return {"", "The transcription was not valid text."};
    }
    std::string text;
    for (const char* it = raw.c_str(); *it;) {
        const auto c = static_cast<char32_t>(g_utf8_get_char(it));
        it = g_utf8_next_char(it);
        append_char(text, c);
    }
    text = strip_annotations(text);
    text = fold_repeats(text);
    text = fold_typography(text);
    const double characters = static_cast<double>(g_utf8_strlen(text.c_str(), -1));
    if (text.empty()) {
        return {"", "No speech was detected."};
    }
    // The speed rule applies only to near-silent clips: text over 25
    // characters per second of quiet audio is a Whisper hallucination of a
    // room, not of a speaker. A clip that contains real sound is typed as
    // heard — dictated speech must never be discarded for being too fast.
    if (peak < audible_peak && seconds >= min_capture_seconds && characters > 25.0 * seconds) {
        return {"", "No speech was detected."};
    }
    return {text, ""};
}

std::vector<char> make_wav(const std::vector<std::int16_t>& samples) {
    const std::uint32_t data_bytes = static_cast<std::uint32_t>(samples.size() * 2);
    std::vector<char> out(44 + samples.size() * 2);
    auto* out_bytes = reinterpret_cast<unsigned char*>(out.data());
    auto put32 = [&](std::size_t at, std::uint32_t value) {
        for (int i = 0; i < 4; ++i) {
            out_bytes[at + i] = static_cast<unsigned char>((value >> (8 * i)) & 0xff);
        }
    };
    auto put16 = [&](std::size_t at, std::uint16_t value) {
        out_bytes[at] = static_cast<unsigned char>(value & 0xff);
        out_bytes[at + 1] = static_cast<unsigned char>(value >> 8);
    };
    std::memcpy(out.data(), "RIFF", 4);
    put32(4, 36 + data_bytes);
    std::memcpy(out.data() + 8, "WAVEfmt ", 8);
    put32(16, 16);
    put16(20, 1); // PCM
    put16(22, 1); // mono
    put32(24, capture_rate);
    put32(28, capture_rate * 2);
    put16(32, 2);
    put16(34, 16);
    std::memcpy(out.data() + 36, "data", 4);
    put32(40, data_bytes);
    std::memcpy(out.data() + 44, samples.data(), samples.size() * 2);
    return out;
}

std::optional<WhisperBackend> detect_whisper_backend(const fs::path& app_dir) {
    const fs::path directory = app_dir.empty() ? executable_dir() : app_dir;
    auto command = detect_command(directory);
    if (command.empty() || !executable(command.front())) {
        return {};
    }
    auto model = detect_model();
    if (!model) {
        return {};
    }
    WhisperBackend backend;
    backend.command = std::move(command);
    backend.model = std::move(*model);
    if (const char* language = std::getenv("FRAMEKEYBOARD_WHISPER_LANGUAGE"); language && *language) {
        std::string value(language);
        if (value == "auto" || (value.size() <= 8 && std::all_of(value.begin(), value.end(), [](char c) {
                                    return std::isalpha(static_cast<unsigned char>(c)) || c == '-';
                                }))) {
            backend.language = std::move(value);
        }
    }
    if (const char* threads = std::getenv("FRAMEKEYBOARD_WHISPER_THREADS"); threads && *threads) {
        const int value = std::atoi(threads);
        if (value >= 1 && value <= 32) {
            backend.threads = value;
        }
    }
    backend.server_command = detect_server_command(backend.command, directory);
    if (const char* idle = std::getenv("FRAMEKEYBOARD_WHISPER_SERVER_IDLE"); idle && *idle) {
        const int value = std::atoi(idle);
        if (value >= 0 && value <= 3600) {
            backend.server_idle_seconds = value;
        }
    }
    return backend;
}

void WhisperInput::capture(void* userdata, unsigned char* stream, int len) {
    auto* self = static_cast<WhisperInput*>(userdata);
    const std::lock_guard guard(self->mutex_);
    if (!self->capturing_ || self->discard_) {
        return;
    }
    const auto* samples = reinterpret_cast<const std::int16_t*>(stream);
    const std::size_t count = static_cast<std::size_t>(len) / 2;
    const std::size_t limit = static_cast<std::size_t>(max_capture_seconds * capture_rate);
    const std::size_t room = limit - std::min(limit, self->buffer_.size());
    if (count > room) {
        self->full_ = true;
    }
    self->buffer_.insert(self->buffer_.end(), samples, samples + std::min(count, room));
}

WhisperInput::WhisperInput(WhisperBackend backend) : backend_(std::move(backend)) {
    server_idle_us_ = static_cast<gint64>(backend_.server_idle_seconds) * 1000;
    worker_ = std::thread([this] { run(); });
}

WhisperInput::~WhisperInput() {
    {
        const std::lock_guard guard(mutex_);
        stopping_ = true;
        discard_ = true;
    }
    wake_.notify_all();
    if (child_ > 0) {
        kill(child_, SIGKILL);
    }
    if (worker_.joinable()) {
        worker_.join();
    }
    {
        // The worker may have left the warm server between requests.
        const std::lock_guard guard(mutex_);
        server_stop();
    }
    stop_capture();
}

bool WhisperInput::capturing() const {
    const std::lock_guard guard(mutex_);
    return capturing_;
}

bool WhisperInput::working() const {
    const std::lock_guard guard(mutex_);
    return busy_ || !jobs_.empty();
}

bool WhisperInput::full() const {
    const std::lock_guard guard(mutex_);
    return full_;
}

bool WhisperInput::server_running() const {
    const std::lock_guard guard(mutex_);
    return server_pid_ > 0;
}

void WhisperInput::set_visible(bool visible) {
    bool changed = false;
    {
        const std::lock_guard guard(mutex_);
        changed = visible_ != visible;
        if (changed) {
            visible_ = visible;
            hidden_us_ = visible ? -1 : g_get_monotonic_time();
        }
    }
    if (changed) {
        wake_.notify_all(); // The warm-server policy reacts immediately.
    }
}

void WhisperInput::stop_capture() {
    {
        const std::lock_guard guard(mutex_);
        capturing_ = false;
    }
    if (device_ != 0) {
        SDL_PauseAudioDevice(device_, 1);
        SDL_CloseAudioDevice(device_); // Joins the capture callback.
        device_ = 0;
    }
}

bool WhisperInput::start(std::string& error) {
    if (capturing()) {
        return true;
    }
    stop_capture();
    {
        const std::lock_guard guard(mutex_);
        buffer_.clear();
        result_.reset();
        full_ = false;
        discard_ = false;
    }
    // App owns SDL audio alone; preview's window never initializes it. The
    // Frame's capture devices stream over PipeWire, while SDL's automatic
    // choice can open a silent PulseAudio default, so prefer PipeWire once.
    const bool prefer_pipewire = !std::getenv("SDL_AUDIODRIVER");
    if (prefer_pipewire) {
        SDL_SetHint(SDL_HINT_AUDIODRIVER, "pipewire");
    }
    if (!SDL_WasInit(SDL_INIT_AUDIO) && SDL_InitSubSystem(SDL_INIT_AUDIO) != 0) {
        error = "Audio is unavailable: " + std::string(SDL_GetError());
        return false;
    }
    SDL_AudioSpec want{}, have{};
    want.freq = capture_rate;
    want.format = AUDIO_S16SYS;
    want.channels = 1;
    want.samples = 1024;
    want.callback = capture;
    want.userdata = this;
    device_ = static_cast<int>(SDL_OpenAudioDevice(nullptr, 1, &want, &have, 0));
    if (device_ == 0 && prefer_pipewire) {
        // Retry on the automatically chosen driver, as on a dev host
        // without PipeWire or with an unusual SDL build.
        SDL_QuitSubSystem(SDL_INIT_AUDIO);
        SDL_SetHint(SDL_HINT_AUDIODRIVER, "");
        if (SDL_InitSubSystem(SDL_INIT_AUDIO) != 0) {
            error = "Audio is unavailable: " + std::string(SDL_GetError());
            return false;
        }
        device_ = static_cast<int>(SDL_OpenAudioDevice(nullptr, 1, &want, &have, 0));
    }
    if (device_ == 0 || have.freq != capture_rate || have.channels != 1 || have.format != AUDIO_S16SYS) {
        if (device_ != 0) {
            SDL_CloseAudioDevice(device_);
            device_ = 0;
        }
        error = "No microphone input was available.";
        return false;
    }
    {
        const std::lock_guard guard(mutex_);
        capturing_ = true;
    }
    SDL_PauseAudioDevice(device_, 0);
    return true;
}

void WhisperInput::stop_and_transcribe() {
    stop_capture();
    std::vector<std::int16_t> samples;
    {
        const std::lock_guard guard(mutex_);
        samples.swap(buffer_);
        full_ = false;
        if (samples.empty()) {
            result_ = Result{"", "No speech was captured."};
            return;
        }
        jobs_.push_back(std::move(samples));
    }
    wake_.notify_all();
}

void WhisperInput::abort() {
    stop_capture();
    {
        const std::lock_guard guard(mutex_);
        buffer_.clear();
        jobs_.clear();
        result_.reset();
        full_ = false;
        discard_ = true; // A CLI still running loses its result on finish.
    }
    if (child_ > 0) {
        kill(child_, SIGKILL);
    }
}

std::optional<WhisperInput::Result> WhisperInput::take_result() {
    const std::lock_guard guard(mutex_);
    auto result = std::move(result_);
    result_.reset();
    return result;
}

void WhisperInput::submit(std::vector<std::int16_t> samples) {
    {
        const std::lock_guard guard(mutex_);
        jobs_.push_back(std::move(samples));
        discard_ = false;
    }
    wake_.notify_all();
}

WhisperInput::Result WhisperInput::transcribe(std::vector<std::int16_t> samples) {
    const double seconds = static_cast<double>(samples.size()) / capture_rate;
    const double peak = peak_amplitude(samples);
    if (seconds < min_capture_seconds) {
        return {"", "The recording was too short."};
    }
    if (std::error_code ec; !fs::is_regular_file(backend_.model, ec)) {
        return {"", "The Whisper model is missing."};
    }
    {
        const std::lock_guard guard(mutex_);
        if (stopping_ || discard_) {
            return {"", "The transcription was cancelled."};
        }
    }
    const auto wav_data = make_wav(samples);
    // The warm server answers in inference time only; any failure falls back
    // to the one-shot CLI below, and repeated failures end server use.
    std::optional<std::string> server_text;
    pid_t pid = -1;
    int port = 0;
    bool server_ready = false;
    {
        const std::lock_guard guard(mutex_);
        if (!server_gave_up_ && server_pid_ > 0) {
            pid = server_pid_;
            port = server_port_;
            server_ready = server_ready_;
        }
    }
    if (pid > 0) {
        // Readiness and the request run without the mutex: a large model
        // loads for tens of seconds, and capture must keep filling. A server
        // that has answered before only needs a moment; a starting one may
        // legitimately take minutes to bind its port.
        const gint64 budget = server_ready ? 5ll * 1000 * 1000 : server_ready_timeout_us;
        if (wait_ready(pid, port, budget) == Readiness::ready && !stopping_ && !discard_) {
            server_text = http_transcribe(port, wav_data);
        }
        const std::lock_guard guard(mutex_);
        if (server_text) {
            server_ready_ = true;
            server_failures_ = 0;
        } else if (!stopping_ && !discard_) {
            if (server_pid_ == pid) {
                server_stop(); // Kills a live-but-broken server; the next tick
                               // restarts it if the warm policy still wants one.
            }
            if (++server_failures_ >= 3) {
                server_gave_up_ = true; // The one-shot CLI is the reliable fallback.
            }
        }
    }
    if (server_text) {
        const auto check = clean_transcript(*server_text, seconds, peak);
        return Result{check.text, check.error};
    }
    fs::path directory = cache_home().empty() ? fs::path("/tmp") : cache_home() / "framekeyboard";
    std::error_code ec;
    fs::create_directories(directory, ec);
    if (ec) {
        return {"", "Cannot create the Whisper working directory."};
    }
    fs::permissions(directory, fs::perms::owner_all, fs::perm_options::replace, ec);
    // The clip exists on disk only as this private file pair, and only until
    // the cleanup below; nothing here is ever logged.
    const std::string base = (directory / ("whisper-" + std::to_string(static_cast<long>(getpid())) +
                                           "-" + std::to_string(g_random_int())))
                                 .string();
    const fs::path wav = base + ".wav";
    const fs::path transcript = base + ".txt";
    {
        const int fd = open(wav.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0) {
            return {"", "Cannot create the Whisper working file."};
        }
        std::FILE* file = fdopen(fd, "wb");
        if (!file || std::fwrite(wav_data.data(), 1, wav_data.size(), file) != wav_data.size()) {
            if (file) {
                std::fclose(file);
            } else {
                ::close(fd);
            }
            std::filesystem::remove(wav, ec);
            return {"", "Cannot write the recording."};
        }
        std::fclose(file);
    }
    std::vector<std::string> argv = backend_.command;
    const std::array<std::pair<const char*, std::string>, 7> options = {
        std::pair{"-m", backend_.model.string()},
        std::pair{"-f", wav.string()},
        std::pair{"-l", backend_.language},
        std::pair{"-t", std::to_string(backend_.threads)},
        std::pair{"-of", base},
        std::pair{"-otxt", ""},
        std::pair{"-np", ""},
    };
    for (const auto& [flag, value] : options) {
        argv.push_back(flag);
        if (!value.empty()) {
            argv.push_back(value);
        }
    }
    std::vector<std::vector<char>> storage;
    std::vector<char*> raw;
    for (auto& argument : argv) {
        storage.push_back(std::vector<char>(argument.begin(), argument.end()));
        storage.back().push_back('\0');
    }
    for (auto& buffer : storage) {
        raw.push_back(buffer.data());
    }
    raw.push_back(nullptr);
    // Diagnostics for GPU-side failures: the engine's own stderr is captured
    // so a whisper helper's last words outlive the process. Transcript text
    // travels on stdout (still discarded) and stderr lines are filtered to
    // engine prefixes below; the file is private and never logged elsewhere.
    int err_pipe[2] = {-1, -1};
    if (pipe2(err_pipe, O_CLOEXEC | O_NONBLOCK) != 0) {
        err_pipe[0] = err_pipe[1] = -1;
    }
    const pid_t cli_child = fork();
    if (cli_child < 0) {
        if (err_pipe[0] >= 0) {
            close(err_pipe[0]);
            close(err_pipe[1]);
        }
        std::filesystem::remove(wav, ec);
        return {"", "Cannot start the Whisper process."};
    }
    if (cli_child == 0) {
        // The whisper helper runs on the GPU when its build has a Vulkan
        // backend; session overlay layers would only interfere.
        setenv("VK_LOADER_LAYERS_DISABLE", "~implicit~", 0);
        const int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            dup2(devnull, STDOUT_FILENO);
            if (err_pipe[1] < 0) {
                dup2(devnull, STDERR_FILENO);
            }
            if (devnull > STDERR_FILENO) {
                close(devnull);
            }
        }
        if (err_pipe[1] >= 0) {
            dup2(err_pipe[1], STDERR_FILENO);
            close(err_pipe[0]);
            close(err_pipe[1]);
        }
        execv(raw[0], raw.data());
        _exit(127);
    }
    if (err_pipe[1] >= 0) {
        close(err_pipe[1]);
    }
    std::string engine_log;
    std::string engine_carry; // trailing partial line between drains
    bool engine_header_written = false;
    const auto engine_flush = [&] {
        if (err_pipe[0] < 0) {
            return;
        }
        // Keep only whole lines; a partial tail waits for the next drain.
        engine_log.insert(0, engine_carry);
        const auto last_newline = engine_log.find_last_of('\n');
        if (last_newline == std::string::npos) {
            engine_carry = engine_log;
            engine_log.clear();
            return;
        }
        engine_carry = engine_log.substr(last_newline + 1);
        engine_log.erase(last_newline + 1);
        // Engine diagnostics only: keep lines the whisper/ggml engine or the
        // Vulkan loader emits. Never a transcript (stdout) and never user
        // text: whisper-cli stderr carries no typed content.
        static constexpr std::array<const char*, 12> kEngineMarkers = {
            "ggml", "whisper", "main:",  "system_info", "load_",   "encode:",
            "vk",   "VULKAN",  "vulkan", "error",       "warning", "register_",
        };
        std::string kept;
        std::istringstream lines(engine_log);
        for (std::string line; std::getline(lines, line);) {
            const std::string lowered = [&line] {
                std::string value = line;
                std::transform(value.begin(), value.end(), value.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                return value;
            }();
            for (const char* marker : kEngineMarkers) {
                if (lowered.find(marker) != std::string::npos) {
                    kept += line;
                    kept += '\n';
                    break;
                }
            }
        }
        engine_log.clear();
        if (kept.empty() && engine_header_written) {
            return;
        }
        const fs::path directory =
            cache_home().empty() ? fs::path("/tmp/framekeyboard") : cache_home() / "framekeyboard";
        std::error_code ec;
        std::filesystem::create_directories(directory, ec);
        if (ec) {
            return;
        }
        const fs::path log = directory / "whisper-helper.log";
        if (const auto size = std::filesystem::file_size(log, ec); !ec && size > 1024 * 1024) {
            std::filesystem::resize_file(log, 0, ec); // keep the file bounded
        }
        const int fd = open(log.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
        if (fd < 0) {
            return;
        }
        std::FILE* file = fdopen(fd, "ae");
        if (!file) {
            ::close(fd);
            return;
        }
        if (!engine_header_written) {
            std::fprintf(file, "[%lld] run started\n", static_cast<long long>(std::time(nullptr)));
            engine_header_written = true;
        }
        std::fputs(kept.c_str(), file);
        std::fflush(file);
        std::fclose(file);
    };
    const auto drain = [&] {
        if (err_pipe[0] < 0) {
            return;
        }
        char chunk[8192];
        while (true) {
            const ssize_t got = read(err_pipe[0], chunk, sizeof(chunk));
            if (got <= 0) {
                return;
            }
            if (engine_log.size() < 256 * 1024) {
                engine_log.append(chunk, static_cast<std::size_t>(got));
            }
        }
    };
    child_ = cli_child;
    int status = -1;
    const double deadline = static_cast<double>(g_get_monotonic_time()) / 1e6 + cli_timeout_seconds;
    bool abandoned = false;
    while (true) {
        drain();
        engine_flush();
        const int outcome = waitpid(cli_child, &status, WNOHANG);
        if (outcome == cli_child) {
            break;
        }
        if (outcome < 0) {
            abandoned = true;
            break;
        }
        bool cancel = false;
        {
            const std::lock_guard guard(mutex_);
            cancel = stopping_ || discard_;
        }
        if (cancel || static_cast<double>(g_get_monotonic_time()) / 1e6 > deadline) {
            kill(cli_child, SIGKILL);
            abandoned = true;
            waitpid(cli_child, &status, 0);
            break;
        }
        g_usleep(50 * 1000);
    }
    // The child is gone: whatever remains in the pipe is its final output.
    drain();
    engine_flush();
    if (err_pipe[0] >= 0) {
        close(err_pipe[0]);
    }
    child_ = -1;
    struct Cleanup {
        fs::path wav, transcript;
        ~Cleanup() {
            std::error_code ec;
            std::filesystem::remove(wav, ec);
            std::filesystem::remove(transcript, ec);
        }
    } cleanup{wav, transcript};
    if (abandoned) {
        return {"", "The transcription was cancelled."};
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        if (WIFEXITED(status) && WEXITSTATUS(status) == 127) {
            return {"", "The Whisper command could not run."};
        }
        return {"", "Whisper failed to transcribe the recording."};
    }
    std::FILE* file = std::fopen(transcript.c_str(), "rb");
    if (!file) {
        return {"", "Whisper produced no text."};
    }
    std::string raw_text;
    std::array<char, 4096> chunk{};
    while (raw_text.size() <= max_transcript_bytes) {
        const auto got = std::fread(chunk.data(), 1, chunk.size(), file);
        raw_text.append(chunk.data(), got);
        if (got < chunk.size()) {
            break;
        }
    }
    std::fclose(file);
    if (raw_text.size() > max_transcript_bytes) {
        return {"", "The result was rejected as unreliable."};
    }
    const auto check = clean_transcript(raw_text, seconds, peak);
    return Result{check.text, check.error};
}

void WhisperInput::run() {
    while (true) {
        std::vector<std::int16_t> samples;
        {
            std::unique_lock lock(mutex_);
            // The 2 s bound is the warm-server tick: it starts when the panel
            // shows and stops once the hidden grace period ends.
            wake_.wait_for(lock, std::chrono::seconds(2),
                           [&] { return stopping_.load() || !jobs_.empty(); });
            if (stopping_) {
                server_stop();
                return;
            }
            manage_server();
            if (jobs_.empty()) {
                continue;
            }
            samples = std::move(jobs_.front());
            jobs_.pop_front();
            busy_ = true;
        }
        // Fixed-shape diagnostics for reports like "long sentences do not
        // work": clip duration and peak, and the size or category of the
        // outcome. Never the transcript itself.
        const double clip_seconds = static_cast<double>(samples.size()) / capture_rate;
        const double clip_peak = peak_amplitude(samples);
        Result result = transcribe(std::move(samples));
        std::fprintf(
            stdout, "Dictation clip: %.1f s, peak %.4f, %s\n", clip_seconds, clip_peak,
            result.text.empty()
                ? ("outcome: " + result.error).c_str()
                : ("result " + std::to_string(g_utf8_strlen(result.text.c_str(), -1)) + " chars")
                      .c_str());
        std::fflush(stdout);
        const std::lock_guard guard(mutex_);
        busy_ = false;
        if (!discard_) {
            result_ = std::move(result);
        }
    }
}

// Worker-only policy, called with the mutex held.
void WhisperInput::manage_server() {
    const gint64 now = g_get_monotonic_time();
    if (server_pid_ > 0 && !visible_ && hidden_us_ >= 0 && now - hidden_us_ >= server_idle_us_) {
        server_stop(); // The panel closed and the grace period ended.
    }
    if (!visible_ || server_gave_up_ || backend_.server_command.empty()) {
        return;
    }
    if (server_pid_ > 0) {
        if (waitpid(server_pid_, nullptr, WNOHANG) != server_pid_) {
            return; // The running server stays as it is.
        }
        server_pid_ = -1; // It died; count it before the retry below.
        ++server_failures_;
    }
    if (server_failures_ >= 3) {
        server_gave_up_ = true; // Clips keep working through the one-shot CLI.
        return;
    }
    // Start the server; its tens-of-seconds model load is absorbed by the
    // request's readiness poll, never by the UI thread.
    const int port = 20000 + static_cast<int>(g_random_int() % 20000);
    std::vector<std::string> argv = backend_.server_command;
    argv.insert(argv.end(), {"--host", "127.0.0.1", "--port", std::to_string(port), "-m",
                             backend_.model.string(), "-t", std::to_string(backend_.threads)});
    std::vector<std::vector<char>> storage;
    std::vector<char*> raw;
    for (auto& argument : argv) {
        storage.push_back(std::vector<char>(argument.begin(), argument.end()));
        storage.back().push_back('\0');
    }
    for (auto& buffer : storage) {
        raw.push_back(buffer.data());
    }
    raw.push_back(nullptr);
    const pid_t pid = fork();
    if (pid < 0) {
        server_gave_up_ = true;
        return;
    }
    if (pid == 0) {
        // An app that dies outside its own cleanup (SIGTERM, crash) must not
        // leave a resident server behind holding a loaded model.
        prctl(PR_SET_PDEATHSIG, SIGTERM);
        // A gaming session injects implicit Vulkan layers (overlay stacks)
        // that break the whisper Vulkan backend; a compute-only helper has
        // no use for them. An explicitly set environment wins.
        setenv("VK_LOADER_LAYERS_DISABLE", "~implicit~", 0);
        const int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            dup2(devnull, STDOUT_FILENO);
            dup2(devnull, STDERR_FILENO);
            if (devnull > STDERR_FILENO) {
                close(devnull);
            }
        }
        execv(raw[0], raw.data());
        _exit(127);
    }
    server_pid_ = pid;
    server_port_ = port;
}

// With the mutex held. The server exits on its own after SIGTERM; the bounded
// wait keeps a wedged one from holding the worker. An already-reaped or
// unknown pid returns at once.
void WhisperInput::server_stop() {
    if (server_pid_ <= 0) {
        return;
    }
    kill(server_pid_, SIGTERM);
    const gint64 deadline = g_get_monotonic_time() + 3ll * 1000 * 1000;
    while (true) {
        const int outcome = waitpid(server_pid_, nullptr, WNOHANG);
        if (outcome == server_pid_ || outcome == -1) {
            break; // Exited, or never ours to reap.
        }
        if (g_get_monotonic_time() > deadline) {
            kill(server_pid_, SIGKILL);
            waitpid(server_pid_, nullptr, 0);
            break;
        }
        g_usleep(50 * 1000);
    }
    server_pid_ = -1;
    server_port_ = 0;
    server_ready_ = false;
}

// Polls without the mutex; the caller copies the pid first.
WhisperInput::Readiness WhisperInput::wait_ready(pid_t pid, int port, std::int64_t timeout_us) {
    const gint64 deadline = g_get_monotonic_time() + timeout_us;
    while (true) {
        if (stopping_) {
            return Readiness::timeout;
        }
        if (waitpid(pid, nullptr, WNOHANG) == pid) {
            return Readiness::dead;
        }
        // A plain blocking loopback connect answers at once; a nonblocking
        // one here never reports completion without a poll, so keep it simple.
        const int probe = socket(AF_INET, SOCK_STREAM, 0);
        if (probe >= 0) {
            timeval timeout{};
            timeout.tv_usec = 250 * 1000;
            setsockopt(probe, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
            setsockopt(probe, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            sockaddr_in address{};
            address.sin_family = AF_INET;
            address.sin_port = htons(static_cast<std::uint16_t>(port));
            address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            const bool open =
                connect(probe, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0;
            close(probe);
            if (open) {
                return Readiness::ready;
            }
        }
        if (g_get_monotonic_time() > deadline) {
            return Readiness::timeout;
        }
        g_usleep(250 * 1000);
    }
}

// One POST of the clip; the response is a JSON object whose "text" is the
// transcription. Loopback only; nothing leaves the Frame and nothing logs.
std::optional<std::string> WhisperInput::http_transcribe(int port, const std::vector<char>& wav) {
    static const std::string boundary = "----framekeyboard-boundary";
    std::string body = "--" + boundary +
                       "\r\nContent-Disposition: form-data; name=\"file\"; "
                       "filename=\"clip.wav\"\r\nContent-Type: audio/wav\r\n\r\n";
    body.append(wav.data(), wav.size());
    body += "\r\n--" + boundary + "\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n";
    body += backend_.language;
    body += "\r\n--" + boundary + "--\r\n";
    std::string request = "POST /inference HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: "
                          "multipart/form-data; boundary=" +
                          boundary + "\r\nContent-Length: " + std::to_string(body.size()) +
                          "\r\nConnection: close\r\n\r\n" + body;
    const int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        return {};
    }
    timeval timeout{};
    timeout.tv_sec = server_request_timeout_seconds;
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<std::uint16_t>(port));
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(sock, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        close(sock);
        return {};
    }
    std::size_t sent = 0;
    while (sent < request.size()) {
        const auto wrote = send(sock, request.data() + sent, request.size() - sent, MSG_NOSIGNAL);
        if (wrote <= 0) {
            close(sock);
            return {};
        }
        sent += static_cast<std::size_t>(wrote);
    }
    std::string response;
    while (response.size() < 8 * 1024 * 1024) {
        char chunk[4096];
        const auto got = recv(sock, chunk, sizeof(chunk), 0);
        if (got <= 0) {
            break;
        }
        response.append(chunk, static_cast<std::size_t>(got));
    }
    close(sock);
    const auto split = response.find("\r\n\r\n");
    if (split == std::string::npos || response.compare(0, 13, "HTTP/1.1 200 ") != 0) {
        return {};
    }
    json_object* root = json_tokener_parse(response.substr(split + 4).c_str());
    if (!root) {
        return {};
    }
    std::optional<std::string> text;
    json_object* value = nullptr;
    if (json_object_object_get_ex(root, "text", &value)) {
        text = json_object_get_string(value);
    }
    json_object_put(root);
    return text;
}
} // namespace framekeyboard
