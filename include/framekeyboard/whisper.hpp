#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace framekeyboard {

// A usable whisper.cpp command and model, plus an optional resident server.
// The commands are the whisper-cli / whisper-server shipped with the keyboard;
// environment overrides may name a wrapper that takes whisper-cli's arguments.
struct WhisperBackend {
    std::vector<std::string> command; // Executable, then fixed leading arguments.
    // whisper-server with the same model, opt-in via FRAMEKEYBOARD_WHISPER_SERVER:
    // loaded once per keyboard appearance so each clip is one localhost
    // request instead of one model load. Empty by default; every clip then
    // runs the one-shot CLI, whose GPU memory is freed when it exits.
    std::vector<std::string> server_command;
    std::filesystem::path model;
    std::string language{"auto"};
    int threads{4};
    // Seconds the warm server may stay loaded after the keyboard is hidden.
    int server_idle_seconds{180};
};

// Finds the bundled transcriber and model: the whisper helpers installed
// beside the running keyboard binary (app_dir), with environment overrides
// still honored. Empty when either is missing, so the mic key stays hidden
// and no capture device is ever opened. The default app_dir is the directory
// of the running executable.
std::optional<WhisperBackend> detect_whisper_backend(const std::filesystem::path& app_dir = {});

// Recognized text is untrusted input. Validation and cleanup: UTF-8, control
// characters, non-speech annotations, whitespace, repeated-word loops and a
// length plausible for the clip's duration. An empty `text` with a fixed
// `error` means there was nothing to type or the phrase failed.
struct TranscriptCheck {
    std::string text, error;
};
TranscriptCheck clean_transcript(const std::string& raw, double seconds, double peak);

// Split text into pieces of at most `limit` characters for one text commit
// each (Gamescope's per-commit limit is 32); splits fall on word boundaries.
std::vector<std::string> split_commits(const std::string& text, std::size_t limit = 32);

// 16 kHz mono little-endian PCM as a RIFF WAVE file.
std::vector<char> make_wav(const std::vector<std::int16_t>& samples);

// Whisper voice input: capture while toggled on, transcribe once when toggled
// off. SDL's capture thread only fills the buffer; one worker runs the CLI.
// App alone delivers results, through its OutputController like any text.
class WhisperInput {
  public:
    explicit WhisperInput(WhisperBackend backend);
    ~WhisperInput();
    WhisperInput(const WhisperInput&) = delete;
    WhisperInput& operator=(const WhisperInput&) = delete;

    bool capturing() const;
    bool working() const;        // Queued or running transcription.
    bool full() const;           // The capture limit was reached; the caller stops.
    bool server_running() const; // The warm whisper-server process is alive.

    // Keyboard visibility drives the warm server: it loads the model once
    // while the panel is shown, and stops once the hidden grace period ends.
    // Cheap to call every frame; App forwards the VR lifecycle signal.
    void set_visible(bool visible);

    // Opens the microphone and starts buffering. Fails when no capture
    // device exists; the buffer is empty afterwards either way.
    bool start(std::string& error);
    // Stops capture and queues everything buffered so far. A capture with no
    // samples yields an immediate result instead of a worker job.
    void stop_and_transcribe();
    // Stops capture and discards the buffer, queued work and results: for the
    // hide and cancel paths, where typing into an invisible target is wrong.
    void abort();

    struct Result {
        std::string text, error;
    };
    std::optional<Result> take_result();

    // Tests feed samples and fake commands directly; no microphone needed.
    void submit(std::vector<std::int16_t> samples);

  private:
    void run(); // Worker loop.
    Result transcribe(std::vector<std::int16_t> samples);
    void stop_capture();
    // Resident-server helpers; all are worker-thread code. The mutex is held
    // on entry except where a comment says the socket work runs unlocked.
    void manage_server(); // Warm/stop policy; lock held.
    void server_stop();   // Terminate and reap; lock held.
    enum class Readiness { ready, dead, timeout };
    Readiness wait_ready(pid_t pid, int port, std::int64_t timeout_us); // Polls the port; no lock.
    std::optional<std::string> http_transcribe(int port, const std::vector<char>& wav);
    // SDL's audio thread only appends under the mutex; it never starts work.
    // `stream` is SDL's Uint8 buffer; the spelling keeps SDL out of this header.
    static void capture(void* userdata, unsigned char* stream, int len);

    WhisperBackend backend_;
    std::thread worker_;
    mutable std::mutex mutex_; // Const accessors (capturing/working/full) lock it.
    std::condition_variable wake_;
    std::deque<std::vector<std::int16_t>> jobs_;
    std::optional<Result> result_;
    std::vector<std::int16_t> buffer_;
    bool capturing_{}, full_{}, busy_{}, discard_{};
    std::atomic<bool> stopping_{}; // Pollable while the worker blocks in a request.
    int device_{};                 // SDL_AudioDeviceID; 0 keeps SDL out of this header.
    std::atomic<pid_t> child_{-1}; // The one-shot CLI, while it runs.
    // The resident server belongs to the worker; its fields need the mutex.
    pid_t server_pid_{-1};
    int server_port_{};
    bool server_ready_{};           // The server has answered its port at least once.
    int server_failures_{};         // Request failures; repeated ones end server use.
    bool server_gave_up_{};         // No binary or repeated start failures: CLI only.
    bool visible_{true};            // App signals otherwise; the preview is always shown.
    std::int64_t hidden_us_{-1};    // When the panel hid, for the idle grace period.
    std::int64_t server_idle_us_{}; // backend_.server_idle_seconds in microseconds.
};
} // namespace framekeyboard
