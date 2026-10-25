# Whisper voice input

Status: shipped as an optional, automatically detected feature. Press the mic key to start capturing, press it again to stop, transcribe on the Frame, and type the result into the focused field. This simple toggle-capture-type flow supersedes the designed editor-and-server dictation of [dictation.md](dictation.md) for what is currently shipped; that plan stays a design document.

Nothing is configured inside the app. Full Keyboard looks for a whisper.cpp command and a model at startup and on every **Reload config**. When either is missing, the mic key is simply not shown and no microphone is ever opened.

## Enable it

Full Keyboard does not bundle whisper.cpp or any model. On the Frame, install the command with the user-local [GNU Guix](https://guix.gnu.org/) package manager and download a model into the app's data directory:

```sh
guix-pkg install whisper-cpp
mkdir -p ~/.local/share/framekeyboard/models
curl -L --create-dirs -o ~/.local/share/framekeyboard/models/ggml-tiny.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin
```

Use `~/.local/bin/guix-pkg` if `guix-pkg` is not on your `PATH`. Both steps are per-user: no `sudo`, no system packages, no SteamOS write access, and nothing outside your home directory.

Then open Full Keyboard (or press **Reload config** in Settings if it was already open). The mic key appears below Paste.

- **Models.** Any `ggml-*.bin` Whisper model in `~/.local/share/framekeyboard/models/` works; the app picks the best installed name, preferring `small`, then `base`, then `tiny`. `ggml-tiny.bin` (77 MB) is the smallest starting point; `ggml-base.bin` (145 MB) and `ggml-small.bin` (494 MB) are more accurate and slower. Models are downloaded from the whisper.cpp model repository, never by the app.
- **Guix details.** Rootless Guix runs its programs inside a container, so the installed `whisper-cli` is not directly executable from the gaming session. Detection finds the store copy and invokes it through the profile wrapper (`~/.local/bin/guix-env whisper-cli`), which resolves the profile's own version inside the container. A plain `whisper-cli` on `PATH` or at `~/.guix-profile/bin/whisper-cli` is preferred when present.

## Use it

| State | Key appearance | Status line |
| --- | --- | --- |
| Off (ready) | Normal mic key | — |
| Capturing | Mic key latched | *Listening. Press the mic key again to transcribe and type.* |
| Transcribing | Mic key dimmed | *Transcribing. Keep the keyboard open.* |
| Done | Normal | Transcription typed into the field; status clears |

- Press the mic key to start capture, speak, then press it again. The keyboard stays usable while the transcription runs; the text is typed as one commit when it is ready.
- Capture stops automatically at 5 minutes and transcribes what it has.
- Hiding or closing the keyboard while capturing discards the recording; nothing is typed into a target that is no longer visible.
- Clips shorter than 0.3 s report *The recording was too short.* Silence and non-speech report *No speech was detected.*

## Warm server

While the keyboard is shown, Full Keyboard keeps a `whisper-server` running beside it so the model stays loaded and clips skip the per-run process and container startup: on the Frame, a short phrase takes roughly 5 s with `ggml-base`, 15–30 s with `ggml-small` and about 2 s with `ggml-tiny` — the model load itself is absorbed by the server, never by your press. The server is started when the keyboard first appears, kept across clips, and stopped shortly after the keyboard hides — nothing Whisper-related stays running while the keyboard is closed.

- **First press.** When the keyboard has just opened, the server is still loading its model; the first transcription waits for that load, later ones do not.
- **Same machine only.** The server listens on a random loopback port on `127.0.0.1`; clips are POSTed in memory (never written to disk on this path), nothing leaves the Frame, and nothing is logged.
- **Automatic fallback.** If the server cannot be found or keeps failing, every later clip silently uses the one-shot CLI path; the keyboard never depends on the server.
- Set `FRAMEKEYBOARD_WHISPER_SERVER` to another command, or to the empty value to disable the server and keep the per-clip CLI. `FRAMEKEYBOARD_WHISPER_SERVER_IDLE` sets how many seconds (0–3600, default 180) the server stays up after the keyboard hides.

## Overrides

All optional; the defaults are what the two install commands above give you. They are read at startup and on **Reload config**, so set them in the environment the app runs in and relaunch or reload.

| Variable | Default | Meaning |
| --- | --- | --- |
| `FRAMEKEYBOARD_WHISPER_CLI` | auto-detect | Command to run, e.g. a full path or wrapper. A missing file disables the mic key. |
| `FRAMEKEYBOARD_WHISPER_MODEL` | best `ggml-*.bin` in the models directory | Exact model file; a missing file disables the mic key. |
| `FRAMEKEYBOARD_WHISPER_LANGUAGE` | `auto` | Language code such as `en` or `de` to skip detection. |
| `FRAMEKEYBOARD_WHISPER_THREADS` | `4` | Inference threads, 1–32. |
| `FRAMEKEYBOARD_WHISPER_SERVER` | auto-detect | Warm-server command; the empty value disables the server (see [Warm server](#warm-server)). |
| `FRAMEKEYBOARD_WHISPER_SERVER_IDLE` | `180` | Seconds (0–3600) the server stays up after the keyboard hides. |

Detection order for the command: `FRAMEKEYBOARD_WHISPER_CLI`, then `whisper-cli` on `PATH`, then `~/.guix-profile/bin/whisper-cli`, then the rootless Guix store via the profile wrapper.

## Privacy and limits

- Audio is captured only between the two presses and lives only in memory. On the warm-server path it never touches disk at all; on the one-shot CLI path it exists only as one private WAV file (`0600` in a `0700` directory under `$XDG_CACHE_HOME/framekeyboard`) that is removed immediately after each run. Nothing is uploaded: recognition is entirely on-device, and the server's port is loopback-only.
- Transcripts, typed text, and errors are never logged. Results pass through the same text-output queue as typed characters, so they are never sent when the backend cannot deliver (a preview launch reports that the transcription was discarded instead).
- Results are treated as untrusted text: invalid UTF-8, control characters, non-speech annotations like `[Music]`, decoder repetition loops, and implausibly long results for the clip's length are removed or rejected with a fixed message.
- The Frame's microphone must be permitted by the system; capture opens the default recording device through SDL's PipeWire backend. If no usable 16 kHz mono capture device exists, the mic key reports the failure and keeps working as unpressed.
- Quality with `ggml-tiny` is limited, like any tiny Whisper model; install `base` or `small` for better results. The mic key appears in all six bundled layouts; custom layouts opt in with a key of `"action": {"kind": "whisper", "value": "toggle"}` and `"icon": "mic"` (see [configuration](configuration.md)).
