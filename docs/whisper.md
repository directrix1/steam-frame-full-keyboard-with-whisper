# Whisper voice input

Status: shipped feature. The whisper.cpp transcriber is built from the vendored source and installed inside the Full Keyboard package; the only user step is downloading a model. Press the mic key to start capturing, press it again to stop, transcribe on the Frame, and type the result into the focused field. This simple toggle-capture-type flow supersedes the designed editor-and-server dictation of [dictation.md](dictation.md) for what is currently shipped; that plan stays a design document.

Nothing is configured inside the app. At startup and on every **Reload config**, Full Keyboard looks for its bundled transcriber and a model. When the model is missing, the mic key is simply not shown and no microphone is ever opened.

## Enable it

The transcriber ships with the keyboard (`whisper-cli` and `whisper-server` beside the keyboard binary), so the only step is a one-time model download into the app's data directory:

```sh
mkdir -p ~/.local/share/framekeyboard/models
curl -L --create-dirs -o ~/.local/share/framekeyboard/models/ggml-tiny.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin
```

The step is per-user: no `sudo`, no system packages, no SteamOS write access, and nothing outside your home directory.

Then open Full Keyboard (or press **Reload config** in Settings if it was already open). The mic key appears below Paste.

- **Models.** Any `ggml-*.bin` Whisper model in `~/.local/share/framekeyboard/models/` works; the app picks the best installed name, preferring `small`, then `base`, then `tiny`. `ggml-tiny.bin` (77 MB) is the smallest starting point; `ggml-base.bin` (145 MB) and `ggml-small.bin` (494 MB) are more accurate and slower. Models are downloaded from the whisper.cpp model repository, never by the app.
- **Bundled whisper.cpp.** The keyboard always runs the whisper helpers installed next to its own binary — never a system or `PATH` copy. `third_party/whisper.cpp` records the pinned upstream revision, license and local pruning.
- **GPU acceleration.** When the package was built on a host with a Vulkan toolchain (`glslc` and Vulkan headers, `-DGGML_VULKAN=ON`), inference is offloaded to the Frame's Adreno GPU through Turnip and a short dictation takes seconds instead of tens of seconds; otherwise the same binary runs on the CPU — roughly 5 s per short phrase with `ggml-base`, 15–30 s with `ggml-small` and about 2 s with `ggml-tiny`. The app sets `VK_LOADER_LAYERS_DISABLE=~implicit~` for itself and the helpers so the gaming session's implicit Vulkan layers (the Steam overlay stack) cannot break ggml's Vulkan backend or the panel's own device.

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

## Warm server (opt-in)

By default every clip runs the one-shot `whisper-cli` beside the keyboard binary: it loads the model, transcribes, and exits, freeing its memory — including its GPU memory — after each dictation. A resident `whisper-server` can skip the model load between clips (about a second on the GPU), but it holds a Vulkan context and its GPU buffers for as long as the keyboard is up, and on the Frame a resident server's request has been observed to fault the GPU while the VR session was rendering (the keyboard's own Vulkan device went `VK_ERROR_DEVICE_LOST` and the VR session restarted). The server is therefore opt-in:

```sh
FRAMEKEYBOARD_WHISPER_SERVER=~/.local/share/framekeyboard/current/bin/whisper-server \
  ~/.local/bin/framekeyboard
```

If you opt in: the server starts when the keyboard first appears, is kept across clips, listens on a random loopback port on `127.0.0.1` (clips are POSTed in memory, never written to disk on this path, nothing leaves the Frame), stops shortly after the keyboard hides, and also stops by itself if the keyboard process dies. If it keeps failing, later clips silently fall back to the one-shot CLI. `FRAMEKEYBOARD_WHISPER_SERVER_IDLE` sets how many seconds (0–3600, default 180) it stays up after the keyboard hides.

## Overrides

All optional; the defaults are what the model download above gives you. They are read at startup and on **Reload config**, so set them in the environment the app runs in and relaunch or reload.

| Variable | Default | Meaning |
| --- | --- | --- |
| `FRAMEKEYBOARD_WHISPER_CLI` | bundled `whisper-cli` | Command to run, e.g. a full path or wrapper. A missing file disables the mic key. |
| `FRAMEKEYBOARD_WHISPER_MODEL` | best `ggml-*.bin` in the models directory | Exact model file; a missing file disables the mic key. |
| `FRAMEKEYBOARD_WHISPER_LANGUAGE` | `auto` | Language code such as `en` or `de` to skip detection. |
| `FRAMEKEYBOARD_WHISPER_THREADS` | `4` | Inference threads, 1–32. |
| `FRAMEKEYBOARD_WHISPER_SERVER` | unset: one-shot CLI | Opt-in warm server; set it to the server command to enable, or to the empty value to force the CLI (see [Warm server (opt-in)](#warm-server-opt-in)). |
| `FRAMEKEYBOARD_WHISPER_SERVER_IDLE` | `180` | Seconds (0–3600) the server stays up after the keyboard hides. |

Detection order for the command: `FRAMEKEYBOARD_WHISPER_CLI`, then the `whisper-cli` installed beside the keyboard binary; there is no `PATH` search.

## Privacy and limits

- Audio is captured only between the two presses and lives only in memory. On the warm-server path it never touches disk at all; on the one-shot CLI path it exists only as one private WAV file (`0600` in a `0700` directory under `$XDG_CACHE_HOME/framekeyboard`) that is removed immediately after each run. Nothing is uploaded: recognition is entirely on-device, and the server's port is loopback-only.
- Transcripts, typed text, and errors are never logged. Results pass through the same text-output queue as typed characters, so they are never sent when the backend cannot deliver (a preview launch reports that the transcription was discarded instead).
- Results are treated as untrusted text: invalid UTF-8, control characters, non-speech annotations like `[Music]`, decoder repetition loops, and implausibly long results for the clip's length are removed or rejected with a fixed message.
- The Frame's microphone must be permitted by the system; capture opens the default recording device through SDL's PipeWire backend. If no usable 16 kHz mono capture device exists, the mic key reports the failure and keeps working as unpressed.
- Quality with `ggml-tiny` is limited, like any tiny Whisper model; install `base` or `small` for better results. The mic key appears in all six bundled layouts; custom layouts opt in with a key of `"action": {"kind": "whisper", "value": "toggle"}` and `"icon": "mic"` (see [configuration](configuration.md)).
