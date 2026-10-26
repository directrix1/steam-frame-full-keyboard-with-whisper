# whisper.cpp (vendored)

whisper.cpp upstream master, commit
`231638a0717b7822784297e5980965ffcec05810`
(https://github.com/ggml-org/whisper.cpp), MIT license (`LICENSE`, `AUTHORS`).

This is the speech transcriber Full Keyboard ships with: the app always uses
this vendored build, installed beside the keyboard binary as `whisper-cli` and
`whisper-server`. No external whisper.cpp installation is used.

## Local changes to the pinned tree

Pruned (directories outside the build were removed; nothing else was edited):

- top level: `bindings/`, `models/`, `media/`, `grammars/`, `ci/`, `scripts/`,
  `samples/`, `tests/`, `.github/`
- `ggml/`: `tests/`, `examples/`
- `ggml/src/`: every backend except `ggml-cpu` and `ggml-vulkan` (`blas`,
  `cann`, `cuda`, `et`, `hip`, `hexagon`, `metal`, `musa`, `opencl`,
  `openvino`, `rpc`, `sycl`, `virtgpu`, `webgpu`, `zdnn`, `zendnn`)
- `examples/`: everything except `cli/` and `server/` plus the shared headers

Patched: `examples/CMakeLists.txt` — removed the unconditional
`add_subdirectory()` calls for the pruned `bench`, `quantize`,
`vad-speech-segments`, `parakeet-cli`, `parakeet-quantize` and
`deprecation-warning` examples. The `WHISPER_SDL2` and Emscripten branches
still reference pruned directories; both must stay disabled.

## Build notes

`GGML_VULKAN=ON` offloads inference to a Vulkan GPU (on the Frame: Turnip on
Adreno) and needs `glslc` and Vulkan headers from the host toolchain;
without them the build falls back to the CPU backend. `SPIRV-Headers` is
vendored next to this directory; the top-level project installs it into a
build-tree prefix and supplies it via `CMAKE_PREFIX_PATH`.
