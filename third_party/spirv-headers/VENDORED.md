# SPIRV-Headers (vendored)

KhronosGroup/SPIRV-Headers, tag `vulkan-sdk-1.4.321.0`, commit
`2a611a970fdbc41ac2e3e328802aed9985352dca`
(https://github.com/KhronosGroup/SPIRV-Headers), under the MIT-style Khronos
license in `LICENSE` and `LICENSES/`.

Header-only companion for the vendored whisper.cpp Vulkan backend, which does
`find_package(SPIRV-Headers CONFIG REQUIRED)`. The top-level project installs
this package into a build-tree prefix at configure time and adds that prefix
to `CMAKE_PREFIX_PATH`; no files were modified, `tests/`, `tools/`,
`examples/`, `docs/` and CI files were removed from the archive.
