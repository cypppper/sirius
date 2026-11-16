#!/usr/bin/env bash
# Build the actual Sirius library and DuckDB extension with the research I/O patch.
set -euo pipefail
cd "$(dirname "$0")/../.."
patch_file="$PWD/experiments/fused_join_aggregate/cucascade-kvikio.patch"
if ! git -C cucascade apply --reverse --check "$patch_file" 2>/dev/null; then
  git -C cucascade apply --check "$patch_file"
  git -C cucascade apply "$patch_file"
fi
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-6}"
make release TEST_BUILD_TARGET= \
  SIRIUS_CMAKE_FLAGS="-DSIRIUS_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=${SIRIUS_RESEARCH_CUDA_ARCH:-120-real} -DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DCMAKE_CUDA_COMPILER_LAUNCHER=sccache -DCMAKE_LINKER_TYPE=MOLD"
cmake --install build/release --prefix "$PWD/build/release/install" --component sirius_library
make -C sirius-duckdb release SIRIUS_DUCKDB_LINKAGE=shared \
  SIRIUS_INSTALL_DIR="$PWD/build/release/install" BUILD_EXTENSIONS_ONLY=0
