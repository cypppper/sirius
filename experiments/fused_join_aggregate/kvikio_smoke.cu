// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include <kvikio/file_handle.hpp>

#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv)
{
  if (argc != 2) return 2;
  constexpr size_t n = 4096;
  std::vector<char> expected(n), host(n), device_result(n);
  std::ifstream reference(argv[1], std::ios::binary);
  reference.read(expected.data(), n);
  if (!reference) throw std::runtime_error("reference read failed");
  kvikio::FileHandle file(argv[1], "r", kvikio::FileHandle::m644, kvikio::CompatMode::ON);
  if (file.pread(host.data(), n, 0).get() != n || host != expected)
    throw std::runtime_error("KvikIO host read mismatch");
  char* device = nullptr;
  cudaStream_t stream;
  if (cudaMalloc(&device, n) != cudaSuccess || cudaStreamCreate(&stream) != cudaSuccess)
    throw std::runtime_error("CUDA allocation failed");
  cudaMemsetAsync(device, 0, n, stream);
  auto pending = file.read_async(device, n, 0, 0, stream);
  if (pending.check_bytes_done() != n) throw std::runtime_error("short GPU read");
  if (cudaMemcpy(device_result.data(), device, n, cudaMemcpyDeviceToHost) != cudaSuccess ||
      device_result != expected)
    throw std::runtime_error("KvikIO device read mismatch");
  cudaStreamDestroy(stream);
  cudaFree(device);
  std::cout << "KvikIO CompatMode::ON: host and stream-ordered GPU reads match reference\n";
}
