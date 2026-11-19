/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "downgrade/spill_policy.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t mib = 1024 * 1024;
constexpr std::array<std::size_t, 4> candidate_mib{1536, 1024, 384, 256};

void check(cudaError_t status)
{
  if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
}

class copy_fixture {
 public:
  copy_fixture()
  {
    try {
      check(cudaEventCreate(&start_));
      check(cudaEventCreate(&stop_));
      check(cudaMallocHost(&host_, candidate_mib.front() * mib));
      for (std::size_t i = 0; i < device_.size(); ++i) {
        check(cudaMalloc(&device_[i], candidate_mib[i] * mib));
        check(cudaMemset(device_[i], static_cast<int>(i + 1), candidate_mib[i] * mib));
      }
      check(cudaDeviceSynchronize());
    } catch (...) {
      release();
      throw;
    }
  }

  ~copy_fixture() { release(); }
  copy_fixture(const copy_fixture&)            = delete;
  copy_fixture& operator=(const copy_fixture&) = delete;

  float copy(std::size_t i, cudaMemcpyKind direction)
  {
    auto* dst       = direction == cudaMemcpyDeviceToHost ? host_ : device_[i];
    auto const* src = direction == cudaMemcpyDeviceToHost ? device_[i] : host_;
    check(cudaEventRecord(start_));
    check(cudaMemcpyAsync(dst, src, candidate_mib[i] * mib, direction));
    check(cudaEventRecord(stop_));
    check(cudaEventSynchronize(stop_));
    float elapsed_ms = 0;
    check(cudaEventElapsedTime(&elapsed_ms, start_, stop_));
    return elapsed_ms;
  }

 private:
  void release()
  {
    for (auto* buffer : device_) {
      if (buffer) { cudaFree(buffer); }
    }
    if (host_) { cudaFreeHost(host_); }
    if (start_) { cudaEventDestroy(start_); }
    if (stop_) { cudaEventDestroy(stop_); }
  }

  std::array<void*, candidate_mib.size()> device_{};
  void* host_{};
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

std::vector<std::size_t> production_order(std::size_t deficit_mib, bool byte_targeted)
{
  const std::vector<std::size_t> sizes(candidate_mib.begin(), candidate_mib.end());
  std::vector<bool> dispatched(sizes.size());
  std::vector<std::size_t> picks;
  std::size_t freed_mib = 0;
  while (freed_mib < deficit_mib) {
    auto remaining = byte_targeted ? std::optional{deficit_mib - freed_mib} : std::nullopt;
    auto index =
      sirius::parallel::select_next_spill_candidate(sizes, dispatched, remaining).value();
    dispatched[index] = true;
    picks.push_back(index);
    freed_mib += sizes[index];
  }
  return picks;
}

// An offline lower bound on bytes for this four-batch example.
std::vector<std::size_t> minimum_bytes(std::size_t deficit_mib)
{
  std::vector<std::size_t> best;
  std::size_t best_size = std::numeric_limits<std::size_t>::max();
  for (std::size_t mask = 1; mask < (1U << candidate_mib.size()); ++mask) {
    std::vector<std::size_t> picks;
    std::size_t size = 0;
    for (std::size_t i = 0; i < candidate_mib.size(); ++i) {
      if (mask & (1U << i)) {
        picks.push_back(i);
        size += candidate_mib[i];
      }
    }
    if (size >= deficit_mib &&
        (size < best_size || (size == best_size && picks.size() < best.size()))) {
      best      = std::move(picks);
      best_size = size;
    }
  }
  return best;
}

}  // namespace

int main()
{
  try {
    copy_fixture fixture;
    for (std::size_t i = 0; i < candidate_mib.size(); ++i) {
      fixture.copy(i, cudaMemcpyDeviceToHost);
      fixture.copy(i, cudaMemcpyHostToDevice);
    }

    constexpr std::array policies{"newest_first", "best_fit", "minimum_bytes"};
    constexpr std::array<std::size_t, 12> deficits{
      128, 256, 257, 384, 385, 640, 641, 1024, 1025, 1600, 1800, 2800};
    constexpr std::size_t repetitions = 5;
    std::cout << "deficit_mib,policy,trial,batches,spilled_mib,spill_ms,restore_ms\n";
    for (auto deficit : deficits) {
      const std::array plans{
        production_order(deficit, false), production_order(deficit, true), minimum_bytes(deficit)};
      for (std::size_t trial = 0; trial < repetitions; ++trial) {
        for (std::size_t offset = 0; offset < policies.size(); ++offset) {
          auto policy             = (trial + offset) % policies.size();
          std::size_t spilled_mib = 0;
          float spill_ms          = 0;
          float restore_ms        = 0;
          for (auto i : plans[policy]) {
            spilled_mib += candidate_mib[i];
            spill_ms += fixture.copy(i, cudaMemcpyDeviceToHost);
            restore_ms += fixture.copy(i, cudaMemcpyHostToDevice);
          }
          std::cout << deficit << ',' << policies[policy] << ',' << trial << ','
                    << plans[policy].size() << ',' << spilled_mib << ',' << spill_ms << ','
                    << restore_ms << '\n';
        }
      }
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
