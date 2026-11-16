// Standalone operator experiment: no SQL rewrites and no pre-aggregation.
#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>

#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#define CUDA(x)                                                            \
  do {                                                                     \
    auto e = (x);                                                          \
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); \
  } while (0)
using I     = int64_t;
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t)
{
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
template <class T>
struct Buffer {
  rmm::device_uvector<T> storage;
  T* p;
  size_t n;
  explicit Buffer(size_t count)
    : storage(std::max(size_t(1), count), cudf::get_default_stream()), p(storage.data()), n(count)
  {
  }
  Buffer(Buffer const&) = delete;
  void upload(std::vector<T> const& v)
  {
    CUDA(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> host()
  {
    std::vector<T> v(n);
    CUDA(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }
};
__device__ unsigned bucket(int k, unsigned mask)
{
  unsigned x = k;
  x ^= x >> 16;
  x *= 0x7feb352d;
  x ^= x >> 15;
  x *= 0x846ca68b;
  x ^= x >> 16;
  return x & mask;
}
__global__ void build(int const* keys, int n, int* heads, int* next, unsigned mask)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) next[i] = atomicExch(heads + bucket(keys[i], mask), i);
}
// Every matching pair is visited, including duplicates; sums happen after matching.
// Small group sets use block-local post-join accumulation; larger sets use global atomics.
__global__ void fused(int const* pk,
                      int const* pg,
                      I const* val,
                      int n,
                      int const* bk,
                      int const* bg,
                      int const* heads,
                      int const* next,
                      unsigned mask,
                      int bg_count,
                      int groups,
                      unsigned long long* out,
                      bool local)
{
  extern __shared__ unsigned long long sums[];
  if (local)
    for (int g = threadIdx.x; g < groups; g += blockDim.x)
      sums[g] = 0;
  __syncthreads();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    for (int j = heads[bucket(pk[i], mask)]; j != -1; j = next[j])
      if (pk[i] == bk[j]) {
        int g = pg[i] * bg_count + bg[j];
        atomicAdd(local ? sums + g : out + g, static_cast<unsigned long long>(val[i]));
      }
  __syncthreads();
  if (local)
    for (int g = threadIdx.x; g < groups; g += blockDim.x)
      if (sums[g]) atomicAdd(out + g, sums[g]);
}
// Controlled materialized baseline shares hash layout, pair visits and reducer with fused.
// Slots per probe are an explicit upper bound, NOT a compact cuDF gather-map implementation.
__global__ void materialize(int const* pk,
                            int const* pg,
                            I const* val,
                            int n,
                            int const* bk,
                            int const* bg,
                            int const* heads,
                            int const* next,
                            unsigned mask,
                            int bg_count,
                            int slots,
                            int* og,
                            I* ov)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int s = 0;
  for (int j = heads[bucket(pk[i], mask)]; j != -1; j = next[j])
    if (pk[i] == bk[j]) {
      size_t pos = size_t(i) * slots + s++;
      og[pos]    = pg[i] * bg_count + bg[j];
      ov[pos]    = val[i];
    }
  for (; s < slots; s++)
    og[size_t(i) * slots + s] = -1;
}
__global__ void aggregate(int const* g, I const* v, size_t n, int groups, unsigned long long* out)
{
  extern __shared__ unsigned long long sums[];
  bool local = groups <= 128;
  if (local)
    for (int k = threadIdx.x; k < groups; k += blockDim.x)
      sums[k] = 0;
  __syncthreads();
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n && g[i] >= 0)
    atomicAdd(local ? sums + g[i] : out + g[i], static_cast<unsigned long long>(v[i]));
  __syncthreads();
  if (local)
    for (int k = threadIdx.x; k < groups; k += blockDim.x)
      if (sums[k]) atomicAdd(out + k, sums[k]);
}
__global__ void combine(int const* pg, int const* bg, int* group, int n, int bg_count)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) group[i] = pg[i] * bg_count + bg[i];
}
__global__ void expressions(int const* bg, I const* v, int n, I* low, I* high, bool scalar)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    low[i]  = scalar ? v[i] : (bg[i] == 0 ? v[i] : 0);
    high[i] = bg[i] != 0 ? v[i] : 0;
  }
}
struct Input {
  int b, n, bg_count, pg_count, slots;
  uint64_t matches;
  std::vector<int> bk, bg, pk, pg;
  std::vector<I> value, expected;
  template <class T>
  void read(std::ifstream& f, std::vector<T>& v, size_t n)
  {
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(T));
    if (!f) throw std::runtime_error("truncated input");
  }
  explicit Input(char const* path)
  {
    std::ifstream f(path, std::ios::binary);
    int h[5];
    f.read(reinterpret_cast<char*>(h), sizeof h);
    if (!f) throw std::runtime_error("missing input");
    b        = h[0];
    n        = h[1];
    bg_count = h[2];
    pg_count = h[3];
    slots    = h[4];
    if (b <= 0 || n <= 0 || bg_count <= 0 || pg_count <= 0 || slots <= 0)
      throw std::runtime_error("invalid input header");
    f.read(reinterpret_cast<char*>(&matches), sizeof matches);
    read(f, bk, b);
    read(f, bg, b);
    read(f, pk, n);
    read(f, pg, n);
    read(f, value, n);
    read(f, expected, bg_count * pg_count);
  }
};
cudf::column_view col(int const* p, int n)
{
  return {cudf::data_type{cudf::type_id::INT32}, n, p, nullptr, 0};
}
cudf::column_view col(I const* p, int n)
{
  return {cudf::data_type{cudf::type_id::INT64}, n, p, nullptr, 0};
}
struct Timer {
  cudaEvent_t a, b;
  Clock::time_point wall;
  Timer()
  {
    CUDA(cudaEventCreate(&a));
    CUDA(cudaEventCreate(&b));
  }
  ~Timer()
  {
    cudaEventDestroy(a);
    cudaEventDestroy(b);
  }
  void start()
  {
    CUDA(cudaDeviceSynchronize());
    wall = Clock::now();
    CUDA(cudaEventRecord(a));
  }
  std::pair<double, double> stop()
  {
    CUDA(cudaEventRecord(b));
    CUDA(cudaEventSynchronize(b));
    float ms;
    CUDA(cudaEventElapsedTime(&ms, a, b));
    return {ms, elapsed(wall)};
  }
};
int main(int argc, char** argv)
{
  try {
    if (argc < 3) throw std::runtime_error("usage: bench input.bin repetitions [method|all]");
    Input h(argv[1]);
    int reps         = std::stoi(argv[2]);
    std::string only = argc > 3 ? argv[3] : "all";
    int groups       = h.bg_count * h.pg_count;
    bool q14         = std::string(argv[1]).find("q14_") != std::string::npos;
    bool q12         = std::string(argv[1]).find("q12_") != std::string::npos;
    Buffer<int> bk(h.b), bg(h.b), pk(h.n), pg(h.n);
    Buffer<I> val(h.n);
    bk.upload(h.bk);
    bg.upload(h.bg);
    pk.upload(h.pk);
    pg.upload(h.pg);
    val.upload(h.value);
    rmm::mr::cuda_memory_resource upstream;
    rmm::mr::pool_memory_resource pool(upstream, 256 * 1024 * 1024);
    bool measure_memory = argc > 4;
    rmm::mr::statistics_resource_adaptor stats(pool);
    if (measure_memory)
      rmm::mr::set_current_device_resource(stats);
    else
      rmm::mr::set_current_device_resource(pool);
    std::vector<std::string> methods = {"cudf", "materialized", "fused", "fused_global"};
    if (only != "all") methods = {only};
    std::mt19937 rng(20261115);
    std::cout << "method,iteration,build_gpu_ms,probe_gpu_ms,build_wall_ms,probe_wall_ms,matches,"
                 "logical_intermediate_bytes,peak_operator_bytes\n";
    for (int iteration = -3; iteration < reps; iteration++) {
      std::shuffle(methods.begin(), methods.end(), rng);
      for (auto const& method : methods) {
        if (measure_memory) stats.push_counters();
        Timer timer;
        std::pair<double, double> bt, pt;
        std::vector<I> result(groups, 0);
        size_t intermediate = 0;
        if (method == "cudf") {
          timer.start();
          auto join = std::make_unique<cudf::hash_join>(cudf::table_view{{col(bk.p, h.b)}},
                                                        cudf::null_equality::UNEQUAL);
          bt        = timer.stop();
          timer.start();
          auto maps = join->inner_join(cudf::table_view{{col(pk.p, h.n)}});
          if (maps.first->size() != h.matches) throw std::runtime_error("match count mismatch");
          int m      = maps.first->size();
          auto probe = cudf::gather(cudf::table_view{{col(pg.p, h.n), col(val.p, h.n)}},
                                    col(maps.first->data(), m));
          auto dim = cudf::gather(cudf::table_view{{col(bg.p, h.b)}}, col(maps.second->data(), m));
          if (q14 || q12) {
            Buffer<I> low(m), high(m);
            expressions<<<(m + 255) / 256, 256>>>(dim->view().column(0).data<int>(),
                                                  probe->view().column(1).data<I>(),
                                                  m,
                                                  low.p,
                                                  high.p,
                                                  q14);
            if (q14) {
              auto sum   = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
              auto total = cudf::reduce(col(low.p, m), *sum, cudf::data_type{cudf::type_id::INT64});
              auto promo =
                cudf::reduce(col(high.p, m), *sum, cudf::data_type{cudf::type_id::INT64});
              pt        = timer.stop();
              auto t    = static_cast<cudf::numeric_scalar<I>*>(total.get())->value();
              auto p    = static_cast<cudf::numeric_scalar<I>*>(promo.get())->value();
              result[0] = t - p;
              result[1] = p;
            } else {
              cudf::groupby::groupby gb(cudf::table_view{{probe->view().column(0)}});
              std::vector<cudf::groupby::aggregation_request> req(2);
              req[0].values = col(low.p, m);
              req[1].values = col(high.p, m);
              for (auto& r : req)
                r.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
              auto output = gb.aggregate(req);
              pt          = timer.stop();
              int count   = output.first->num_rows();
              std::vector<int> kg(count);
              std::vector<I> sums(count);
              CUDA(cudaMemcpy(kg.data(),
                              output.first->view().column(0).data<int>(),
                              count * 4,
                              cudaMemcpyDeviceToHost));
              for (int j = 0; j < 2; j++) {
                CUDA(cudaMemcpy(sums.data(),
                                output.second[j].results[0]->view().data<I>(),
                                count * 8,
                                cudaMemcpyDeviceToHost));
                for (int i = 0; i < count; i++)
                  result.at(kg[i] * 2 + j) = sums[i];
              }
            }
            intermediate = size_t(m) * 40;
          } else {
            Buffer<int> keys(m);
            combine<<<(m + 255) / 256, 256>>>(probe->view().column(0).data<int>(),
                                              dim->view().column(0).data<int>(),
                                              keys.p,
                                              m,
                                              h.bg_count);
            cudf::groupby::groupby gb(cudf::table_view{{col(keys.p, m)}});
            std::vector<cudf::groupby::aggregation_request> requests(1);
            requests[0].values = probe->view().column(1);
            requests[0].aggregations.push_back(
              cudf::make_sum_aggregation<cudf::groupby_aggregation>());
            auto output = gb.aggregate(requests);
            pt          = timer.stop();
            int count   = output.first->num_rows();
            std::vector<int> kg(count);
            std::vector<I> sums(count);
            CUDA(cudaMemcpy(kg.data(),
                            output.first->view().column(0).data<int>(),
                            count * 4,
                            cudaMemcpyDeviceToHost));
            CUDA(cudaMemcpy(sums.data(),
                            output.second[0].results[0]->view().data<I>(),
                            count * 8,
                            cudaMemcpyDeviceToHost));
            for (int i = 0; i < count; i++)
              result.at(kg[i]) = sums[i];
            intermediate = size_t(m) * 28;
          }
        } else {
          unsigned buckets = 1;
          while (buckets < unsigned(h.b) * 2)
            buckets *= 2;
          timer.start();
          Buffer<int> heads(buckets), next(h.b);
          CUDA(cudaMemsetAsync(heads.p, 255, buckets * 4));
          build<<<(h.b + 255) / 256, 256>>>(bk.p, h.b, heads.p, next.p, buckets - 1);
          bt = timer.stop();
          timer.start();
          Buffer<unsigned long long> out(groups);
          CUDA(cudaMemsetAsync(out.p, 0, groups * 8));
          if (method == "materialized") {
            size_t m = size_t(h.n) * h.slots;
            Buffer<int> og(m);
            Buffer<I> ov(m);
            materialize<<<(h.n + 255) / 256, 256>>>(pk.p,
                                                    pg.p,
                                                    val.p,
                                                    h.n,
                                                    bk.p,
                                                    bg.p,
                                                    heads.p,
                                                    next.p,
                                                    buckets - 1,
                                                    h.bg_count,
                                                    h.slots,
                                                    og.p,
                                                    ov.p);
            aggregate<<<(m + 255) / 256, 256, groups <= 128 ? groups * 8 : 0>>>(
              og.p, ov.p, m, groups, out.p);
            pt           = timer.stop();
            intermediate = m * 12;
          } else {
            bool local = method != "fused_global" && groups <= 128;
            fused<<<(h.n + 255) / 256, 256, local ? groups * 8 : 0>>>(pk.p,
                                                                      pg.p,
                                                                      val.p,
                                                                      h.n,
                                                                      bk.p,
                                                                      bg.p,
                                                                      heads.p,
                                                                      next.p,
                                                                      buckets - 1,
                                                                      h.bg_count,
                                                                      groups,
                                                                      out.p,
                                                                      local);
            pt = timer.stop();
          }
          auto host = out.host();
          for (int i = 0; i < groups; i++)
            result[i] = static_cast<I>(host[i]);
        }
        CUDA(cudaGetLastError());
        if (result != h.expected) {
          for (int i = 0; i < groups; i++)
            if (result[i] != h.expected[i])
              std::cerr << i << ":" << result[i] << " != " << h.expected[i] << "\n";
          throw std::runtime_error("incorrect aggregate: " + method);
        }
        int64_t peak_bytes = measure_memory ? stats.pop_counters().first.peak : 0;
        if (iteration >= 0)
          std::cout << method << ',' << iteration << ',' << bt.first << ',' << pt.first << ','
                    << bt.second << ',' << pt.second << ',' << h.matches << ',' << intermediate
                    << ',' << peak_bytes << '\n';
      }
    }
    rmm::mr::set_current_device_resource(upstream);
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
