#ifndef MEDS_TAB_PARALLEL_H_
#define MEDS_TAB_PARALLEL_H_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace meds_tab {

// The number of threads ParallelFor uses for `size` items.
inline std::size_t WorkerCount(int threads, std::size_t size) {
  return std::max<std::size_t>(1, std::min<std::size_t>(threads, size));
}

// Runs function(worker, i) for every i in [0, size) on up to `threads` threads.
// Items are claimed in order, one at a time. worker identifies the executing
// thread (the caller is worker 0), so callers can give each worker its own
// scratch. The first exception stops further claims; every thread is joined and
// the exception is rethrown.
template <typename Function>
void ParallelFor(std::size_t size, int threads, Function function) {
  std::atomic<std::size_t> next{0};
  std::atomic<bool> failed{false};
  std::exception_ptr error;
  std::mutex error_mutex;
  const auto run = [&](std::size_t worker) {
    try {
      while (!failed.load(std::memory_order_relaxed)) {
        const auto index = next.fetch_add(1, std::memory_order_relaxed);
        if (index >= size) return;
        function(worker, index);
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(error_mutex);
      if (!error) error = std::current_exception();
      failed.store(true, std::memory_order_relaxed);
    }
  };
  std::vector<std::thread> workers;
  try {
    for (std::size_t i = 1; i < WorkerCount(threads, size); ++i) workers.emplace_back(run, i);
    run(0);
  } catch (...) {
    // Thread creation failed: stop the started workers before propagating.
    failed.store(true, std::memory_order_relaxed);
    for (auto& worker : workers) worker.join();
    throw;
  }
  for (auto& worker : workers) worker.join();
  if (error) std::rethrow_exception(error);
}

}  // namespace meds_tab
#endif
