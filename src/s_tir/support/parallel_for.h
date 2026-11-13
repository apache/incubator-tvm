/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file parallel_for.h
 * \brief An implementation to run loop in parallel.
 */
#ifndef TVM_S_TIR_SUPPORT_PARALLEL_FOR_H_
#define TVM_S_TIR_SUPPORT_PARALLEL_FOR_H_

#include <tvm/base.h>
#include <tvm/ffi/error.h>
#include <tvm/runtime/base.h>

#include <atomic>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace tvm {
namespace support {

using PartitionerFuncType = std::function<std::vector<std::vector<int>>(int, int, int, int)>;

/*!
 * \brief A partitioner to split the task to each thread in Round-robin manner.
 * \param begin The start index of this parallel loop(inclusive).
 * \param end The end index of this parallel loop(exclusive).
 * \param step The traversal step to the index.
 * \param num_threads The number of threads(the number of tasks to be partitioned to).
 * \return A list with `num_threads` elements, and each is a list of integers indicating the loop
 * indexes for the corresponding thread to process.
 */
TVM_DLL inline std::vector<std::vector<int>> rr_partitioner(int begin, int end, int step,
                                                            int num_threads) {
  int total_task_count = (end - begin) / step;
  TVM_FFI_ICHECK_GE(total_task_count, 0)
      << "Infinite loop condition with begin: " << begin << " end: " << end << " step: " << step;
  std::vector<std::vector<int>> ret;
  ret.reserve(num_threads);
  for (size_t thread = 0; begin < end; begin += step, thread = (thread + 1) % num_threads) {
    if (thread >= ret.size()) {
      ret.push_back(std::vector<int>());
    }
    ret[thread].push_back(begin);
  }
  return ret;
}

/*!
 * \brief A runtime api provided to run the task function in parallel.
 * e.g. A for loop:
 *   for (int i = 0; i < 10; i++) {
 *     a[i] = i;
 *   }
 * should work the same as:
 *   parallel_for(0, 10, [&a](int index) {
 *     a[i] = i;
 *   });
 * \param begin The start index of this parallel loop(inclusive).
 * \param end The end index of this parallel loop(exclusive).
 * \param f The task function to be executed. Assert to take an int index as input with no output.
 * \param step The traversal step to the index.
 * \param partitioner A partition function to split tasks to different threads. Use Round-robin
 * partitioner by default.
 * \note 1. Currently do not support nested parallel_for; 2. The order of execution in each thread
 * is not guaranteed, the for loop task should be thread independent and thread safe.
 */
TVM_DLL inline void parallel_for(int begin, int end, const std::function<void(int)>& f,
                                 int step = 1,
                                 const PartitionerFuncType partitioner = rr_partitioner) {
  static bool GLOBAL_PARALLEL_FOR_FLAG{false};
  static std::mutex M_GLOBAL_PARALLEL_FOR_FLAG;
  {
    std::unique_lock<std::mutex> l(M_GLOBAL_PARALLEL_FOR_FLAG);
    TVM_FFI_ICHECK(!GLOBAL_PARALLEL_FOR_FLAG)
        << "There's another parallel_for running. Maybe you're "
        << "currently inside another parallel_for loop.";
    GLOBAL_PARALLEL_FOR_FLAG = true;
  }

  int default_num_threads = std::thread::hardware_concurrency();
  const auto& run_partitions = partitioner(begin, end, step, default_num_threads);

  std::vector<std::thread> threads;
  threads.reserve(run_partitions.size());
  std::vector<std::future<void>> res_vec;
  res_vec.reserve(run_partitions.size());
  for (const auto& run_partition : run_partitions) {
    std::packaged_task<void(const std::vector<int>&, const std::function<void(int)>&)> task(
        [](const std::vector<int>& run_partition, const std::function<void(int)>& f) {
          for (const auto& i : run_partition) {
            f(i);
          }
        });
    res_vec.emplace_back(task.get_future());
    threads.emplace_back(std::move(task), run_partition, f);
  }

  for (auto&& thread : threads) {
    thread.join();
  }
  {
    std::unique_lock<std::mutex> l(M_GLOBAL_PARALLEL_FOR_FLAG);
    TVM_FFI_ICHECK(GLOBAL_PARALLEL_FOR_FLAG);
    GLOBAL_PARALLEL_FOR_FLAG = false;
  }
  try {
    for (auto&& i : res_vec) {
      i.get();
    }
  } catch (const std::exception& e) {
    TVM_FFI_THROW(InternalError) << "Parallel_for error with " << e.what();
  }
}

/*!
 * \brief An API to launch fix amount of threads to run the specific functor in parallel.
 * Different from `parallel_for`, the partition is determined dynamically on the fly,
 * i.e. any time when a thread is idle, it fetches the next task to run.
 * The behavior is similar to dynamic scheduling in OpenMP:
 *
 *   \#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
 *   for (int i = 0; i < 10; i++) {
 *     a[i] = i;
 *   }
 *
 * \param begin The start index of this parallel loop (inclusive).
 * \param end The end index of this parallel loop (exclusive).
 * \param num_threads The number of threads to be used.
 * \param f The task function to be executed. Takes the thread index and the task index as
 * input with no output.
 * \note `step` support is left for future work.
 */
TVM_DLL inline void parallel_for_dynamic(int begin, int end, int num_threads,
                                         const std::function<void(int thread_id, int task_id)>& f) {
  // Step 1. Sanity checks
  if (begin == end) {
    return;
  }
  TVM_FFI_CHECK_LE(begin, end, ValueError) << "The interval [begin, end) requires `begin <= end`";
  TVM_FFI_CHECK_GT(num_threads, 0, ValueError) << "`num_threads` should be positive";
  // Step 2. Launch threads
  // Step 2.1. Launch worker 1 to worker `num_threads - 1`
  std::atomic<int> counter{begin};
  std::vector<std::future<void>> futures;
  std::vector<std::thread> threads;
  futures.reserve(num_threads - 1);
  threads.reserve(num_threads - 1);
  auto worker = [end, &counter, &f](int thread_id) -> void {
    for (int task_id; (task_id = counter++) < end;) {
      f(thread_id, task_id);
    }
  };
  for (int thread_id = 1; thread_id < num_threads; ++thread_id) {
    std::packaged_task<void(int)> task(worker);
    futures.emplace_back(task.get_future());
    threads.emplace_back(std::move(task), thread_id);
  }
  // Step 2.2. Launch worker 0 inplace
  try {
    worker(0);
  } catch (const std::exception& e) {
    for (auto&& thread : threads) {
      thread.join();
    }
    TVM_FFI_THROW(RuntimeError) << "parallel_for_dynamic error with " << e.what();
  }
  // Step 3. Join threads and check exceptions
  for (auto&& thread : threads) {
    thread.join();
  }
  try {
    for (auto&& future : futures) {
      future.get();
    }
  } catch (const std::exception& e) {
    TVM_FFI_THROW(RuntimeError) << "parallel_for_dynamic error with " << e.what();
  }
}

}  // namespace support
}  // namespace tvm

#endif  // TVM_S_TIR_SUPPORT_PARALLEL_FOR_H_
