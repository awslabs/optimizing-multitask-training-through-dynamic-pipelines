/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modifications Copyright (c) Facebook, Inc.
 * See: https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp
 */

#include <c10/cuda/CUDACachingAllocator.h>

#include <torch/csrc/cuda/CUDAPluggableAllocator.h>

#include <c10/core/impl/GPUTrace.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <c10/util/llvmMathExtras.h>

#include <ATen/Context.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include "traceback.h"

#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <bitset>
#include <cstdint>
#include <deque>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <utility>
#include <vector>

namespace dynapipe {
namespace CUDACachingAllocator {

//
// Yet another caching allocator for CUDA device allocations.
//
// - Allocations are associated with a stream. Once freed, blocks can be
//   re-allocated on the same stream, but not on any other stream.
// - The allocator attempts to find the smallest cached block that will fit the
//   requested size. If the block is larger than the requested size, it may be
//   split. If no block is found, the allocator will delegate to cudaMalloc.
// - If the cudaMalloc fails, the allocator will attempt to free one cached
//   block of sufficient size that is not split and retry the allocation.
//   If this also fails, the allocator will attempt to free all cached blocks
//   that are not split and retry the allocation.
// - Large (>1MB) and small allocations are stored in separate pools.
//   Small requests are packed into 2MB buffers. Large requests will use the
//   smallest available free block or allocate a new block using cudaMalloc.
// - To reduce fragmentation, requests between 1MB and 10MB will allocate and
//   split a 20MB block, if no free block of sufficient size is available.
// - To further reduce fragmentation, blocks >= 200MB are not allowed to be
//   split. These oversize cached blocks will still satisfy requests within
//   20MB of the oversize cached block size.
//
// With this allocator, allocations and frees should logically be considered
// "usages" of the memory segment associated with streams, just like kernel
// launches. The programmer must insert the proper synchronization if memory
// segments are used from multiple streams.
//
// The library provides a recordStream() function to help insert the correct
// synchronization when allocations are used on multiple streams. This will
// ensure that the block is not reused before each recorded stream completes
// work.
//

/**
 * Note [Interaction with CUDA graph capture]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Graph capture performs a dry run of a region of execution, freezing all CUDA
 * work (and virtual addresses used during that work) into a "graph." The graph
 * may be "replayed" like a single giant kernel, with greatly reduced CPU
 * overhead as well as modestly improved GPU performance.
 *
 * Because capture bakes in memory addresses, the memory used during capture
 * must be available for the graph to use during replay. DeviceCachingAllocator
 * assigns and frees memory eagerly and dynamically, so if we're not careful
 * about managing graphs' memory, at replay time those memory addresses could be
 * used by other tensors.
 *
 * To guarantee a graph's baked in addresses are safe to reuse in replay,
 * DeviceAllocator satisfies allocations from a graph-private memory pool during
 * capture, and doesn't begin cudaFreeing those addresses until the graph is
 * destroyed.
 *
 * Within the private pool, allocations are freed and reassigned as usual during
 * capture. Memory regions will be used in a consistent order during replay. So
 * a private pool doesn't use memory more wastefully than the default pools
 * during capture, but it does reserve its high-water mark of used memory away
 * from the default pools as long as the capture(s) it served survive
 * (regardless whether those captures are idle or replaying).
 *
 * CUDAGraph's requests for private pools are mediated by
 * DeviceAllocator::notifyCaptureBegin,
 *                  notifyCaptureAboutToEnd,
 *                  notifyCaptureEnded,
 *                  notifyCaptureDestroy.
 */

constexpr size_t kMinBlockSize =
    512; // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 0; // use a single pool
constexpr size_t kSmallBuffer =
    2097152; // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer =
    20971520; // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc =
    10485760; // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge = 2097152; // round up large allocations to 2 MiB
constexpr size_t kRoundUpPowerOfTwoIntervals = 16;

using c10::cuda::MempoolId_t;

// Size pretty-printer
inline std::string format_size(uint64_t size) {
  std::ostringstream os;
  os.precision(2);
  os << std::fixed;
  if (size <= 1024) {
    os << size << " bytes";
  } else if (size <= 1048576) {
    os << (size / 1024.0);
    os << " KiB";
  } else if (size <= 1073741824ULL) {
    os << size / 1048576.0;
    os << " MiB";
  } else {
    os << size / 1073741824.0;
    os << " GiB";
  }
  return os.str();
}

namespace {

using stream_set = ska::flat_hash_set<c10::cuda::CUDAStream>;
using c10::GatheredContext;
using c10::DeviceIndex;
using c10::FreeCudaMemoryCallbacksRegistry;
using c10::cuda::CUDACachingAllocator::StatType;
using c10::cuda::CUDACachingAllocator::Stat;
using c10::cuda::CUDACachingAllocator::StatArray;
using c10::cuda::CUDACachingAllocator::History;
using c10::cuda::CUDACachingAllocator::AllocatorState;
using c10::cuda::CUDACachingAllocator::DeviceStats;
using c10::cuda::CUDACachingAllocator::CreateContextFn;
using c10::cuda::CUDACachingAllocator::TraceEntry;
using c10::cuda::CUDACachingAllocator::OutOfMemoryObserver;
using c10::cuda::CUDACachingAllocator::SegmentInfo;
using c10::cuda::CUDACachingAllocator::BlockInfo;

using c10::cuda::MempoolId_t;

using StatTypes = std::array<bool, static_cast<size_t>(StatType::NUM_TYPES)>;

void update_stat(Stat& stat, int64_t amount) {
  stat.current += amount;

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stat.current >= 0,
      "Negative tracked stat in CUDA allocator (likely logic error).");

  stat.peak = std::max(stat.current, stat.peak);
  if (amount > 0) {
    stat.allocated += amount;
  }
  if (amount < 0) {
    stat.freed += -amount;
  }
}

void reset_accumulated_stat(Stat& stat) {
  stat.allocated = 0;
  stat.freed = 0;
}

void reset_peak_stat(Stat& stat) {
  stat.peak = stat.current;
}

template <typename Func>
void for_each_selected_stat_type(const StatTypes& stat_types, Func f) {
  for (const auto stat_type : c10::irange(stat_types.size())) {
    if (stat_types[stat_type]) {
      f(stat_type);
    }
  }
}

void update_stat_array(
    StatArray& stat_array,
    int64_t amount,
    const StatTypes& stat_types) {
  for_each_selected_stat_type(
      stat_types, [&stat_array, amount](size_t stat_type) {
        update_stat(stat_array[stat_type], amount);
      });
}

struct Block;
struct PrivatePool;
typedef bool (*Comparison)(const Block*, const Block*);

struct BlockPool {
  BlockPool(
      Comparison comparator,
      bool small,
      PrivatePool* private_pool = nullptr)
      : blocks(comparator), is_small(small), owner_PrivatePool(private_pool) {}
  std::set<Block*, Comparison> blocks;
  const bool is_small;
  PrivatePool* owner_PrivatePool;
};

struct HistoryChain {
  History h;
  std::unique_ptr<HistoryChain> next; // when blocks are merged we keep records
                                      // of what used to be in the block
};

struct Block {
  int device; // gpu
  cudaStream_t stream; // allocation stream
  stream_set stream_uses; // streams on which the block was used
  size_t size; // block size in bytes
  size_t requested_size; // memory originally requested
  BlockPool* pool{nullptr}; // owning memory pool
  void* ptr{nullptr}; // memory address
  bool allocated{false}; // in-use flag
  Block* prev{nullptr}; // prev block if split from a larger allocation
  Block* next{nullptr}; // next block if split from a larger allocation
  int event_count{0}; // number of outstanding CUDA events
  int gc_count{0}; // counter for prioritizing older / less useful blocks for
                   // garbage collection
  std::unique_ptr<HistoryChain> history;
  HistoryChain* history_last{nullptr};

  Block(
      int device,
      cudaStream_t stream,
      size_t size,
      BlockPool* pool,
      void* ptr)
      : device(device),
        stream(stream),
        stream_uses(),
        size(size),
        requested_size(0),
        pool(pool),
        ptr(ptr) {}

  // constructor for search key
  Block(int device, cudaStream_t stream, size_t size)
      : device(device),
        stream(stream),
        stream_uses(),
        size(size),
        requested_size(0) {}

  bool is_split() const {
    return (prev != nullptr) || (next != nullptr);
  }
};

// BlockState, BlockPoolState, and PrivatePoolState contain the information
// needed to reconstruct a private pool to a previous state. See note
// [Checkpointing PrivatePoolState]
struct BlockState {
  int device = 0;
  cudaStream_t stream = 0;
  stream_set stream_uses = {};
  size_t size = 0;
  void* ptr = nullptr;
  bool allocated = false;
  int gc_count = 0;
  // maintain invariant that event_count == 0 ;
  // history will be left alone in checkpoint

  BlockState(Block* block);
};

struct SegmentState {
  std::vector<BlockState> blocks;
  bool is_small = false;

  SegmentState(Block* head);
};

struct PrivatePoolState : AllocatorState {
  // omitting use_count, and cudaMalloc_count as they remain the same
  MempoolId_t owner_id = {0, 0};

  std::vector<SegmentState> segments;

  PrivatePoolState(
      MempoolId_t pool_id,
      const std::vector<Block*>& private_pool_head_blocks);
};

struct RestoreResult {
  std::vector<void*> allocations_freed;
  std::vector<Block*> allocations_created;
};

static bool BlockComparator(const Block* a, const Block* b) {
  if (a->stream != b->stream) {
    return (uintptr_t)a->stream < (uintptr_t)b->stream;
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

struct AllocParams {
  AllocParams(
      int device,
      size_t size,
      cudaStream_t stream,
      BlockPool* pool,
      size_t alloc_size,
      DeviceStats& stats)
      : search_key(device, stream, size),
        pool(pool),
        alloc_size(alloc_size),
        block(nullptr),
        err(cudaSuccess) {}

  int device() const {
    return search_key.device;
  }
  cudaStream_t stream() const {
    return search_key.stream;
  }
  size_t size() const {
    return search_key.size;
  }

  Block search_key;
  BlockPool* pool;
  size_t alloc_size;
  Block* block;
  StatTypes stat_types = {false};
  cudaError_t err;
};

int trimHistoryBefore(Block* block, void* point) {
  int n = 0;
  while (block->history && block->history->h.addr < point) {
    block->history = std::move(block->history->next);
    ++n;
  }
  if (!block->history) {
    block->history_last = nullptr;
  }
  return n;
}

// Note: cudaEventCreate when concurrently invoked from multiple threads can be
// very expensive (at least on certain device/driver combinations). Thus, we a)
// serialize event creation at a per-device level, and b) pool the events to
// avoid constantly calling cudaEventCreate/cudaEventDestroy. This results in
// significant improvements in multithreaded workloads with high allocation
// rates.
class EventPool {
 public:
  using Event = std::unique_ptr<cudaEvent_t, std::function<void(cudaEvent_t*)>>;
  // TODO: Explicit device count
  EventPool() : pools_(at::cuda::device_count()) {}

  Event get(int device) {
    TORCH_INTERNAL_ASSERT(0 <= device);
    TORCH_INTERNAL_ASSERT(device < static_cast<int>(pools_.size()));
    auto& pool = pools_[device];
    auto destructor = [&pool](cudaEvent_t* event) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.push_back(std::unique_ptr<cudaEvent_t>(event));
    };

    // Try to acquire an event from the per-device pool.
    {
      std::lock_guard<std::mutex> g(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        auto* event = pool.event_pool_.back().release();
        pool.event_pool_.pop_back();
        return Event(event, destructor);
      }
    }
    // otherwise, allocate a new event that will be returned to the pool on
    // destruction.
    auto new_ptr = std::make_unique<cudaEvent_t>();
    C10_CUDA_CHECK(
        cudaEventCreateWithFlags(new_ptr.get(), cudaEventDisableTiming));

    return Event(new_ptr.release(), destructor);
  }

  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

 private:
  struct PerDevicePool {
    alignas(64) std::mutex mutex_;
    std::vector<std::unique_ptr<cudaEvent_t>> event_pool_;
  };
  std::vector<PerDevicePool> pools_;
};

// CUDA graphs helper
struct PrivatePool {
  PrivatePool()
      : use_count(1),
        cudaMalloc_count(0),
        large_blocks(BlockComparator, /*is_small=*/false, this),
        small_blocks(BlockComparator, /*is_small=*/true, this) {}
  PrivatePool(const PrivatePool&) = delete;
  PrivatePool(PrivatePool&&) = delete;
  PrivatePool& operator=(const PrivatePool&) = delete;
  // Number of live graphs using this pool
  int use_count;
  // Number of unfreed cudaMallocs made for this pool. When use_count and
  // cudaMalloc_count drop to zero, we can delete this PrivatePool from
  // graph_pools.
  int cudaMalloc_count;
  // Instead of maintaining private BlockPools here, I could stuff all blocks
  // (private or no) into the top-level large_blocks and small_blocks, and
  // distinguish private blocks by adding a "pool id" check above the stream
  // check in BlockComparator. BlockComparator is performance- critial though,
  // I'd rather not add more logic to it.
  BlockPool large_blocks;
  BlockPool small_blocks;
};

BlockState::BlockState(Block* block)
    : stream(block->stream),
      stream_uses(block->stream_uses),
      size(block->size),
      ptr(block->ptr),
      allocated(block->allocated),
      gc_count(block->gc_count) {
  TORCH_CHECK(
      block->event_count == 0,
      "Events should have synchronized when checkpointing block");
};

SegmentState::SegmentState(Block* head) {
  TORCH_INTERNAL_ASSERT(head->prev == nullptr && head->pool != nullptr);
  is_small = head->pool->is_small;

  for (Block* curr = head; curr != nullptr; curr = curr->next) {
    blocks.emplace_back(curr);
  }
}

PrivatePoolState::PrivatePoolState(
    MempoolId_t pool_id,
    const std::vector<Block*>& private_pool_head_blocks)
    : owner_id(pool_id) {
  for (Block* head : private_pool_head_blocks) {
    segments.emplace_back(head);
  }
}

struct MempoolIdHash {
  std::size_t operator()(const MempoolId_t& mempool_id) const noexcept {
    return mempool_id.first != 0 ? mempool_id.first : mempool_id.second;
  }
};

cudaError_t cudaMallocMaybeCapturing(void** p, size_t size) {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  if (at::cuda::currentStreamCaptureStatusMayInitCtx() ==
      at::cuda::CaptureStatus::None) {
#endif
    return C10_CUDA_ERROR_HANDLED(cudaMalloc(p, size));
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
  } else {
    // It's ok to capture cudaMallocs, as long as we never cudaFree those
    // addresses before replay.
    // Capturing cudaMalloc behaves nicely: it gives the graph new VA,
    // but is ignored (won't leakily allocate new memory) in replays.
    at::cuda::CUDAStreamCaptureModeGuard g{cudaStreamCaptureModeRelaxed};
    return C10_CUDA_ERROR_HANDLED(cudaMalloc(p, size));
  }
#endif
}

} // anonymous namespace

// Environment config parser
// Defined here, rather than its own .cpp file,
// because parseArgs needs to know kLargeBuffer.
// Defined outside namespace Native because it's not Native-specific.
class CachingAllocatorConfig {
 public:
  static size_t max_split_size() {
    return instance().m_max_split_size;
  }
  static double garbage_collection_threshold() {
    return instance().m_garbage_collection_threshold;
  }

  // This is used to round-up allocation size to nearest power of 2 divisions.
  // More description below in function roundup_power2_next_division
  // As ane example, if we want 4 divisions between 2's power, this can be done
  // using env variable: PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:4
  static size_t roundup_power2_divisions(size_t size) {
    size_t log_size = (63 - c10::llvm::countLeadingZeros(size));

    // Our intervals start at 1MB and end at 64GB
    const size_t interval_start =
        63 - c10::llvm::countLeadingZeros(static_cast<size_t>(1048576));
    const size_t interval_end =
        63 - c10::llvm::countLeadingZeros(static_cast<size_t>(68719476736));
    TORCH_CHECK(
        (interval_end - interval_start == kRoundUpPowerOfTwoIntervals),
        "kRoundUpPowerOfTwoIntervals mismatch");

    int index = static_cast<int>(log_size) - static_cast<int>(interval_start);

    index = std::max(0, index);
    index = std::min(
        index, static_cast<int>(kRoundUpPowerOfTwoIntervals) - 1);
    return instance().m_roundup_power2_divisions[index];
  }

  static CachingAllocatorConfig& instance() {
    static CachingAllocatorConfig* s_instance = ([]() {
      auto inst = new CachingAllocatorConfig();
      const char* env = getenv("PYTORCH_CUDA_ALLOC_CONF");
      inst->parseArgs(env);
      return inst;
    })();
    return *s_instance;
  }

  void parseArgs(const char* env);

 private:
  CachingAllocatorConfig()
      : m_max_split_size(std::numeric_limits<size_t>::max()),
        m_garbage_collection_threshold(0) {
    m_roundup_power2_divisions.assign(kRoundUpPowerOfTwoIntervals, 0);
  }

  void lexArgs(const char* env, std::vector<std::string>& config);
  void consumeToken(
      const std::vector<std::string>& config,
      size_t i,
      const char c);
  size_t parseMaxSplitSize(const std::vector<std::string>& config, size_t i);
  size_t parseGarbageCollectionThreshold(
      const std::vector<std::string>& config,
      size_t i);
  size_t parseRoundUpPower2Divisions(
      const std::vector<std::string>& config,
      size_t i);
  size_t parseAllocatorConfig(
      const std::vector<std::string>& config,
      size_t i,
      bool& used_cudaMallocAsync);

  std::atomic<size_t> m_max_split_size;
  std::vector<size_t> m_roundup_power2_divisions;
  std::atomic<double> m_garbage_collection_threshold;
};

void CachingAllocatorConfig::lexArgs(
    const char* env,
    std::vector<std::string>& config) {
  std::vector<char> buf;

  size_t env_length = strlen(env);
  for (size_t i = 0; i < env_length; i++) {
    if (env[i] == ',' || env[i] == ':' || env[i] == '[' || env[i] == ']') {
      if (buf.size() != 0) {
        config.emplace_back(buf.begin(), buf.end());
        buf.clear();
      }
      config.emplace_back(1, env[i]);
    } else if (env[i] != ' ') {
      buf.emplace_back(static_cast<char>(env[i]));
    }
  }
  if (!buf.empty()) {
    config.emplace_back(buf.begin(), buf.end());
  }
}

void CachingAllocatorConfig::consumeToken(
    const std::vector<std::string>& config,
    size_t i,
    const char c) {
  TORCH_CHECK(
      i < config.size() && config[i].compare(std::string(1, c)) == 0,
      "Error parsing CachingAllocator settings, expected ",
      c,
      "");
}

size_t CachingAllocatorConfig::parseMaxSplitSize(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    size_t val1 = stoi(config[i]);
    TORCH_CHECK(
        val1 > kLargeBuffer / (1024 * 1024),
        "CachingAllocator option max_split_size_mb too small, must be > ",
        kLargeBuffer / (1024 * 1024),
        "");
    val1 = std::max(val1, kLargeBuffer / (1024 * 1024));
    val1 = std::min(val1, (std::numeric_limits<size_t>::max() / (1024 * 1024)));
    m_max_split_size = val1 * 1024 * 1024;
  } else {
    TORCH_CHECK(false, "Error, expecting max_split_size_mb value", "");
  }
  return i;
}

size_t CachingAllocatorConfig::parseGarbageCollectionThreshold(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    double val1 = stod(config[i]);
    TORCH_CHECK(
        val1 > 0, "garbage_collect_threshold too small, set it 0.0~1.0", "");
    TORCH_CHECK(
        val1 < 1.0, "garbage_collect_threshold too big, set it 0.0~1.0", "");
    m_garbage_collection_threshold = val1;
  } else {
    TORCH_CHECK(
        false, "Error, expecting garbage_collection_threshold value", "");
  }
  return i;
}

size_t CachingAllocatorConfig::parseRoundUpPower2Divisions(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  bool first_value = true;

  if (++i < config.size()) {
    if (config[i].compare("[") == 0) {
      size_t last_index = 0;
      while (++i < config.size() && config[i].compare("]") != 0) {
        std::string val1 = config[i];
        size_t val2 = 0;

        consumeToken(config, ++i, ':');
        if (++i < config.size()) {
          val2 = stoi(config[i]);
        } else {
          TORCH_CHECK(
              false, "Error parsing roundup_power2_divisions value", "");
        }
        TORCH_CHECK(
            c10::llvm::isPowerOf2_64(val2),
            "For roundups, the divisons has to be power of 2 ",
            "");

        if (val1.compare(">") == 0) {
          std::fill(
              std::next(
                  m_roundup_power2_divisions.begin(),
                  static_cast<std::vector<unsigned long>::difference_type>(
                      last_index)),
              m_roundup_power2_divisions.end(),
              val2);
        } else {
          size_t val1_long = stoul(val1);
          TORCH_CHECK(
              c10::llvm::isPowerOf2_64(val1_long),
              "For roundups, the intervals have to be power of 2 ",
              "");

          size_t index = 63 - c10::llvm::countLeadingZeros(val1_long);
          index = std::max((size_t)0, index);
          index = std::min(index, m_roundup_power2_divisions.size() - 1);

          if (first_value) {
            std::fill(
                m_roundup_power2_divisions.begin(),
                std::next(
                    m_roundup_power2_divisions.begin(),
                    static_cast<std::vector<unsigned long>::difference_type>(
                        index)),
                val2);
            first_value = false;
          }
          if (index < m_roundup_power2_divisions.size()) {
            m_roundup_power2_divisions[index] = val2;
          }
          last_index = index;
        }

        if (config[i + 1].compare("]") != 0) {
          consumeToken(config, ++i, ',');
        }
      }
    } else { // Keep this for backwards compatibility
      size_t val1 = stoi(config[i]);
      TORCH_CHECK(
          c10::llvm::isPowerOf2_64(val1),
          "For roundups, the divisons has to be power of 2 ",
          "");
      std::fill(
          m_roundup_power2_divisions.begin(),
          m_roundup_power2_divisions.end(),
          val1);
    }
  } else {
    TORCH_CHECK(false, "Error, expecting roundup_power2_divisions value", "");
  }
  return i;
}

size_t CachingAllocatorConfig::parseAllocatorConfig(
    const std::vector<std::string>& config,
    size_t i,
    bool& used_cudaMallocAsync) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    TORCH_CHECK(
        ((config[i] == "native") || (config[i] == "cudaMallocAsync")),
        "Unknown allocator backend, "
        "options are native and cudaMallocAsync");
    used_cudaMallocAsync = (config[i] == "cudaMallocAsync");
    if (used_cudaMallocAsync) {
#if CUDA_VERSION >= 11040
      int version;
      C10_CUDA_CHECK(cudaDriverGetVersion(&version));
      TORCH_CHECK(
          version >= 11040,
          "backend:cudaMallocAsync requires CUDA runtime "
          "11.4 or newer, but cudaDriverGetVersion returned ",
          version);
#else
      TORCH_CHECK(
          false,
          "backend:cudaMallocAsync requires PyTorch to be built with "
          "CUDA 11.4 or newer, but CUDA_VERSION is ",
          CUDA_VERSION);
#endif
    }
    TORCH_INTERNAL_ASSERT(
        config[i] == c10::cuda::CUDACachingAllocator::get()->name(),
        "Allocator backend parsed at runtime != "
        "allocator backend parsed at load time");
  } else {
    TORCH_CHECK(false, "Error parsing backend value", "");
  }
  return i;
}

void CachingAllocatorConfig::parseArgs(const char* env) {
  // If empty, set the default values
  m_max_split_size = std::numeric_limits<size_t>::max();
  m_roundup_power2_divisions.assign(kRoundUpPowerOfTwoIntervals, 0);
  m_garbage_collection_threshold = 0;
  bool used_cudaMallocAsync = false;
  bool used_native_specific_option = false;

  if (env == nullptr) {
    return;
  }

  std::vector<std::string> config;
  lexArgs(env, config);

  for (size_t i = 0; i < config.size(); i++) {
    if (config[i].compare("max_split_size_mb") == 0) {
      i = parseMaxSplitSize(config, i);
      used_native_specific_option = true;
    } else if (config[i].compare("garbage_collection_threshold") == 0) {
      i = parseGarbageCollectionThreshold(config, i);
      used_native_specific_option = true;
    } else if (config[i].compare("roundup_power2_divisions") == 0) {
      i = parseRoundUpPower2Divisions(config, i);
      used_native_specific_option = true;
    } else if (config[i].compare("backend") == 0) {
      i = parseAllocatorConfig(config, i, used_cudaMallocAsync);
    } else {
      TORCH_CHECK(false, "Unrecognized CachingAllocator option: ", config[i]);
    }

    if (i + 1 < config.size()) {
      consumeToken(config, ++i, ',');
    }
  }

  if (used_cudaMallocAsync && used_native_specific_option) {
    TORCH_WARN(
        "backend:cudaMallocAsync ignores max_split_size_mb, roundup_bypass_threshold_mb,"
        "roundup_power2_divisions, and garbage_collect_threshold.");
  }
}

class DeviceCachingAllocator {
 private:
  // lock around all operations
  mutable std::recursive_mutex mutex;

  // device statistics
  DeviceStats stats;

  // unallocated cached blocks larger than 1 MB
  BlockPool large_blocks;

  // unallocated cached blocks 1 MB or smaller
  BlockPool small_blocks;

  // allocated or in use by a stream. Holds all active allocations,
  // whether they came from graph_pools or one of the BlockPools above.
  ska::flat_hash_set<Block*> active_blocks;

  // captures_underway tracks if a capture might be underway on any stream.
  // Most of the time it's zero, in which case malloc can avoid calling
  // cudaStreamGetCaptureInfo in the hot path.
  int captures_underway = 0;
  // See free() for this thing's purpose
  std::vector<Block*> needs_events_deferred_until_no_capture;
  // outstanding cuda events
  ska::flat_hash_map<
      c10::cuda::CUDAStream,
      std::deque<std::pair<EventPool::Event, Block*>>>
      cuda_events;

  // record used memory.
  size_t total_allocated_memory = 0;

  size_t allowed_memory_maximum = 0;

  bool set_fraction = false;

  bool record_history = false;
  std::atomic<CreateContextFn> context_recorder_;
  size_t alloc_trace_next = 0;
  bool alloc_trace_record_context_ = false;
  size_t alloc_trace_max_entries_ = 1;
  std::vector<TraceEntry>*
      alloc_trace; // pointer because we need to intentionally leak this on
                   // deallocation it can hold references to Python state which
                   // will already be destroyed when we are in exit handlers

  // Members specific to CUDA graphs

  // Private pools for CUDA graphs
  ska::flat_hash_map<MempoolId_t, std::unique_ptr<PrivatePool>, MempoolIdHash>
      graph_pools;
  // Pools no longer referenced by any graph. Their BlockPools are eligible for
  // free_blocks. Can't be a vector or deque because we might erase entries in
  // any order. Could be an std::list, but we don't care much, access and
  // insert/erase are rare.
  ska::flat_hash_map<MempoolId_t, PrivatePool*, MempoolIdHash>
      graph_pools_freeable;

  // Indicates that a current stream should be allocated to a pool
  // rather than the global memory.
  ska::flat_hash_map<cudaStream_t, MempoolId_t> stream_to_pool_map;

  // XXX - maybe we should generalize and have multiple events
  std::vector<OutOfMemoryObserver> oom_observers_;

  ska::flat_hash_map<void*, Block*> ptr_to_block_map;

  size_t num_cuda_mallocs = 0;
  size_t num_cuda_frees = 0;

 public:
  DeviceCachingAllocator()
      : large_blocks(BlockComparator, /*is_small=*/false),
        small_blocks(BlockComparator, /*is_small=*/true),
        alloc_trace(new std::vector<TraceEntry>()) {
    stats.max_split_size = CachingAllocatorConfig::max_split_size();
    context_recorder_.store(nullptr);
  }

  void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      bool alloc_trace_record_context) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    record_history = enabled;
    context_recorder_.store(context_recorder);
    alloc_trace_max_entries_ = std::max(size_t(1), alloc_trace_max_entries);
    alloc_trace_record_context_ = alloc_trace_record_context;
    alloc_trace_next = 0;
    alloc_trace->clear();
  }

  void attachOutOfMemoryObserver(OutOfMemoryObserver observer) {
    oom_observers_.emplace_back(std::move(observer));
  }

  // All public methods (except the above) acquire the allocator mutex.
  // Thus, do not call a public method from another public method.

  void* malloc(int device, size_t orig_size, cudaStream_t stream) {
    // done outside the lock because we don't know what locks the recorder needs
    // to have...
    CreateContextFn context_recorder = context_recorder_.load();
    std::shared_ptr<GatheredContext> context =
        context_recorder ? context_recorder() : nullptr;

    std::unique_lock<std::recursive_mutex> lock(mutex);

    if (C10_LIKELY(captures_underway == 0)) {
      // Processes end-of-life events for outstanding allocations used on
      // multiple streams (checks if their GPU-side uses are complete and
      // recycles their memory if so)
      //
      // Q. Why skip process_events if a capture might be underway?
      // A. process_events involves cudaEventQueries, illegal during CUDA graph
      //    capture.
      //    Dumb simple solution: defer reclaiming these allocations until after
      //    capture. Cross-stream memory use is uncommon, so the deferral's
      //    effect on memory use during capture should be small.
      process_events();
    }
    size_t size = round_size(orig_size);
    auto& pool = get_pool(size, stream);
    const size_t alloc_size = get_allocation_size(size);
    AllocParams params(device, size, stream, &pool, alloc_size, stats);
    params.stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    params.stat_types[static_cast<size_t>(get_stat_type_for_pool(pool))] = true;

    // First, try to get a block from the existing pool.
    bool block_found =
        // Search pool
        get_free_block(params)
        // Trigger callbacks and retry search
        || (trigger_free_memory_callbacks(params) && get_free_block(params));

    // Can't reuse an existing block; try to get a new one.
    if (!block_found) {
      // Do garbage collection if the flag is set.
      if (C10_UNLIKELY(
              set_fraction &&
              CachingAllocatorConfig::garbage_collection_threshold() > 0.0)) {
        garbage_collect_cached_blocks();
      }
      // Attempt allocate
      block_found = alloc_block(params, false)
          // Free enough available cached blocks to satisfy alloc and retry
          // alloc.
          || (release_available_cached_blocks(params) &&
              alloc_block(params, false))
          // Free all non-split cached blocks and retry alloc.
          || (C10_LIKELY(captures_underway == 0) && release_cached_blocks() &&
              alloc_block(params, true));
      if (record_history && block_found) {
        record_trace(
            TraceEntry::SEGMENT_ALLOC,
            int64_t(params.block->ptr),
            params.block->size,
            params.stream(),
            context);
      }
    }

    if (!block_found) {
      // For any error code other than cudaErrorMemoryAllocation,
      // alloc_block should have thrown an exception already.
      TORCH_INTERNAL_ASSERT(params.err == cudaErrorMemoryAllocation);

      size_t device_free;
      size_t device_total;
      C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
      std::string allowed_info;

      if (set_fraction) {
        allowed_info = format_size(allowed_memory_maximum) + " allowed; ";
      }

      if (record_history) {
        record_trace(
            TraceEntry::OOM,
            device_free,
            params.size(),
            params.stream(),
            std::move(context));
      }
      stats.num_ooms += 1;

      c10::reportOutOfMemoryToProfiler(
          size,
          stats.allocated_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
              .current,
          stats.reserved_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
              .current,
          c10::Device(c10::DeviceType::CUDA, static_cast<DeviceIndex>(device)));

      auto allocated_bytes =
          stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;
      auto reserved_bytes =
          stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;
      auto observers_local = oom_observers_;

      // Make sure we do not have the device lock before calling our
      // observers which might need hold the GIL
      // It is safe to release at this point because will no longer
      // be reading any allocator state.

      lock.unlock();

      for (const auto& obs : observers_local) {
        obs(device,
            alloc_size,
            set_fraction ? allowed_memory_maximum : device_total,
            device_free);
      }

      // "total capacity": total global memory on GPU
      // "allowed": memory is allowed to use, which set by fraction.
      // "already allocated": memory allocated by the program using the
      //                      caching allocator
      // "free": free memory as reported by the CUDA API
      // "cached": memory held by the allocator but not used by the program
      //
      // The "allocated" amount  does not include memory allocated outside
      // of the caching allocator, such as memory allocated by other programs
      // or memory held by the driver.
      //
      // The sum of "allocated" + "free" + "cached" may be less than the
      // total capacity due to memory held by the driver and usage by other
      // programs.
      //
      // Note that at this point free_cached_blocks has already returned all
      // possible "cached" memory to the driver. The only remaining "cached"
      // memory is split from a larger block that is partially in-use.
      TORCH_CHECK_WITH(
          OutOfMemoryError,
          false,
          "CUDA out of memory. Tried to allocate ",
          format_size(alloc_size),
          " (GPU ",
          device,
          "; ",
          format_size(device_total),
          " total capacity; ",
          format_size(allocated_bytes),
          " already allocated; ",
          format_size(device_free),
          " free; ",
          allowed_info,
          format_size(reserved_bytes),
          " reserved in total by PyTorch)",
          " If reserved memory is >> allocated memory try setting max_split_size_mb to avoid"
          " fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF",
          "");
    }

    bool split_remainder = should_split(params.block, params.size());
    Block* allocated_block = alloc_found_block(
        std::move(params), orig_size, std::move(context), split_remainder);
    TORCH_CHECK(ptr_to_block_map.find(allocated_block->ptr) == ptr_to_block_map.end(), "Block already in allocation map.");
    ptr_to_block_map[allocated_block->ptr] = allocated_block;
    return allocated_block->ptr;
  }

  Block* alloc_found_block(
      AllocParams params,
      size_t orig_size,
      std::shared_ptr<GatheredContext> context,
      bool split_remainder) {
    auto size = params.size();
    auto device = params.device();
    auto pool = params.pool;
    auto stream = params.stream();

    TORCH_INTERNAL_ASSERT(
        params.err == cudaSuccess && params.block != nullptr &&
        params.block->ptr != nullptr);
    Block* block = params.block;
    Block* remaining = nullptr;

    const bool already_split = block->is_split();
    if (split_remainder) {
      remaining = block;

      block = new Block(device, stream, size, pool, block->ptr);
      block->prev = remaining->prev;
      if (block->prev) {
        block->prev->next = block;
      }
      block->next = remaining;

      remaining->prev = block;
      remaining->ptr = static_cast<char*>(remaining->ptr) + size;
      remaining->size -= size;
      bool inserted = pool->blocks.insert(remaining).second;
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

      if (record_history) {
        trimHistoryBefore(remaining, (char*)block->ptr + size);
      }

      if (already_split) {
        // An already-split inactive block is being shrunk by size bytes.
        update_stat_array(
            stats.inactive_split_bytes,
            -static_cast<std::int64_t>(block->size),
            params.stat_types);
      } else {
        // A new split inactive block is being created from a previously unsplit
        // block, size remaining->size bytes.
        for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
          update_stat(
              stats.inactive_split_bytes[stat_type],
              static_cast<std::int64_t>(remaining->size));
          update_stat(stats.inactive_split[stat_type], 1);
        });
      }

    } else if (already_split) {
      // An already-split block is becoming active
      for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
        update_stat(
            stats.inactive_split_bytes[stat_type],
            -static_cast<std::int64_t>(block->size));
        update_stat(stats.inactive_split[stat_type], -1);
      });
    }

    block->allocated = true;
    block->requested_size = orig_size;
    if (record_history) {
      trimHistoryBefore(block, (char*)block->ptr + size);
      block->history = std::make_unique<HistoryChain>(HistoryChain{
          History{block->ptr, orig_size, std::move(context)},
          std::move(block->history)});
      if (!block->history_last) {
        block->history_last = block->history.get();
      }
      record_trace(
          TraceEntry::ALLOC,
          int64_t(block->ptr),
          orig_size,
          block->stream,
          block->history->h.context);
    }

    bool inserted = active_blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

    for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
      update_stat(stats.allocation[stat_type], 1);
      update_stat(
          stats.allocated_bytes[stat_type],
          static_cast<std::int64_t>(block->size));
      update_stat(stats.active[stat_type], 1);
      update_stat(
          stats.active_bytes[stat_type],
          static_cast<std::int64_t>(block->size));
      update_stat(
          stats.requested_bytes[stat_type],
          static_cast<std::int64_t>(block->requested_size));
    });
    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_allocations, 1);

    c10::reportMemoryUsageToProfiler(
        block->ptr,
        block->size,
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(c10::DeviceType::CUDA, device));

    return block;
  }

  void free(void* ptr) {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    TORCH_CHECK(ptr_to_block_map.find(ptr) != ptr_to_block_map.end(), "ptr not in allocation map.");
    Block* block = ptr_to_block_map.at(ptr);
    ptr_to_block_map.erase(ptr);

    block->allocated = false;

    // following logic might modifying underlaying Block, causing the size
    // changed. We store ahead for reporting
    auto orig_block_ptr = block->ptr;
    auto orig_block_size = block->size;

    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*(block->pool)))] =
        true;
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.allocation[stat_type], -1);
      update_stat(
          stats.allocated_bytes[stat_type],
          -static_cast<std::int64_t>(block->size));
    });
    if (block->history) {
      record_trace(
          TraceEntry::FREE_REQUESTED,
          int64_t(block->ptr),
          block->history->h.real_size,
          block->stream,
          block->history->h.context);
    }
    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_allocations, -1);

    if (!block->stream_uses.empty()) {
      if (C10_UNLIKELY(captures_underway)) {
        // It's forbidden to cudaEventQuery an event recorded during CUDA graph
        // capture. We conservatively defer recording end-of-life events until
        // the next call to process_events() (which won't happen until no
        // captures are underway)
        needs_events_deferred_until_no_capture.push_back(block);
      } else {
        insert_events(block);
      }
    } else {
      free_block(block);
    }

    c10::reportMemoryUsageToProfiler(
        orig_block_ptr,
        -orig_block_size,
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(c10::DeviceType::CUDA, block->device));
  }

  void* getBaseAllocation(Block* block, size_t* outSize) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    while (block->prev) {
      block = block->prev;
    }
    void* basePtr = block->ptr;
    if (outSize) {
      size_t size = 0;
      while (block) {
        size += block->size;
        block = block->next;
      }
      *outSize = size;
    }
    return basePtr;
  }

  void recordStream(void* ptr, c10::cuda::CUDAStream stream) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    TORCH_CHECK(ptr_to_block_map.find(ptr) != ptr_to_block_map.end(), "ptr not in allocation map.");
    Block* block = ptr_to_block_map.at(ptr);
    if (stream.stream() == block->stream) {
      // ignore uses on the allocation stream, since those don't require any
      // special synchronization
      return;
    }
    block->stream_uses.insert(stream);
  }

  /** set memory fraction to limit maximum allocated memory **/
  void setMemoryFraction(double fraction) {
    size_t device_free;
    size_t device_total;
    C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
    allowed_memory_maximum = static_cast<size_t>(fraction * device_total);
    set_fraction = true;
  }

  /** returns cached blocks to the system allocator **/
  void emptyCache() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    release_cached_blocks();
  }

  /** Retrieves size of largest unused block held by the memory cache **/
  void cacheInfo(size_t* largest) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (*largest ==
        0) { // make an initial guess if a zero *largest is passed in
      size_t tmp_bytes;
      C10_CUDA_CHECK(cudaMemGetInfo(
          largest, // Use free memory as an optimistic initial guess of *largest
          &tmp_bytes));
    }
    cache_info_aux(large_blocks, largest);
    cache_info_aux(small_blocks, largest);
    for (const auto& gp : graph_pools) {
      cache_info_aux(gp.second->large_blocks, largest);
      cache_info_aux(gp.second->small_blocks, largest);
    }
  }

  /** Returns a copy of the memory allocator stats **/
  DeviceStats getStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return stats;
  }

  /** Resets the historical accumulation stats for the device **/
  void resetAccumulatedStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      reset_accumulated_stat(stats.allocation[statType]);
      reset_accumulated_stat(stats.segment[statType]);
      reset_accumulated_stat(stats.active[statType]);
      reset_accumulated_stat(stats.inactive_split[statType]);
      reset_accumulated_stat(stats.allocated_bytes[statType]);
      reset_accumulated_stat(stats.reserved_bytes[statType]);
      reset_accumulated_stat(stats.active_bytes[statType]);
      reset_accumulated_stat(stats.inactive_split_bytes[statType]);
      reset_accumulated_stat(stats.requested_bytes[statType]);
    }

    stats.num_alloc_retries = 0;
    stats.num_ooms = 0;
    reset_accumulated_stat(stats.oversize_allocations);
    reset_accumulated_stat(stats.oversize_segments);
  }

  /** Resets the historical peak stats for the device **/
  void resetPeakStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      reset_peak_stat(stats.allocation[statType]);
      reset_peak_stat(stats.segment[statType]);
      reset_peak_stat(stats.active[statType]);
      reset_peak_stat(stats.inactive_split[statType]);
      reset_peak_stat(stats.allocated_bytes[statType]);
      reset_peak_stat(stats.reserved_bytes[statType]);
      reset_peak_stat(stats.active_bytes[statType]);
      reset_peak_stat(stats.inactive_split_bytes[statType]);
      reset_peak_stat(stats.requested_bytes[statType]);
    }
    reset_peak_stat(stats.oversize_allocations);
    reset_peak_stat(stats.oversize_segments);
  }

  int getNumCudaMallocs() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return num_cuda_mallocs;
  }

  int getNumCudaFrees() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return num_cuda_frees;
  }

  int64_t getPeakReservedCudaMemory() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return stats.reserved_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
              .peak;
  }

  int64_t getCurrentReservedCudaMemory() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return stats.reserved_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
              .current;
  }

  int64_t getPeakAllocatedCudaMemory() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return stats.allocated_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
              .peak;
  }

  int64_t getCurrentAllocatedCudaMemory() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return stats.allocated_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
              .current;
  }

  int64_t getPeakRequestedCudaMemory() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return stats.requested_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
              .peak;
  }

  int64_t getCurrentRequestedCudaMemory() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return stats.requested_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
              .current;
  }

  /* Checkpoint the state of a private pool necessary to return it to its
   * current state */
  std::unique_ptr<PrivatePoolState> getCheckpointState(MempoolId_t id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    auto pool = graph_pools.find(id);
    if (pool != graph_pools.end()) {
      auto private_pool_head_blocks =
          get_private_pool_head_blocks(pool->second.get());
      return std::make_unique<PrivatePoolState>(id, private_pool_head_blocks);
    } else if (graph_pools_freeable.count(id)) {
      TORCH_CHECK(false, "Not expected to checkpoint freeable graph");
    } else {
      TORCH_CHECK(false, "Could not find pool of id");
    }
  }

  void freeBlocksAllocatedToPool(PrivatePool* private_pool, RestoreResult& rr) {
    std::unordered_map<void*, Block*> orig_ptrs_to_blocks;

    auto pool_blocks = get_private_pool_head_blocks(private_pool);

    std::vector<Block*> head_blocks;
    for (Block* block : pool_blocks) {
      if (block->prev == nullptr) {
        head_blocks.push_back(block);
      }
    }

    for (Block* block : head_blocks) {
      Block* curr = block;

      while (curr) {
        // When we free a block, its pointer should never change
        // only its adjacent blocks, so free, then look at pointer
        if (curr->allocated) {
          TORCH_CHECK(
              curr->event_count == 0,
              "Events should have synchronized when setting checkpointed block");
          rr.allocations_freed.push_back(curr->ptr);
          free(curr);
          TORCH_CHECK(!curr->allocated)
        }
        curr = curr->next;
      }
    }

    for (Block* b : get_private_pool_head_blocks(private_pool)) {
      Block* curr = b;
      while (curr) {
        TORCH_CHECK(!curr->allocated);
        curr = curr->next;
      }
    }
  }

  // checkpoint the state of an allocation that may have been
  // split into multiple blocks
  void setSegmentStateToCheckpoint(
      Block* block,
      SegmentState& segment,
      std::shared_ptr<GatheredContext> context,
      RestoreResult& rr) {
    Block* curr_block = block;
    Block* last_block = block;

    TORCH_INTERNAL_ASSERT(block->pool);
    BlockPool& pool = *block->pool;
    const auto segment_len = segment.blocks.size();

    // allocate all blocks in the segment
    for (size_t i = 0; i < segment_len; ++i) {
      auto& block_state = segment.blocks.at(i);
      AllocParams params(
          block_state.device,
          block_state.size,
          block_state.stream,
          &pool,
          block_state.size,
          stats);
      pool.blocks.erase(curr_block);
      params.block = curr_block;
      params.stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
      params.stat_types[static_cast<size_t>(get_stat_type_for_pool(pool))] =
          true;

      // splitting a block depends on `max_split_size`, which may have changed
      // between whe checkpoint was taken and now, so we make sure to recreate
      // the behavior from the checkpoint.
      bool split = (i + 1) < segment.blocks.size();

      // curr_block will become next pointer if it is split, so reassign with
      // the returned value
      curr_block = alloc_found_block(
          std::move(params), block_state.size, context, split);

      TORCH_CHECK(curr_block->ptr == block_state.ptr);
      TORCH_CHECK(curr_block->size == block_state.size);

      last_block = curr_block;
      curr_block = curr_block->next;

      TORCH_CHECK((curr_block != nullptr) == ((i + 1) < (segment_len)));
    }

    while (last_block->prev) {
      last_block = last_block->prev;
    }

    // free blocks that are not allocated in the checkpoint
    curr_block = last_block;

    for (size_t i = 0; i < segment_len; ++i, curr_block = curr_block->next) {
      auto& block_state = segment.blocks.at(i);
      TORCH_INTERNAL_ASSERT(curr_block != nullptr);

      if (block_state.allocated) {
        rr.allocations_created.push_back(curr_block);
        continue;
      }

      free(curr_block);

      TORCH_CHECK(curr_block->ptr == block_state.ptr);
      TORCH_CHECK(curr_block->allocated == block_state.allocated);
      TORCH_CHECK(curr_block->size == block_state.size);
    }
  }

  /**
   * Note [Checkpointing PrivatePoolState]
   *
   * Refer above to Note [Interaction with CUDA graph capture]. Allocations made
   * during graph capture are made from a separate private pool. During graph
   * capture allocations behave as usual. During graph replay the allocator
   * state does not change even as new tensors are created. The private pool
   * will not free its blocks to the main caching allocator until cuda graph use
   * is finished to prevent an allocation from eager clobbering the memory from
   * a live but unaccounted for tensor that was created during replay.
   *
   * `make_graphed_callables`, a series of separate callables chained in
   * successive cuda graphs, can share a memory pool because after a cuda graph
   * recording the allocations in the shared private pool exactly reflect the
   * tensors that are allocated.
   *
   * We would like to extend callable chaining to support a graphed callable
   * tree. In this scenario, we have a tree of callable chains which will be
   * captured with cuda graphs. In the diagram below, we have a tree with four
   * callables, A, B, C, and D. Suppose we have captured, and subsequently
   * replayed, A, B, and C. Then on a new invocation, we replay A and B, but
   * would now like to record D. At this point the private pool will not reflect
   * any of the live tensors created during graph replay. Allocations made
   * during a new recording with the pool could overwrite those live tensors.
   *
   * In order to record a new graph capture after replaying prior callables in
   * the tree, we need the allocator to reflect the state of the live tensors.
   * We checkpoint the state of the private pool after each recording, and then
   * reapply it when we are starting a new recording chain. Additionally, we
   * must free the allocations for any tensors that died between the end of our
   * previous graph replaying and our new recording. All of the allocated
   * segments that existed in the checkpointed state must still exist in the
   * pool. There may also exist new allocated blocks.
   * (TODO : link note [live tensors between iterations] when it exists). For
   * every block that is currently allocated but no allocated in the snapshot,
   * we will return a pointer to their block.
   *.
   *
   *
   *  ---------------> A ---------------> B ---------------> C
   *                                      |
   *                                      |
   *                                      |
   *                                      |
   *                                      ╰ ---------------> D
   */
  RestoreResult setCheckpointPoolState(PrivatePoolState& pps) {
    // To reset the caching allocator state we will
    // - Free all the blocks currently allocated to the pool (see [live tensors
    // between iterations])
    // - Allocate all the blocks in a checkpointed segment, whether they are
    // live or not
    // - Free the blocks in a checkpointed segment which are not live
    // This could be optimized, but it nicely reuses exiting apis, and this
    // is not on the hot path.

    // following `done outside the lock because we don't know what locks the
    // recorder needs to have...`

    CreateContextFn context_recorder = context_recorder_.load();
    std::shared_ptr<GatheredContext> context =
        context_recorder ? context_recorder() : nullptr;

    std::lock_guard<std::recursive_mutex> lock(mutex);

    RestoreResult rr;

    TORCH_CHECK(
        !graph_pools_freeable.count(pps.owner_id),
        "Not expected to checkpoint freeable graph");

    auto pool = graph_pools.find(pps.owner_id);
    TORCH_CHECK(pool != graph_pools.end(), "Could not find private pool id");

    PrivatePool* private_pool = pool->second.get();

    freeBlocksAllocatedToPool(private_pool, rr);

    std::unordered_map<void*, Block*> ptrs_to_blocks;
    // at this point, all of the blocks should be free, so they will all be in
    // the block set
    for (Block* block : private_pool->small_blocks.blocks) {
      ptrs_to_blocks[block->ptr] = block;
    }
    for (Block* block : private_pool->large_blocks.blocks) {
      ptrs_to_blocks[block->ptr] = block;
    }

    for (auto& segment : pps.segments) {
      auto ptr = segment.blocks.at(0).ptr;
      TORCH_CHECK(ptrs_to_blocks.count(ptr), " could not find ", ptr)
      auto block = ptrs_to_blocks[ptr];

      setSegmentStateToCheckpoint(block, segment, context, rr);
    }
    return rr;
  }

  /** Dump a complete snapshot of the memory held by the allocator. Potentially
   * VERY expensive. **/
  std::vector<SegmentInfo> snapshot() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    std::unordered_map<PrivatePool*, MempoolId_t> pool_to_id;
    pool_to_id.reserve(graph_pools.size() + graph_pools_freeable.size());
    for (const auto& pair : graph_pools) {
      pool_to_id[pair.second.get()] = pair.first;
    }
    for (const auto& pair : graph_pools_freeable) {
      pool_to_id[pair.second] = pair.first;
    }

    size_t total_active = 0;
    std::vector<SegmentInfo> result;
    const auto all_blocks = get_all_blocks();

    for (const Block* const head_block : all_blocks) {
      if (head_block->prev != nullptr) {
        continue;
      }
      result.emplace_back();
      SegmentInfo& segment_info = result.back();
      segment_info.device = head_block->device;
      segment_info.address = reinterpret_cast<int64_t>(head_block->ptr);
      segment_info.stream = head_block->stream;
      segment_info.is_large = (!head_block->pool->is_small);

      auto mempool_id = pool_to_id.find(head_block->pool->owner_PrivatePool);
      if (mempool_id != pool_to_id.end()) {
        segment_info.owner_private_pool_id = mempool_id->second;
      }

      const Block* block = head_block;
      while (block != nullptr) {
        segment_info.blocks.emplace_back();
        BlockInfo& block_info = segment_info.blocks.back();

        block_info.size = block->size;
        block_info.requested_size = block->requested_size;
        block_info.allocated = block->allocated;
        block_info.active = block->allocated || (block->event_count > 0) ||
            !block->stream_uses.empty();

        segment_info.total_size += block_info.size;
        if (block_info.allocated) {
          segment_info.allocated_size += block_info.size;
        }
        if (block_info.active) {
          segment_info.active_size += block_info.size;
          segment_info.requested_size += block_info.requested_size;
        }
        HistoryChain* h = block->history.get();
        while (h) {
          block_info.history.push_back(h->h);
          h = h->next.get();
        }
        block = block->next;
      }
      total_active += segment_info.active_size;
    }

    std::sort(
        result.begin(),
        result.end(),
        [](const SegmentInfo& a, const SegmentInfo& b) {
          return a.address < b.address;
        });

    if (record_history) {
      record_trace(TraceEntry::SNAPSHOT, 0, total_active, 0, nullptr);
    }
    return result;
  }

  std::vector<TraceEntry> trace() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    std::vector<TraceEntry> result;
    result.reserve(alloc_trace->size());
    result.insert(
        result.end(),
        alloc_trace->begin() + alloc_trace_next,
        alloc_trace->end());
    result.insert(
        result.end(),
        alloc_trace->begin(),
        alloc_trace->begin() + alloc_trace_next);
    return result;
  }

  // This function takes the size and number of divisions argument and rounds
  // up the size argument for the nearest power-of-2 division.
  // For example, if we need to round-up 1200 and number of divisions is 4,
  // the size 1200 lies between 1024 and 2048 and if we do 4 divisions between
  // them, the values are 1024, 1280, 1536, and 1792. So the function will
  // return 1280 as the nearest ceiling of power-2 divison.
  static size_t roundup_power2_next_division(size_t size, size_t divisions) {
    if (C10_UNLIKELY(size <= 4 || divisions <= 1)) {
      return size;
    }
    if (c10::llvm::isPowerOf2_64(size)) {
      return size;
    }

    // divide the space between these 2's power into equal divisions
    // If division is zero, return the power-of-2 ceiling.
    size_t power2_floor = c10::llvm::PowerOf2Floor(size);
    size_t power2_divison =
        power2_floor >> (63 - c10::llvm::countLeadingZeros(divisions));
    if (C10_UNLIKELY(power2_divison == 0)) {
      return (power2_floor << 1);
    }
    size_t round_size_floor = size & (~(power2_divison - 1));
    return (round_size_floor == size) ? size
                                      : round_size_floor + power2_divison;
  }

  static size_t round_size(size_t size) {
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    } else {
      auto divisions = CachingAllocatorConfig::roundup_power2_divisions(size);
      if (divisions > 0 && size > (kMinBlockSize * divisions)) {
        return roundup_power2_next_division(size, divisions);
      } else {
        return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
      }
    }
  }

  // See Note [Interaction with CUDA graph capture]

  // Called by CUDAGraph::capture_begin
  void beginAllocateStreamToPool(cudaStream_t stream, MempoolId_t mempool_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    captures_underway++;
    auto it = graph_pools.find(mempool_id);
    if (it == graph_pools.end()) {
      // mempool_id does not reference an existing pool. Make a new pool for
      // this capture.
      graph_pools.emplace(mempool_id, std::make_unique<PrivatePool>());
    } else {
      // mempool_id references an existing pool, which the current capture will
      // share. Check this pool is live (at least one other capture already
      // references it).
      TORCH_INTERNAL_ASSERT(it->second->use_count > 0);
      it->second->use_count++;
    }

    // Maps this stream to mempool_id and makes sure this graph_id wasn't
    // somehow assigned a mempool_id already. Keeps essential effect (insert)
    // out of macro.
    bool inserted = stream_to_pool_map.insert({stream, mempool_id}).second;
    TORCH_INTERNAL_ASSERT(inserted);
  }

  // Called by CUDAGraph::capture_end
  void endAllocateStreamToPool(cudaStream_t stream) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    captures_underway--;
    auto it = stream_to_pool_map.find(stream);
    TORCH_INTERNAL_ASSERT(it != stream_to_pool_map.end());
    stream_to_pool_map.erase(it);
  }

  // Called by CUDAGraph::reset
  void releasePool(MempoolId_t mempool_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    // The instantiated cudaGraphExec_t has been destroyed. We can't blindly
    // delete and cudaFree the mempool its capture used, because
    //  1. other graph(s) might share the same pool
    //  2. the user might still hold references to output tensors allocated
    //  during capture.
    // To handle 1 and 2, we track the number of graphs using this particular
    // mempool. When the count reaches 0, we tell free_cached_blocks it may now
    // cudaFree blocks from this graph's pool when it discovers they're unused
    // (unsplit).
    auto it = graph_pools.find(mempool_id);
    TORCH_INTERNAL_ASSERT(it != graph_pools.end());
    auto uc = --(it->second->use_count);
    TORCH_INTERNAL_ASSERT(uc >= 0);
    if (uc == 0) {
      // Allows free_cached_blocks to begin cudaFreeing this pool's memory,
      // and makes sure this pool wasn't somehow made freeable already.
      bool inserted =
          graph_pools_freeable.insert({mempool_id, it->second.get()}).second;
      TORCH_INTERNAL_ASSERT(inserted);
    }
  }

 private:
  // All private methods do not acquire the allocator mutex.

  std::vector<const Block*> get_all_blocks() const {
    std::vector<const Block*> blocks;
    blocks.insert(
        blocks.end(), small_blocks.blocks.begin(), small_blocks.blocks.end());
    blocks.insert(
        blocks.end(), large_blocks.blocks.begin(), large_blocks.blocks.end());
    for (const auto& gp : graph_pools) {
      blocks.insert(
          blocks.end(),
          gp.second->small_blocks.blocks.begin(),
          gp.second->small_blocks.blocks.end());
      blocks.insert(
          blocks.end(),
          gp.second->large_blocks.blocks.begin(),
          gp.second->large_blocks.blocks.end());
    }
    blocks.insert(blocks.end(), active_blocks.begin(), active_blocks.end());
    return blocks;
  }

  std::vector<Block*> get_private_pool_head_blocks(PrivatePool* pool) const {
    std::vector<Block*> blocks;
    for (Block* b : active_blocks) {
      if ((b->pool == &pool->small_blocks || b->pool == &pool->large_blocks) &&
          b->prev == nullptr) {
        blocks.push_back(b);
      }
    }

    for (Block* b : pool->small_blocks.blocks) {
      if (b->prev == nullptr) {
        blocks.push_back(b);
      }
    }
    for (Block* b : pool->large_blocks.blocks) {
      if (b->prev == nullptr) {
        blocks.push_back(b);
      }
    }

    return blocks;
  }

  /** moves a block into a pool of cached free blocks */
  void free_block(Block* block) {
    TORCH_INTERNAL_ASSERT(
        !block->allocated && block->event_count == 0 &&
        block->stream_uses.empty());
    if (block->history) {
      record_trace(
          TraceEntry::FREE_COMPLETED,
          int64_t(block->ptr),
          block->history->h.real_size,
          block->stream,
          block->history->h.context);
    }
    size_t original_block_size = block->size;
    size_t requested_size = block->requested_size;

    auto& pool = *block->pool;
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    const std::array<Block*, 2> merge_candidates = {block->prev, block->next};
    for (Block* merge_candidate : merge_candidates) {
      const int64_t subsumed_size =
          try_merge_blocks(block, merge_candidate, pool);
      if (subsumed_size > 0) {
        net_change_inactive_split_blocks -= 1;
        net_change_inactive_split_size -= subsumed_size;
      }
    }

    active_blocks.erase(block);
    // Makes sure the Block* isn't already present in the pool we're freeing it
    // back into.
    bool inserted = pool.blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT(inserted);

    if (block->is_split()) {
      net_change_inactive_split_blocks += 1;
      net_change_inactive_split_size += block->size;
    }

    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(pool))] = true;
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(
          stats.inactive_split[stat_type], net_change_inactive_split_blocks);
      update_stat(
          stats.inactive_split_bytes[stat_type],
          net_change_inactive_split_size);
      update_stat(stats.active[stat_type], -1);
      update_stat(
          stats.active_bytes[stat_type],
          -static_cast<std::int64_t>(original_block_size));
      update_stat(
          stats.requested_bytes[stat_type],
          -static_cast<std::int64_t>(requested_size));
    });
  }

  /** combine previously split blocks. returns the size of the subsumed block,
   * or 0 on failure. */
  size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
    if (!src || src->allocated || src->event_count > 0 ||
        !src->stream_uses.empty()) {
      return 0;
    }

    AT_ASSERT(dst->is_split() && src->is_split());

    if (dst->prev == src) { // [src dst]
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
      if (!dst->history) {
        dst->history = std::move(src->history);
        dst->history_last = src->history_last;
      } else if (src->history) {
        src->history_last->next = std::move(dst->history);
        dst->history = std::move(src->history);
      }
      src->history_last = nullptr;
    } else { // [dest src]
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }

      if (!dst->history) {
        dst->history = std::move(src->history);
        dst->history_last = src->history_last;
      } else if (src->history) {
        dst->history_last->next = std::move(src->history);
        dst->history_last = src->history_last;
      }
      src->history_last = nullptr;
    }
    const size_t subsumed_size = src->size;
    dst->size += subsumed_size;
    auto erased = pool.blocks.erase(src);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(erased == 1);
    delete src;

    return subsumed_size;
  }

  BlockPool& get_pool(size_t size, cudaStream_t stream) {
#if !defined(USE_ROCM) || ROCM_VERSION >= 50300
    // captures_underway is a conservative guess that the current stream may be
    // capturing. It's only > 0 if some thread has begun and not yet ended a
    // capture, so it's usually 0, and we can short-circuit
    // cudaStreamCaptureStatus (which does a TLS lookup).
    if (C10_UNLIKELY(captures_underway)) {
      auto it0 = stream_to_pool_map.find(stream);
      if (it0 != stream_to_pool_map.end()) {
        auto it1 = graph_pools.find(it0->second);
        TORCH_INTERNAL_ASSERT(it1 != graph_pools.end());
        if (size <= kSmallSize) {
          return it1->second->small_blocks;
        } else {
          return it1->second->large_blocks;
        }
      }
    }
#endif
    if (size <= kSmallSize) {
      return small_blocks;
    } else {
      return large_blocks;
    }
  }

  StatType get_stat_type_for_pool(const BlockPool& pool) {
    return pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL;
  }

  bool should_split(const Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool->is_small) {
      return remaining >= kMinBlockSize;
    } else {
      return (size < CachingAllocatorConfig::max_split_size()) &&
          (remaining > kSmallSize);
    }
  }

  static size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
      return kLargeBuffer;
    } else {
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }

  bool get_free_block(AllocParams& p) {
    BlockPool& pool = *p.pool;

    if (C10_UNLIKELY(
            set_fraction &&
            CachingAllocatorConfig::garbage_collection_threshold() > 0.0)) {
      // Track block reuse interval only when garbage collection is enabled.
      for (auto& b : pool.blocks) {
        ++b->gc_count;
      }
    }
    auto it = pool.blocks.lower_bound(&p.search_key);
    if (it == pool.blocks.end() || (*it)->stream != p.stream())
      return false;
    // Do not return an oversized block for a large request
    if ((p.size() < CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= CachingAllocatorConfig::max_split_size()))
      return false;
    // Allow oversized block size to be rounded up but within a limit
    if ((p.size() >= CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= p.size() + kLargeBuffer))
      return false;
    p.block = *it;
    (*it)->gc_count = 0; // Denote this block has been used
    pool.blocks.erase(it);
    return true;
  }

  bool trigger_free_memory_callbacks(AllocParams& p) {
    bool freed_memory = false;
    for (const auto& name : FreeCudaMemoryCallbacksRegistry()->Keys()) {
      freed_memory |=
          FreeCudaMemoryCallbacksRegistry()->Create(name)->Execute();
    }
    return freed_memory;
  }

  void garbage_collect_cached_blocks() {
    // Free unused cached blocks to reclaim GPU memory.
    // Unlike release_cached_blocks(), this does not enforce synchronization and
    // therefore should be of less overheads.

    size_t gc_threshold = static_cast<size_t>(
        CachingAllocatorConfig::garbage_collection_threshold() *
        allowed_memory_maximum);
    // No need to trigger GC yet
    if (total_allocated_memory <= gc_threshold) {
      return;
    }
    const auto target_size = total_allocated_memory - gc_threshold;
    size_t gc_reclaimed = 0;

    // Calculate the total age of the free-able blocks. We'll use it later to
    // get "avg age" threshold.
    double total_age = 0.0;
    int freeable_block_count = 0;
    for (auto& b : large_blocks.blocks) {
      if (!b->is_split()) {
        total_age += b->gc_count;
        ++freeable_block_count;
      }
    }
    // No free-able blocks?
    if (freeable_block_count == 0) {
      return;
    }

    // Repeat GC until we reach reclaim > target size.
    bool block_freed = true;
    while (gc_reclaimed < target_size && block_freed == true &&
           freeable_block_count > 0) {
      // Free blocks exceeding this age threshold first.
      double age_threshold = total_age / freeable_block_count;
      // Stop iteration if we can no longer free a block.
      block_freed = false;

      // Free blocks of > avg age. Don't stop upon reaching the target_size,
      // we don't want this GC to be triggered frequently.
      auto it = large_blocks.blocks.begin();
      while (it != large_blocks.blocks.end()) {
        Block* block = *it;
        ++it;
        if (!block->is_split() && block->gc_count >= age_threshold) {
          block_freed = true;
          gc_reclaimed += block->size;
          total_age -= block->gc_count; // Decrement the age
          freeable_block_count--; // One less block that can be freed
          release_block(block);
        }
      }
    }
  }

  bool alloc_block(AllocParams& p, bool isRetry) {
    // Defensively checks for preexisting CUDA error state.
    C10_CUDA_CHECK(cudaGetLastError());

    size_t size = p.alloc_size;
    void* ptr;

    if (isRetry) {
      stats.num_alloc_retries += 1;
    }

    if (set_fraction &&
        total_allocated_memory + size > allowed_memory_maximum) {
      p.err = cudaErrorMemoryAllocation;
      return false;
    } else {
      p.err = cudaMallocMaybeCapturing(&ptr, size);
      if (p.err != cudaSuccess) {
        if (p.err == cudaErrorMemoryAllocation) {
          // If this is the first attempt (!isRetry), we can forgive and clear
          // CUDA's internal error state.
          //
          // If this is the second attempt (isRetry), malloc's TORCH_CHECK_WITH
          // will take over to throw a helpful exception. The user can choose
          // to catch the exception, free some stuff in their script, and
          // attempt the allocation again. In this case, we can also forgive and
          // clear CUDA's internal error state.
          cudaGetLastError();
        } else {
          // If the error's unrelated to memory allocation, we should throw
          // immediately.
          C10_CUDA_CHECK(p.err);
        }
        return false;
      }
      num_cuda_mallocs ++;
    }

    if (p.pool->owner_PrivatePool) {
      // The block is for a CUDA graph's PrivatePool.
      p.pool->owner_PrivatePool->cudaMalloc_count++;
    }

    total_allocated_memory += size;
    p.block = new Block(p.device(), p.stream(), size, p.pool, (char*)ptr);
    for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
      update_stat(stats.segment[stat_type], 1);
      update_stat(stats.reserved_bytes[stat_type], size);
    });
    if (size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_segments, 1);

    // p.block came from new, not cudaMalloc. It should not be nullptr here.
    TORCH_INTERNAL_ASSERT(p.block != nullptr && p.block->ptr != nullptr);
    return true;
  }

  /** Free one or more oversize blocks to the system allocator.  But only enough
   * **/
  /** to satisfy the target size **/
  bool release_available_cached_blocks(const AllocParams& p) {
    if (CachingAllocatorConfig::max_split_size() ==
        std::numeric_limits<size_t>::max())
      return false;
    BlockPool& pool = *p.pool;

    // because of std::unique_ptr, block cannot be trivially copied
    Block key(
        p.search_key.device,
        p.search_key.stream,
        p.search_key.size,
        p.search_key.pool,
        p.search_key.ptr);
    key.size = (key.size < CachingAllocatorConfig::max_split_size())
        ? CachingAllocatorConfig::max_split_size()
        : key.size;
    auto it = pool.blocks.lower_bound(&key);
    if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
      // No single block is large enough; free multiple oversize blocks,
      // starting with the largest
      if (it == pool.blocks.begin())
        return false;
      size_t totalReleased = 0;
      --it; // Back up one item.  Now on the largest block for the correct
            // stream
      while ((totalReleased < key.size) &&
             ((*it)->size >= CachingAllocatorConfig::max_split_size()) &&
             ((*it)->stream == p.stream())) {
        auto cur = it;
        totalReleased += (*it)->size;
        if (it != pool.blocks.begin()) {
          --it;
          release_block(*cur);
        } else {
          release_block(*cur);
          break;
        }
      }
      if (totalReleased < key.size)
        return false;
    } else {
      release_block(*it);
    }
    return true;
  }

  bool release_cached_blocks() {
    // First ensure that all blocks that can't currently be allocated due to
    // outstanding events are returned to the pool.
    synchronize_and_free_events();

    // Free all non-split cached blocks to system allocator
    release_blocks(large_blocks);
    release_blocks(small_blocks);

    for (auto it = graph_pools_freeable.begin();
         it != graph_pools_freeable.end();) {
      // See notifyCaptureDestroy for the strategy here.
      TORCH_INTERNAL_ASSERT(it->second->use_count == 0);
      release_blocks(it->second->small_blocks);
      release_blocks(it->second->large_blocks);
      if (it->second->cudaMalloc_count == 0) {
        auto erase_count = graph_pools.erase(it->first);
        TORCH_INTERNAL_ASSERT(erase_count == 1);
        it = graph_pools_freeable.erase(it);
      } else {
        ++it;
      }
    }

    return true;
  }

  void release_block(Block* block) {
    num_cuda_frees ++;
    C10_CUDA_CHECK(cudaFree((void*)block->ptr));
    total_allocated_memory -= block->size;

    auto* pool = block->pool;
    if (pool->owner_PrivatePool) {
      // The cudaFreed block belonged to a CUDA graph's PrivatePool.
      TORCH_INTERNAL_ASSERT(pool->owner_PrivatePool->cudaMalloc_count > 0);
      pool->owner_PrivatePool->cudaMalloc_count--;
    }

    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(get_stat_type_for_pool(*pool))] = true;
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.segment[stat_type], -1);
      update_stat(
          stats.reserved_bytes[stat_type],
          -static_cast<std::int64_t>(block->size));
    });
    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_segments, -1);
    if (block->history) {
      record_trace(
          TraceEntry::SEGMENT_FREE,
          int64_t(block->ptr),
          block->size,
          block->stream,
          block->history->h.context);
    }
    pool->blocks.erase(block);
    delete block;
  }

  void release_blocks(BlockPool& pool) {
    // Frees all non-split blocks
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
      Block* block = *it;
      ++it;
      if (!block->prev && !block->next) {
        release_block(block);
      }
    }
  }

  EventPool::Event create_event_internal(int idx) {
    // Leak the event pool to avoid shutdown issues.
    static auto* event_pool = new EventPool();
    return event_pool->get(idx);
  }

  void synchronize_and_free_events() {
    // Synchronize on outstanding events and then free associated blocks.

    // This function syncs, so capture should not be underway. Might as well
    // make sure capture-deferred end of life events get processed too.
    TORCH_INTERNAL_ASSERT(captures_underway == 0);
    insert_events_deferred_until_no_capture();

    for (auto& st : cuda_events) {
      for (auto& e : st.second) {
        EventPool::Event event = std::move(e.first);
        Block* block = e.second;

        C10_CUDA_CHECK(cudaEventSynchronize(*event));

        block->event_count--;
        if (block->event_count == 0) {
          free_block(block);
        }
      }
    }

    cuda_events.clear();
  }

  void insert_events(Block* block) {
    int prev_device;
    C10_CUDA_CHECK(cudaGetDevice(&prev_device));

    stream_set streams(std::move(block->stream_uses));
    AT_ASSERT(block->stream_uses.empty());
    for (auto& stream : streams) {
      C10_CUDA_CHECK(cudaSetDevice(stream.device_index()));

      EventPool::Event event =
          create_event_internal(static_cast<int>(stream.device_index()));
      C10_CUDA_CHECK(cudaEventRecord(*event, stream.stream()));

      block->event_count++;
      cuda_events[stream].emplace_back(std::move(event), block);
    }

    C10_CUDA_CHECK(cudaSetDevice(prev_device));
  }

  void insert_events_deferred_until_no_capture() {
    if (C10_UNLIKELY(needs_events_deferred_until_no_capture.size() > 0)) {
      for (auto* block : needs_events_deferred_until_no_capture) {
        TORCH_INTERNAL_ASSERT(!block->stream_uses.empty());
        insert_events(block);
      }
      needs_events_deferred_until_no_capture.clear();
    }
  }

  void process_events() {
    insert_events_deferred_until_no_capture();

    // Process outstanding cudaEvents. Events that are completed are
    // removed from the queue, and the 'event_count' for the
    // corresponding allocation is decremented. We maintain a separate
    // list of events per stream to avoid head-of-line delays if one
    // or more streams has long-running operations.

    // Iterate over different streams.
    for (auto it = cuda_events.begin(); it != cuda_events.end();) {
      // Iterate over this stream's (event, block) pairs.
      while (!it->second.empty()) {
        auto& e = it->second.front();
        EventPool::Event event = std::move(e.first);
        Block* block = e.second;

        cudaError_t err = C10_CUDA_ERROR_HANDLED(cudaEventQuery(*event));
        if (err == cudaErrorNotReady) {
          // ignore and clear the error if not ready
          cudaGetLastError();
          // Return the ownership of the Event (unique ptr)
          e.first = std::move(event);
          break;
        } else if (err != cudaSuccess) {
          C10_CUDA_CHECK(err);
        }

        block->event_count--;
        if (block->event_count == 0) {
          free_block(block);
        }
        it->second.pop_front();
      }

      if (it->second.empty()) {
        it = cuda_events.erase(it);
      } else {
        it++;
      }
    }
  }

  // Iterates over sizes of all memory blocks for given device in given pool
  void cache_info_aux(const BlockPool& pool, size_t* largest) {
    for (const auto& block : pool.blocks) {
      const auto blocksize = block->size;
      if (blocksize > *largest) {
        *largest = blocksize;
      }
    }
  }

  void record_trace(
      TraceEntry::Action action,
      int64_t addr,
      size_t size,
      cudaStream_t stream,
      std::shared_ptr<GatheredContext> context) {
    auto te = TraceEntry(
        action,
        addr,
        size,
        stream,
        alloc_trace_record_context_ ? std::move(context) : nullptr);
    if (alloc_trace->size() < alloc_trace_max_entries_) {
      alloc_trace->emplace_back(te);
    } else {
      (*alloc_trace)[alloc_trace_next++] = te;
      if (alloc_trace_next == alloc_trace_max_entries_) {
        alloc_trace_next = 0;
      }
    }
  }
};



} // namespace CUDACachingAllocator


// Copied from torch/csrc/cuda/memory_snapshot.cpp
using c10::Dict;
using c10::List;
using c10::IValue;
using torch::jit::Pickler;

using c10::cuda::CUDACachingAllocator::BlockInfo;
using c10::cuda::CUDACachingAllocator::History;
using c10::cuda::CUDACachingAllocator::SnapshotInfo;
using c10::cuda::CUDACachingAllocator::SegmentInfo;
using torch::CapturedTraceback;

static std::unique_ptr<CUDACachingAllocator::DeviceCachingAllocator> allocator(
    new CUDACachingAllocator::DeviceCachingAllocator());

std::string write_pickle(const IValue& v) {
  std::vector<char> result;
  {
    auto writer = [&](const char* data, size_t size) {
      result.insert(result.end(), data, data + size);
    };
    Pickler pickler(writer, nullptr, nullptr, nullptr, nullptr, false);
    pickler.protocol();
    pickler.pushIValue(v);
    pickler.stop();
  }
  return std::string(result.begin(), result.end());
}
Dict<IValue, IValue> new_dict() {
  return Dict<IValue, IValue>(c10::AnyType::get(), c10::AnyType::get());
}
c10::List<IValue> new_list() {
  return List<IValue>(c10::AnyType::get());
}

std::vector<IValue> ivalue_symbolize(
    std::vector<CapturedTraceback*>& to_symbolize) {
  // we dedup repeated to_symbolize objects to prevent
  // creating a bunch of duplicated frame objects
  std::unordered_map<CapturedTraceback*, uint64_t> cached_frames;
  std::vector<CapturedTraceback*> unique_frames;
  for (const auto& sc : to_symbolize) {
    auto it = cached_frames.find(sc);
    if (it == cached_frames.end()) {
      cached_frames.insert({sc, unique_frames.size()});
      unique_frames.push_back(sc);
    }
  }
  auto s = symbolize(unique_frames);

  IValue line_s = "line";
  IValue name_s = "name";
  IValue filename_s = "filename";
  std::vector<IValue> all_frames;
  for (const auto& f : s.all_frames) {
    auto d = new_dict();
    d.insert(name_s, f.funcname);
    d.insert(filename_s, f.filename);
    d.insert(line_s, int64_t(f.lineno));
    all_frames.emplace_back(std::move(d));
  }

  std::vector<IValue> py_unique_frames;
  for (const auto& t : s.tracebacks) {
    auto l = new_list();
    for (const auto& e : t) {
      l.push_back(all_frames.at(e));
    }
    py_unique_frames.push_back(std::move(l));
  }

  std::vector<IValue> result;
  for (const auto& sc : to_symbolize) {
    result.push_back(py_unique_frames.at(cached_frames.at(sc)));
  }
  return result;
}

std::shared_ptr<c10::GatheredContext> gather() {
  return CapturedTraceback::gather(true, true, false);
}

std::shared_ptr<c10::GatheredContext> gather_with_cpp() {
  return CapturedTraceback::gather(true, true, true);
}

CapturedTraceback* getFromContext(
    const std::shared_ptr<c10::GatheredContext>& x) {
  if (CapturedTraceback* sc = dynamic_cast<CapturedTraceback*>(x.get())) {
    return sc;
  }
  TORCH_CHECK(
      false,
      "attempting to gather stack context from the wrong StackContext type.");
}

std::string memory_snapshot_pickled() {
  IValue device_s = "device";
  IValue address_s = "address";
  IValue total_size_s = "total_size";
  IValue allocated_size_s = "allocated_size";
  IValue active_size_s = "active_size";
  IValue requested_size_s = "requested_size";
  IValue stream_s = "stream";
  IValue segment_type_s = "segment_type";
  IValue segment_pool_id = "segment_pool_id";
  IValue large_s = "large";
  IValue small_s = "small";
  IValue size_s = "size";
  IValue state_s = "state";
  IValue active_allocated_s = "active_allocated";
  IValue active_pending_free_s = "active_pending_free";
  IValue inactive_s = "inactive";
  IValue addr_s = "addr";
  IValue real_size_s = "real_size";
  IValue filename_s = "filename";
  IValue name_s = "name";
  IValue line_s = "line";
  IValue frames_s = "frames";
  IValue history_s = "history";
  IValue blocks_s = "blocks";

  auto empty_frames = new_list();

  std::vector<CapturedTraceback*> frame_tracebacks;
  std::vector<Dict<IValue, IValue>> frame_dict;

  const auto segmentInfoToDict = [&](const SegmentInfo& segmentInfo) {
    auto segmentDict = new_dict();
    segmentDict.insert(device_s, segmentInfo.device);
    segmentDict.insert(address_s, segmentInfo.address);
    segmentDict.insert(total_size_s, segmentInfo.total_size);
    segmentDict.insert(allocated_size_s, segmentInfo.allocated_size);
    segmentDict.insert(active_size_s, segmentInfo.active_size);
    segmentDict.insert(requested_size_s, segmentInfo.requested_size);
    segmentDict.insert(stream_s, int64_t(segmentInfo.stream));
    segmentDict.insert(
        segment_type_s, (segmentInfo.is_large ? large_s : small_s));
    segmentDict.insert(
        segment_pool_id,
        std::tuple<int64_t, int64_t>(segmentInfo.owner_private_pool_id));

    auto blocks = new_list();
    for (const auto& blockInfo : segmentInfo.blocks) {
      auto blockDict = new_dict();
      blockDict.insert(size_s, blockInfo.size);
      blockDict.insert(requested_size_s, blockInfo.requested_size);
      blockDict.insert(
          state_s,
          (blockInfo.allocated
               ? active_allocated_s
               : (blockInfo.active ? active_pending_free_s : inactive_s)));
      if (blockInfo.history.size()) {
        auto history = new_list();
        for (const History& h : blockInfo.history) {
          auto history_entry = new_dict();
          history_entry.insert(addr_s, (int64_t)h.addr);
          history_entry.insert(real_size_s, (int64_t)h.real_size);
          if (h.context) {
            frame_tracebacks.push_back(getFromContext(h.context));
            frame_dict.push_back(history_entry);
          }
          history.push_back(std::move(history_entry));
        }
        blockDict.insert(history_s, std::move(history));
      }
      blocks.push_back(blockDict);
    }
    segmentDict.insert(blocks_s, blocks);

    return segmentDict;
  };

  SnapshotInfo snapshot;
  auto snap = allocator->snapshot();
  snapshot.segments.insert(snapshot.segments.end(), snap.begin(), snap.end());

  auto segments = new_list();
  for (const auto& segmentInfo : snapshot.segments) {
    segments.push_back(segmentInfoToDict(segmentInfo));
  }

  auto traces = new_list();
  IValue action_s = "action";
  IValue alloc_s = "alloc";
  IValue free_requested_s = "free_requested";
  IValue free_completed_s = "free_completed";
  IValue segment_alloc_s = "segment_alloc";
  IValue segment_free_s = "segment_free";
  IValue snapshot_s = "snapshot";
  IValue oom_s = "oom";
  IValue device_free_s = "device_free";

  using namespace c10::cuda::CUDACachingAllocator;

  auto action_to_str = [&](TraceEntry::Action action) {
    switch (action) {
      case TraceEntry::ALLOC:
        return alloc_s;
      case TraceEntry::FREE_REQUESTED:
        return free_requested_s;
      case TraceEntry::FREE_COMPLETED:
        return free_completed_s;
      case TraceEntry::SEGMENT_ALLOC:
        return segment_alloc_s;
      case TraceEntry::SEGMENT_FREE:
        return segment_free_s;
      case TraceEntry::OOM:
        return oom_s;
      case TraceEntry::SNAPSHOT:
        return snapshot_s;
    }
    throw std::runtime_error("unreachable");
  };

  for (const auto& traceInfo : snapshot.device_traces) {
    auto trace = new_list();
    for (const auto& te : traceInfo) {
      auto trace_entry = new_dict();
      trace_entry.insert(action_s, action_to_str(te.action_));
      trace_entry.insert(
          TraceEntry::OOM == te.action_ ? device_free_s : addr_s, te.addr_);
      trace_entry.insert(size_s, (int64_t)te.size_);
      trace_entry.insert(stream_s, int64_t(te.stream_));
      if (te.context_) {
        auto sc = getFromContext(te.context_);
        frame_tracebacks.push_back(sc);
        frame_dict.push_back(trace_entry);
      }
      trace.push_back(trace_entry);
    }
    traces.push_back(trace);
  }

  auto result = new_dict();
  result.insert("segments", segments);
  result.insert("device_traces", traces);

  auto frames = ivalue_symbolize(frame_tracebacks);
  for (auto i : c10::irange(frames.size())) {
    frame_dict.at(i).insert(frames_s, frames.at(i));
  }

  return write_pickle(result);
}

} // namespace dynapipe

extern "C" {

using c10::cuda::MempoolId_t;

void* dynapipe_malloc(ssize_t size, int device, cudaStream_t stream) {
    return dynapipe::allocator->malloc(device, size, stream);
}

void dynapipe_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
    return dynapipe::allocator->free(ptr);
}

void dynapipe_set_memory_fraction(double fraction, int device) {
    dynapipe::allocator->setMemoryFraction(fraction);
}

void dynapipe_release_pool(int device, MempoolId_t mempool_id) {
    dynapipe::allocator->releasePool(mempool_id);
}

void dynapipe_reset() {
    dynapipe::allocator->emptyCache();
}

int dynapipe_get_num_cuda_mallocs() {
    return dynapipe::allocator->getNumCudaMallocs(); 
}

int dynapipe_get_num_cuda_frees() {
    return dynapipe::allocator->getNumCudaFrees();
}

int64_t dynapipe_get_peak_reserved_cuda_memory() {
    return dynapipe::allocator->getPeakReservedCudaMemory();
}

int64_t dynapipe_get_peak_allocated_cuda_memory() {
    return dynapipe::allocator->getPeakAllocatedCudaMemory();
}

int64_t dynapipe_get_peak_requested_cuda_memory() {
    return dynapipe::allocator->getPeakRequestedCudaMemory();
}

int64_t dynapipe_get_current_reserved_cuda_memory() {
    return dynapipe::allocator->getCurrentReservedCudaMemory();
}

int64_t dynapipe_get_current_allocated_cuda_memory() {
    return dynapipe::allocator->getCurrentAllocatedCudaMemory();
}

int64_t dynapipe_get_current_requested_cuda_memory() {
    return dynapipe::allocator->getCurrentRequestedCudaMemory();
}

void dynapipe_reset_peak_stats() {
    dynapipe::allocator->resetPeakStats();
}

void dynapipe_reset_accumulated_stats() {
    dynapipe::allocator->resetAccumulatedStats();
}

const char* dynapipe_get_memory_snapshot(size_t* size) {
    auto snapshot_str = dynapipe::memory_snapshot_pickled();
    *size = snapshot_str.size();
    return snapshot_str.c_str();
}

} // extern "C"