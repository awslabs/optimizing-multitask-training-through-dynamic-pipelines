# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sortedcontainers import SortedList


class AllocatorSimulator:
    def __init__(self, max_memory_mbytes) -> None:
        self.max_memory = max_memory_mbytes * 1e6

    def malloc(self, size):
        pass

    def free(self, ptr):
        pass


@dataclass
class TorchCachingAllocatorConfig:
    max_split_size: float = float("inf")
    garbage_collection_threshold: float = 0.0
    kMinBlockSize: int = 512
    # kSmallSize: int = 1048576
    kSmallSize: int = 0
    kSmallBuffer: int = 2097152
    kLargeBuffer: int = 20971520
    kMinLargeAlloc: int = 10485760
    kRoundLarge: int = 2097152
    kRoundUpPowerOfTwoIntervals: int = 16


@dataclass
class TorchBlock:
    stream: int
    size: int
    ptr: int = -1
    pool: "TorchBlockPool" = field(default=None, repr=False)
    allocated: bool = False
    requested_size: int = -1
    prev: "TorchBlock" = None
    next: "TorchBlock" = None

    @classmethod
    def compare_key(cls, x: "TorchBlock"):
        return x.stream, x.size, x.ptr

    def is_split(self):
        return self.prev is not None or self.next is not None


@dataclass
class TorchBlockPool:
    blocks: SortedList = field(
        default_factory=lambda: SortedList(key=TorchBlock.compare_key)
    )
    is_small: bool = False


@dataclass
class TorchAllocParams:
    size: int
    stream: int
    pool: TorchBlockPool
    alloc_size: int


class TorchCachingAllocatorSimulator(AllocatorSimulator):
    def __init__(
        self,
        max_memory_mbytes,
        allocator_config: TorchCachingAllocatorConfig = None,
    ) -> None:
        super().__init__(max_memory_mbytes)
        if allocator_config is None:
            allocator_config = TorchCachingAllocatorConfig()
        self.config = allocator_config
        self.allocated_bytes = 0
        self.peak_allocated_bytes = 0
        self.allocated_segments = 0
        self.peak_allocated_segments = 0
        self.backend_allocated_bytes = 0
        self.peak_backend_allocated_bytes = 0
        self.n_backend_mallocs = 0
        self.n_backend_frees = 0
        self.n_alloc_large_pool = 0
        self.n_alloc_small_pool = 0
        # we only increase cuda_ptr
        # this means we assume cuda malloc has zero fragmentation
        self._backend_ptr = 0
        self._backend_ptr_to_size = {}
        self._large_pool = TorchBlockPool(is_small=False)
        self._small_pool = TorchBlockPool(is_small=True)
        self._timestep = 0

    def _round_size(self, size: int):
        mb = self.config.kMinBlockSize
        if size < mb:
            return mb
        else:
            return mb * ((size + mb - 1) // mb)

    def _get_pool(self, size: int):
        if size <= self.config.kSmallSize:
            return self._small_pool
        else:
            return self._large_pool

    def _get_allocation_size(self, size: int):
        if size <= self.config.kSmallSize:
            return self.config.kSmallBuffer
        elif size < self.config.kMinLargeAlloc:
            return self.config.kLargeBuffer
        else:
            rl = self.config.kRoundLarge
            return rl * ((size + rl - 1) // rl)

    def _get_free_block(self, params: TorchAllocParams):
        pool = params.pool
        stream = params.stream
        size = params.size
        block_index = pool.blocks.bisect_left(
            TorchBlock(stream=stream, size=size)
        )
        if (
            block_index == len(pool.blocks)
            or pool.blocks[block_index].stream != stream
        ):
            return None
        block: TorchBlock = pool.blocks[block_index]
        # Do not return an oversized block for a large request
        if (size < self.config.max_split_size) and (
            block.size >= self.config.max_split_size
        ):
            return None
        # Allow oversized block size to be rounded up but within a limit
        if (size >= self.config.max_split_size) and (
            block.size >= size + self.config.kLargeBuffer
        ):
            return None
        pool.blocks.remove(block)
        return block

    def _alloc_block(self, params: TorchAllocParams):
        size = params.alloc_size
        backend_ptr = self.backend_malloc(size)
        if backend_ptr == -1:
            return None
        return TorchBlock(
            stream=params.stream, size=size, ptr=backend_ptr, pool=params.pool
        )

    def _release_block(self, block: TorchBlock):
        if block.ptr != -1:
            self.backend_free(block.ptr)
        pool = block.pool
        pool.blocks.remove(block)

    def _release_available_cached_blocks(self, params: TorchAllocParams):
        if self.config.max_split_size == float("inf"):
            return False
        pool = params.pool
        key_block = TorchBlock(stream=params.stream, size=params.size)
        if key_block.size < self.config.max_split_size:
            key_block.size = self.config.max_split_size
        key_index = pool.blocks.bisect_left(key_block)
        if (
            key_index == len(pool.blocks)
            or pool.blocks[key_index].stream != params.stream
        ):
            # No single block is large enough; free multiple oversize blocks,
            # starting with the largest
            if key_index == 0:
                return False
            total_released = 0
            key_index -= 1
            while (
                total_released < key_block.size
                and pool.blocks[key_index].size >= self.config.max_split_size
                and pool.blocks[key_index].stream == params.stream
            ):
                cur_block = pool.blocks[key_index]
                total_released += cur_block.size
                if key_index != 0:
                    key_index -= 1
                    self._release_block(cur_block)
                else:
                    self._release_block(cur_block)
                    break
            if total_released < key_block.size:
                return False
        else:
            self._release_block(pool.blocks[key_index])
        return True

    def _release_blocks(self, pool: TorchBlockPool):
        for block in pool.blocks.copy():
            if block.prev is None and block.next is None:
                self._release_block(block)

    def _release_cached_blocks(self):
        self._release_blocks(self._large_pool)
        self._release_blocks(self._small_pool)

    def _should_split(self, block: TorchBlock, size: int):
        remaining = block.size - size
        if block.pool.is_small:
            return remaining >= self.config.kMinBlockSize
        else:
            return (
                size < self.config.max_split_size
                and remaining > self.config.kSmallSize
            )

    def _alloc_found_block(
        self,
        block: TorchBlock,
        params: TorchAllocParams,
        orig_size: int,
        split_remainder: bool,
    ):
        size = params.size
        pool = params.pool
        stream = params.stream
        assert block is not None and block.ptr != -1
        if split_remainder:
            remaining = block
            block = TorchBlock(
                stream=stream, size=size, ptr=block.ptr, pool=pool
            )
            block.prev = remaining.prev
            if block.prev:
                block.prev.next = block
            block.next = remaining
            remaining.prev = block
            remaining.ptr += size
            remaining.size -= size
            pool.blocks.add(remaining)
        block.allocated = True
        block.requested_size = orig_size
        assert block.size > 0
        self.allocated_bytes += block.size
        self.peak_allocated_bytes = max(
            self.peak_allocated_bytes, self.allocated_bytes
        )
        return block

    # combine previously split blocks. returns the size of the subsumed block,
    # or 0 on failure.
    def _try_merge_blocks(
        self, dst: TorchBlock, src: TorchBlock, pool: TorchBlockPool
    ):
        if not src or src.allocated:
            return 0
        assert src.is_split() and dst.is_split()
        if dst.prev == src:  # [src, dst]
            dst.ptr = src.ptr
            dst.prev = src.prev
            if dst.prev:
                dst.prev.next = dst
        else:  # [dst, src]
            dst.next = src.next
            if dst.next:
                dst.next.prev = dst
        subsumed_size = src.size
        dst.size += subsumed_size
        pool.blocks.remove(src)
        return subsumed_size

    def _free_block(self, block: TorchBlock):
        assert not block.allocated
        pool = block.pool

        merge_candidates = (block.prev, block.next)
        for candidate in merge_candidates:
            self._try_merge_blocks(block, candidate, pool)
        pool.blocks.add(block)

    def backend_malloc(self, size):
        if self.backend_allocated_bytes + size > self.max_memory:
            return -1
        self._backend_ptr += size
        self.backend_allocated_bytes += size
        self._backend_ptr_to_size[self._backend_ptr] = size
        self.allocated_segments += 1
        self.peak_allocated_segments = max(
            self.peak_allocated_segments, self.allocated_segments
        )
        self.peak_backend_allocated_bytes = max(
            self.peak_backend_allocated_bytes, self.backend_allocated_bytes
        )
        self.n_backend_mallocs += 1
        return self._backend_ptr

    def backend_free(self, ptr):
        assert ptr in self._backend_ptr_to_size
        size = self._backend_ptr_to_size[ptr]
        self.backend_allocated_bytes -= size
        self.allocated_segments -= 1
        self.n_backend_frees += 1

    def malloc(self, size, stream=0):
        size = self._round_size(size)
        pool = self._get_pool(size)
        if pool.is_small:
            self.n_alloc_small_pool += 1
        else:
            self.n_alloc_large_pool += 1
        alloc_size = self._get_allocation_size(size)
        param = TorchAllocParams(
            size=size, stream=stream, pool=pool, alloc_size=alloc_size
        )
        block = self._get_free_block(param)
        # we don't simulate free block callbacks for now
        if block is None:
            # attempt allocation
            block = self._alloc_block(param)
            if block is None:
                self._release_available_cached_blocks(param)
                block = self._alloc_block(param)
            if block is None:
                self._release_cached_blocks()
                block = self._alloc_block(param)
            if block is None:
                # we are out of memory
                raise RuntimeError("Out of Memory")
        assert block is not None
        should_split_remainder = self._should_split(block, param.size)
        self._timestep += 1
        return self._alloc_found_block(
            block, param, size, should_split_remainder
        )

    def free(self, block: TorchBlock):
        block.allocated = False
        orig_size = block.size
        self._free_block(block)
        self.allocated_bytes -= orig_size
        self._timestep += 1

    def clear_cache(self):
        self._release_cached_blocks()

    def reset_peak_stats(self):
        self.peak_allocated_bytes = 0
        self.peak_allocated_segments = 0
        self.peak_backend_allocated_bytes = 0
