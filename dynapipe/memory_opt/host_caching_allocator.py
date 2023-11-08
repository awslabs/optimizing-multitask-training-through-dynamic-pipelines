# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import collections
import queue
from typing import Tuple, Union

import torch
from torch._utils import ExceptionWrapper
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL

from dynapipe.memory_opt.allocation_simulator import (
    TorchCachingAllocatorSimulator,
)


class HostCachingAllocatorPtr:
    # This is used as a index to find and access the allocated torch
    # tensors. This acts as a normal pointer which supports addition.
    def __init__(self, tensor_hash: int, offset):
        if not isinstance(tensor_hash, int):
            raise RuntimeError("tensor_hash must be an integer")
        self.tensor_hash = tensor_hash
        self.offset = offset

    def __hash__(self) -> int:
        return hash((self.tensor_hash, self.offset))

    def __add__(self, other: Union["HostCachingAllocatorPtr", int]):
        if not isinstance(other, (HostCachingAllocatorPtr, int)):
            raise RuntimeError(
                "Cannot add a HostCachingAllocatorPtr and a non integer"
            )
        if isinstance(other, int):
            return HostCachingAllocatorPtr(
                self.tensor_hash, self.offset + other
            )
        if self.tensor_hash != hash(other.tensor_hash):
            raise RuntimeError("Cannot add two different tensors")
        return HostCachingAllocatorPtr(self.tensor_hash, self.offset + other)

    def __radd__(self, other: Union["HostCachingAllocatorPtr", int]):
        return self.__add__(other)

    def __eq__(self, other: "HostCachingAllocatorPtr"):
        if not isinstance(other, HostCachingAllocatorPtr):
            return False
        return (
            self.tensor_hash == other.tensor_hash
            and self.offset == other.offset
        )

    def __lt__(self, other: "HostCachingAllocatorPtr"):
        if not isinstance(other, HostCachingAllocatorPtr):
            # it may be used to compare with nullptr (-1)
            # we always want to return false in this case
            return False
        if self.tensor_hash == other.tensor_hash:
            return self.offset < other.offset
        return self.tensor_hash < other.tensor_hash

    def __gt__(self, other: "HostCachingAllocatorPtr"):
        if self.tensor_hash == other.tensor_hash:
            return self.offset > other.offset
        return self.tensor_hash > other.tensor_hash

    def __le__(self, other: "HostCachingAllocatorPtr"):
        if self.tensor_hash == other.tensor_hash:
            return self.offset <= other.offset
        return self.tensor_hash <= other.tensor_hash

    def __ge__(self, other: "HostCachingAllocatorPtr"):
        if self.tensor_hash == other.tensor_hash:
            return self.offset >= other.offset
        return self.tensor_hash >= other

    def __ne__(self, other: "HostCachingAllocatorPtr"):
        if not isinstance(other, HostCachingAllocatorPtr):
            return True
        return (
            self.tensor_hash != other.tensor_hash
            or self.offset != other.offset
        )


class DestructionCallback:
    # This is attached to the tensor as an attribute.
    # The destruction callback is called when the tensor is deleted.
    def __init__(self, callback):
        self.callback = callback

    def __del__(self):
        self.callback()


class HostCachingAllocator(TorchCachingAllocatorSimulator):
    def __init__(self, allocator_config=None) -> None:
        super().__init__(float("inf"), allocator_config)
        self._segment_map = {}
        self._block_map = {}

    # override
    def backend_malloc(self, size):
        # we don't need to manually pose a limit on the memory
        # _backend_ptr is not used
        tensor = torch.empty(
            size, dtype=torch.uint8, device="cpu", pin_memory=True
        )
        tensor_hash = hash(tensor)
        self._segment_map[tensor_hash] = tensor
        tensor_ptr = HostCachingAllocatorPtr(tensor_hash, 0)
        self.backend_allocated_bytes += size
        self._backend_ptr_to_size[tensor_ptr] = size
        self.allocated_segments += 1
        self.peak_allocated_segments = max(
            self.peak_allocated_segments, self.allocated_segments
        )
        self.peak_backend_allocated_bytes = max(
            self.peak_backend_allocated_bytes, self.backend_allocated_bytes
        )
        self.n_backend_mallocs += 1
        return tensor_ptr

    def backend_free(self, ptr):
        if not isinstance(ptr, HostCachingAllocatorPtr):
            raise RuntimeError("ptr must be a HostCachingAllocatorPtr")
        if ptr not in self._backend_ptr_to_size:
            raise RuntimeError("ptr is not a valid pointer")
        tensor_hash = ptr.tensor_hash
        if tensor_hash not in self._segment_map:
            raise RuntimeError("tensor_hash do not map to a allocated tensor")
        del self._segment_map[tensor_hash]
        size = self._backend_ptr_to_size[ptr]
        del self._backend_ptr_to_size[ptr]
        self.backend_allocated_bytes -= size
        self.allocated_segments -= 1
        self.n_backend_frees += 1

    # wrapper around the original malloc, returning a pinned tensor
    def malloc(
        self, shape: Tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        size = 1
        for dim in shape:
            size *= dim
        size *= torch._utils._element_size(dtype)
        block = super().malloc(size)
        ptr: HostCachingAllocatorPtr = block.ptr
        tensor_hash = ptr.tensor_hash
        if tensor_hash not in self._segment_map:
            raise RuntimeError("tensor_hash do not map to a allocated tensor")
        tensor = self._segment_map[tensor_hash]
        self._block_map[ptr] = block
        tensor_view = (
            tensor[ptr.offset : ptr.offset + size].view(dtype).view(shape)
        )

        def destructor():
            self.free(ptr)

        tensor_view._destructor = DestructionCallback(destructor)
        return tensor_view

    def free(self, ptr: HostCachingAllocatorPtr):
        if ptr not in self._block_map:
            raise RuntimeError("ptr is not a valid pointer")
        block = self._block_map[ptr]
        del self._block_map[ptr]
        super().free(block)


# copied from torch.utils.data._utils.pin_memory
def _pin_memory_loop(in_queue, out_queue, device_id, done_event, device):
    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.
    torch.set_num_threads(1)

    if device == "cuda":
        torch.cuda.set_device(device_id)
    elif device == "xpu":
        torch.xpu.set_device(device_id)  # type: ignore[attr-defined]

    hca = HostCachingAllocator()
    # pre-allocate a large chunk of memory
    x = hca.malloc((1024 * 1024 * 1024,), torch.uint8)
    del x

    def do_one_step():
        try:
            r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            return
        idx, data = r
        if not done_event.is_set() and not isinstance(data, ExceptionWrapper):
            try:
                data = _pin_memory(hca, data, device)
            except Exception:
                data = ExceptionWrapper(
                    where="in pin memory thread for device {}".format(
                        device_id
                    )
                )
            r = (idx, data)
        while not done_event.is_set():
            try:
                out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue

    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details
    # on the logic of this function.
    while not done_event.is_set():
        # Make sure that we don't preserve any object from one iteration
        # to the next
        do_one_step()


# monkey patches
# copied from torch.utils.data._utils.pin_memory
def _pin_memory(hca: HostCachingAllocator, data, device=None):
    if isinstance(data, torch.Tensor):
        pinned_tensor = hca.malloc(data.shape, data.dtype)
        pinned_tensor.copy_(data)
        return pinned_tensor
    elif isinstance(data, str):
        return data
    elif isinstance(data, collections.abc.Mapping):
        try:
            return type(data)(
                {
                    k: _pin_memory(hca, sample, device)
                    for k, sample in data.items()
                }
            )
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {
                k: _pin_memory(hca, sample, device)
                for k, sample in data.items()
            }
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return type(data)(
            *(_pin_memory(hca, sample, device) for sample in data)
        )
    elif isinstance(data, tuple):
        return [
            _pin_memory(hca, sample, device) for sample in data
        ]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence):
        try:
            return type(data)(
                [_pin_memory(hca, sample, device) for sample in data]
            )
        except TypeError:
            # The sequence type may not support `__init__(iterable)`
            # (e.g., `range`).
            return [_pin_memory(hca, sample, device) for sample in data]
    elif hasattr(data, "pin_memory"):
        raise RuntimeError("Custom pin_memory function not supported.")
    else:
        return data


def apply_monkey_patch():
    # monkey patch the pin_memory function to use our caching HCA
    torch.utils.data._utils.pin_memory._pin_memory_loop = _pin_memory_loop
