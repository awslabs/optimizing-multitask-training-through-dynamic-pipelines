# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import glob
import os

import torch

_allocator = None


class DynaPipeCachingAllocator:
    # wrapper for the C++ allocator
    def __init__(self, dll):
        self.dll = dll
        self._c_peak_reserved_cuda_memory = self.get_func(
            "dynapipe_get_peak_reserved_cuda_memory"
        )
        self._c_peak_reserved_cuda_memory.restype = ctypes.c_int64
        self._c_peak_allocated_cuda_memory = self.get_func(
            "dynapipe_get_peak_allocated_cuda_memory"
        )
        self._c_peak_allocated_cuda_memory.restype = ctypes.c_int64
        self._c_peak_requested_cuda_memory = self.get_func(
            "dynapipe_get_peak_requested_cuda_memory"
        )
        self._c_peak_requested_cuda_memory.restype = ctypes.c_int64
        self._c_current_reserved_cuda_memory = self.get_func(
            "dynapipe_get_current_reserved_cuda_memory"
        )
        self._c_current_reserved_cuda_memory.restype = ctypes.c_int64
        self._c_current_allocated_cuda_memory = self.get_func(
            "dynapipe_get_current_allocated_cuda_memory"
        )
        self._c_current_allocated_cuda_memory.restype = ctypes.c_int64
        self._c_current_requested_cuda_memory = self.get_func(
            "dynapipe_get_current_requested_cuda_memory"
        )
        self._c_current_requested_cuda_memory.restype = ctypes.c_int64
        self._c_get_memory_snapshot = self.get_func(
            "dynapipe_get_memory_snapshot"
        )
        self._c_get_memory_snapshot.argtypes = [
            ctypes.POINTER(ctypes.c_size_t)
        ]
        self._c_get_memory_snapshot.restype = ctypes.POINTER(ctypes.c_char)

    def get_func(self, func_name):
        return getattr(self.dll, func_name)

    def get_func_ptr(self, func_name):
        return ctypes.cast(getattr(self.dll, func_name), ctypes.c_void_p).value

    def num_cuda_mallocs(self):
        return self.get_func("dynapipe_get_num_cuda_mallocs")()

    def num_cuda_frees(self):
        return self.get_func("dynapipe_get_num_cuda_frees")()

    def reset_peak_stats(self):
        return self.get_func("dynapipe_reset_peak_stats")()

    def reset_accumulated_stats(self):
        return self.get_func("dynapipe_reset_accumulated_stats")()

    def peak_reserved_cuda_memory(self):
        return self._c_peak_reserved_cuda_memory()

    def peak_allocated_cuda_memory(self):
        return self._c_peak_allocated_cuda_memory()

    def peak_requested_cuda_memory(self):
        return self._c_peak_requested_cuda_memory()

    def current_reserved_cuda_memory(self):
        return self._c_current_reserved_cuda_memory()

    def current_allocated_cuda_memory(self):
        return self._c_current_allocated_cuda_memory()

    def current_requested_cuda_memory(self):
        return self._c_current_requested_cuda_memory()

    def get_memory_snapshot(self):
        buf_size = ctypes.c_size_t()
        c_ptr = self._c_get_memory_snapshot(ctypes.byref(buf_size))
        return ctypes.string_at(c_ptr, buf_size.value)


def find_library():
    """Find the compiled library."""
    # Find the library path.
    library_path = os.path.join(
        os.path.dirname(__file__), "build", "lib*", "dynapipe_cuda_allocator.*"
    )
    library_path = glob.glob(library_path)[0]
    return library_path


def get_allocator():
    """Get the custom allocator wrapper."""
    global _allocator
    assert _allocator is not None, "Allocator not overriden"
    return _allocator


def _remove_args(func):
    """Remove the arguments from the function."""

    def wrapper(*args, **kwargs):
        return func()

    return wrapper


def override_allocator():
    """Override the default PyTorch allocator with the custom allocator."""
    global _allocator
    if _allocator is not None:
        return
    # Load the library.
    library_path = find_library()
    dll = ctypes.CDLL(library_path)
    allocator_wrapper = DynaPipeCachingAllocator(dll)
    new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
        library_path, "dynapipe_malloc", "dynapipe_free"
    )
    new_alloc._allocator.set_memory_fraction_fn(
        allocator_wrapper.get_func_ptr("dynapipe_set_memory_fraction")
    )
    new_alloc._allocator.set_release_pool(
        allocator_wrapper.get_func_ptr("dynapipe_release_pool")
    )
    new_alloc._allocator.set_reset_fn(
        allocator_wrapper.get_func_ptr("dynapipe_reset")
    )
    torch.cuda.memory.change_current_allocator(new_alloc)
    _allocator = allocator_wrapper
    # override torch's get memory stats function to avoid errors
    # Note: all args (e.g. device_index) are removed, all stats are
    # only for current device. Use with caution.
    torch.cuda.memory_allocated = _remove_args(
        _allocator.current_allocated_cuda_memory
    )
    torch.cuda.max_memory_allocated = _remove_args(
        _allocator.peak_allocated_cuda_memory
    )
    torch.cuda.memory_reserved = _remove_args(
        _allocator.current_reserved_cuda_memory
    )
    torch.cuda.max_memory_reserved = _remove_args(
        _allocator.peak_reserved_cuda_memory
    )
    torch.cuda.reset_accumulated_memory_stats = _remove_args(
        _allocator.reset_accumulated_stats
    )
    torch.cuda.reset_peak_memory_stats = _remove_args(
        _allocator.reset_peak_stats
    )
