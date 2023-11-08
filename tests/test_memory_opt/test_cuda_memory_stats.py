# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
from pprint import pprint

import torch

from dynapipe.memory_opt.cuda_caching_allocator import (
    get_allocator,
    override_allocator,
)


def test_cuda_stats():
    override_allocator()
    allocator = get_allocator()

    a = torch.zeros((1024, 1024), device="cuda")
    del a
    b = torch.zeros((1024, 1024), device="cuda")  # noqa: F841
    c = torch.zeros((64, 32), device="cuda")  # noqa: F841

    pickled_snapshot = allocator.get_memory_snapshot()

    py_snapshot = pickle.loads(pickled_snapshot)
    pprint(py_snapshot["segments"])

    print(
        "Peak reserved memory: {} MB".format(
            allocator.peak_reserved_cuda_memory() / 1e6
        )
    )
    print(
        "Peak allocated memory: {} MB".format(
            allocator.peak_allocated_cuda_memory() / 1e6
        )
    )
    print(
        "Peak requested memory: {} MB".format(
            allocator.peak_requested_cuda_memory() / 1e6
        )
    )
    print(
        "Current reserved memory: {} MB".format(
            allocator.current_reserved_cuda_memory() / 1e6
        )
    )
    print(
        "Current allocated memory: {} MB".format(
            allocator.current_allocated_cuda_memory() / 1e6
        )
    )
    print(
        "Current requested memory: {} MB".format(
            allocator.current_requested_cuda_memory() / 1e6
        )
    )
    del b
    del c

    # reset stats
    allocator.reset_peak_stats()
    print(
        "Peak reserved memory: {} MB".format(
            allocator.peak_reserved_cuda_memory() / 1e6
        )
    )
    print(
        "Peak allocated memory: {} MB".format(
            allocator.peak_allocated_cuda_memory() / 1e6
        )
    )
    print(
        "Peak requested memory: {} MB".format(
            allocator.peak_requested_cuda_memory() / 1e6
        )
    )


if __name__ == "__main__":
    test_cuda_stats()
