# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import random
import time

import torch
from tqdm import trange

from dynapipe.memory_opt.host_caching_allocator import HostCachingAllocator


def test_host_caching_allocator(preallocate=False):
    hca = HostCachingAllocator()
    dtype_dict = {
        1: torch.uint8,
        2: torch.int8,
        3: torch.float32,
        4: torch.float16,
    }
    pinned_tensors = []
    torch_tensors = []
    random.seed(42)
    torch.manual_seed(0)
    total_malloc_time = 0
    if preallocate:
        x = hca.malloc((32 * 1024 * 1024 * 4 * 20,), torch.uint8)  # noqa: F841
        del x
    for step in trange(1000):
        mbs = random.randint(1, 32)
        enc_seqlen = random.randint(1, 1024)
        dec_seqlen = random.randint(1, 1024)
        shape = (mbs, enc_seqlen, dec_seqlen)
        dtype = dtype_dict[random.randint(1, 4)]
        start = time.time()
        pinned_tensor = hca.malloc(shape, dtype)
        total_malloc_time += time.time() - start
        if dtype == torch.uint8 or dtype == torch.int8:
            torch_tensor = torch.randint(
                0, 128, shape, dtype=dtype, device="cpu"
            )
        else:
            torch_tensor = torch.rand(shape, dtype=dtype, device="cpu")
        pinned_tensor.copy_(torch_tensor)
        pinned_tensors.append(pinned_tensor)
        torch_tensors.append(torch_tensor)

        if step > 20:
            # free some tensors
            avail_idx = [
                i for i, x in enumerate(pinned_tensors) if x is not None
            ]
            idx = random.choice(avail_idx)
            # check if the tensor is still the same
            assert torch.allclose(pinned_tensors[idx], torch_tensors[idx])
            pinned_tensors[idx] = None
            torch_tensors[idx] = None
    print(f"total mallcs: {hca.n_backend_mallocs}")
    print(f"total malloc time: {total_malloc_time:.2f} seconds")


if __name__ == "__main__":
    # test_host_caching_allocator(preallocate=False)
    test_host_caching_allocator(preallocate=True)
