# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="dynapipe_cuda_allocator",
    ext_modules=[
        CUDAExtension(
            name="dynapipe_cuda_allocator",
            sources=["cuda_caching_allocator.cpp"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
