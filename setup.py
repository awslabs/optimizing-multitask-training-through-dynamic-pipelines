# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pybind11.setup_helpers import build_ext, intree_extensions
from setuptools import find_packages, setup

ext_modules = intree_extensions(
    ["dynapipe/data_opt/dp_helper.cpp"],
)

setup(
    name="dynapipe",
    version="0.0.1",
    packages=find_packages(),
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
)
