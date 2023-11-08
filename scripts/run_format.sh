#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# If any command fails, exit immediately with that command's exit status
set -eo pipefail

# Run isort against all code
isort . --profile black --line-length 79
echo "isort on code passed!"

# Run black against all code
black . --line-length 79
echo "black on code passed!"

# Run flake8 against all code in the `source_code` directory
flake8 . --extend-ignore F405,E203
echo "flake8 on code passed!"

clang-format -i ./dynapipe/data_opt/dp_helper.cpp