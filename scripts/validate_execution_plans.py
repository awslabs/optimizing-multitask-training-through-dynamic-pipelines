# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import pickle

from dynapipe.pipe.utils import check_deadlock

# Validate execution plans by checking for deadlocks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("execution_plan", type=str)
    return parser.parse_args()


def main(args):
    with open(args.execution_plan, "rb") as f:
        eps = pickle.load(f)
    check_deadlock(eps)


if __name__ == "__main__":
    main(parse_args())
