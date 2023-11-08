# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json

import numpy as np

# Generate random microbatch specifications using specified distribution


def parse_arguments():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--microbatches",
        type=int,
        required=True,
        help="Number of microbatches.",
    )
    parser.add_argument(
        "-s", "--stages", type=int, required=True, help="Number of stages."
    )
    parser.add_argument(
        "-d",
        "--device-assignment",
        type=str,
        required=True,
        help="Assignment of stages to devices.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path where the output config json file will be stored.",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        choices=["normal", "geometric"],
        help="Distribution used to generate the multipliers.",
    )
    parser.add_argument(
        "-p",
        "--success-probability",
        type=float,
        default=0.5,
        help="Success probability for the geometric distribution.",
    )
    parser.add_argument(
        "-std",
        "--stddev",
        type=float,
        default=1.0,
        help="Standard deviation of the microbatch multipliers.",
    )

    args = parser.parse_args()
    args.device_assignment = [
        int(x) for x in args.device_assignment.split(",")
    ]
    if args.output is None:
        args.output = "mb_spec"
        if args.distribution == "geometric":
            dist_spec = "-p" + str(args.success_probability)
        else:
            dist_spec = "-std" + str(args.stddev)
        args.output += f"_{args.distribution}{dist_spec}"
        args.output += (
            f"_m{args.microbatches}s{args.stages}p{args.stddev}.json"
        )
    return args


if __name__ == "__main__":
    args = parse_arguments()

    # Load model specification
    base_exec_times = [1000] * args.stages
    microbatch_multipliers = []
    for m in range(args.microbatches):
        mult = 0
        while mult <= 0:
            if args.distribution == "normal":
                mult = np.random.normal(1, args.stddev)
            elif args.distribution == "geometric":
                mult = np.random.geometric(args.success_probability)
            else:
                raise ValueError(
                    "Unknown distribution: {}".format(args.distribution)
                )
        microbatch_multipliers.append([mult] * args.stages)
    per_stage_device_assignment = args.device_assignment
    per_device_stage_assignment = [
        [] for _ in range(max(args.device_assignment) + 1)
    ]
    for i, d in enumerate(per_stage_device_assignment):
        per_device_stage_assignment[d].append(i)

    json_dict = {
        "base_exec_times": base_exec_times,
        "per_microbatch_multiplier": microbatch_multipliers,
        "per_device_stage_assignment": per_device_stage_assignment,
    }
    with open(args.output, "w") as f:
        json.dump(json_dict, f, indent=4)
