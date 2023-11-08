# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os

from dynapipe.model import SchedulerMinibatchSpec
from dynapipe.schedule_opt import get_available_schedulers, get_scheduler_class

# Reads in a microbatch specification json file, generate the resulting
# timeline after permutating the microbatches.


def parse_arguments():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--microbatch-spec",
        type=str,
        required=True,
        help="Path to microbatch specification json file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path where the output timeline json file will be stored."
        "Default to {microbatch_spec_name}"
        "_{schedule_simulator}_timeline.json.",
    )
    parser.add_argument(
        "-p",
        "--microbatch-permutation",
        type=str,
        help="Reorder the microbatches according to the permutation.",
    )
    parser.add_argument(
        "--schedule-simulator",
        type=str,
        choices=get_available_schedulers(),
        default="wait-free-cyclic",
        help="The schedule simulator to use. Defaults to wait-free-cyclic.",
    )

    args = parser.parse_args()
    if args.output is None:
        mbspec_basename = os.path.basename(args.microbatch_spec).rsplit(
            ".", 1
        )[0]
        args.output = (
            mbspec_basename + f"_{args.schedule_simulator}" + "_timeline.json"
        )
    if args.microbatch_permutation is not None:
        args.microbatch_permutation = [
            int(x) for x in args.microbatch_permutation.split(",")
        ]
    return args


if __name__ == "__main__":
    args = parse_arguments()

    # Load model specification
    with open(args.microbatch_spec, "r") as f:
        microbatch_spec = json.load(f)
    base_exec_times = microbatch_spec["base_exec_times"]
    per_microbatch_multiplier = microbatch_spec["per_microbatch_multiplier"]
    if args.microbatch_permutation is not None:
        assert len(args.microbatch_permutation) == len(
            per_microbatch_multiplier
        ), (
            "The length of the microbatch permutation "
            "({}) must be the same as the number "
            "of microbatches ({}).".format(
                len(args.microbatch_permutation), len(base_exec_times)
            )
        )
        new_per_microbatch_multiplier = []
        for i in range(len(per_microbatch_multiplier)):
            new_per_microbatch_multiplier.append(
                per_microbatch_multiplier[args.microbatch_permutation[i]]
            )
        per_microbatch_multiplier = new_per_microbatch_multiplier
    per_device_stage_assignment = microbatch_spec[
        "per_device_stage_assignment"
    ]
    per_stage_device_assignment = [-1] * len(base_exec_times)
    for dev, stages in enumerate(per_device_stage_assignment):
        for stage in stages:
            per_stage_device_assignment[stage] = dev
    for stage, dev in enumerate(
        per_stage_device_assignment[: len(per_stage_device_assignment) // 2]
    ):
        assert dev != -1, f"Stage {stage} is not assigned to any device."
        assert dev == per_stage_device_assignment[-stage - 1], (
            f"FW stage {stage} is assigned to device {dev} "
            f"but BW stage {-stage - 1} is assigned to "
            f"device {per_stage_device_assignment[-stage - 1]}."
        )

    # get scheduler
    fw_len = len(base_exec_times) // 2
    scheduler_params = SchedulerMinibatchSpec(
        base_exec_times[:fw_len],
        [0] * (fw_len - 1),
        [1000] * fw_len,
        [1000] * fw_len,
        per_stage_device_assignment[:fw_len],
        base_exec_times[fw_len:],
        bw_comm_times=[0] * (fw_len - 1),
    )
    scheduler_class = get_scheduler_class(args.schedule_simulator)
    simulator = scheduler_class(scheduler_params, separate_comm_stage=False)

    # run simulation
    timeline_json = simulator.schedule(
        len(per_microbatch_multiplier),
        microbatch_multiplier=per_microbatch_multiplier,
    )
    print("# Makespan: {} ms".format(simulator.get_makespan() / 1000.0))
    # save timeline
    with open(args.output, "w") as f:
        json.dump(timeline_json, f, indent=4)
