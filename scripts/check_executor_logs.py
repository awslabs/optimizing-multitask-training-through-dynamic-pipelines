# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from datetime import datetime

# Checks executor logs for slow instructions and mismatched instructions,
# for debugging purposes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_paths", nargs="+", default=[])
    return parser.parse_args()


def main(args):
    for log in args.log_paths:
        rank = log.split("_")[-1].split(".")[0]
        mismatched_instrs = {}
        last_time = None
        with open(log, "r") as f:
            for line in f:
                datetime_str = line.split(" ")[1].split(",")[0]
                time = datetime.strptime(datetime_str, "%H:%M:%S")
                if last_time is not None:
                    timedelta = time - last_time
                    if timedelta.total_seconds() > 1:
                        print(
                            f"Rank {rank} has slow instr "
                            f"({timedelta.total_seconds()} seconds):\n\t{line}"
                        )
                last_time = time
                if "Executing instruction:" in line:
                    instr = line.split(":")[-1].split("\x1b")[0].strip()
                    if instr not in mismatched_instrs:
                        mismatched_instrs[instr] = 1
                    else:
                        mismatched_instrs[instr] += 1
                elif "finished" in line:
                    instr = (
                        line.split(":")[-1]
                        .split("finished")[0]
                        .split("\x1b")[0]
                        .strip()
                    )
                    mismatched_instrs[instr] -= 1
                    assert mismatched_instrs[instr] >= 0
                    if mismatched_instrs[instr] == 0:
                        del mismatched_instrs[instr]
        for instr, cnt in mismatched_instrs.items():
            print(
                f"Rank {rank} has mismatched instructions: "
                f"{instr}, repeat cnt: {cnt}"
            )


if __name__ == "__main__":
    main(parse_args())
