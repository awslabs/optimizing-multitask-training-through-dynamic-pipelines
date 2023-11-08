# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from collections import defaultdict

# Parse the last generated EP from the planner logs for debugging purposes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str, help="Path to the log directory")
    return parser.parse_args()


def get_last_generated_ep(planner_logs):
    all_generated_eps = []
    last_ep_per_worker = {}
    for worker_id, lines in planner_logs.items():
        for line in lines:
            if "Pushing EP" in line:
                iteration = int(line.split()[-4])
                all_generated_eps.append(iteration)
                last_ep_per_worker[worker_id] = iteration
    max_iter = max(all_generated_eps)
    assert sorted(all_generated_eps) == list(range(max_iter + 1))
    return max_iter, last_ep_per_worker


def get_last_received_ep(worker_logs):
    all_received_eps = []
    last_ep_per_worker = {}
    for worker_id, lines in worker_logs.items():
        for line in lines:
            if "Got data for" in line:
                iteration = int(line.split()[-1][:-1])
                all_received_eps.append(iteration)
                last_ep_per_worker[worker_id] = iteration
    max_iter = max(all_received_eps)
    assert sorted(all_received_eps) == list(range(max_iter + 1))
    return max_iter, last_ep_per_worker


def parse_rank_from_log_path(log_path):
    log_path = os.path.basename(log_path).split(".")[0]
    rank = int(log_path.split("_")[0][1:])
    virtual_rank = int(log_path.split("_")[1][2:])
    worker_id = int(log_path.split("_")[2][1:])
    return rank, virtual_rank, worker_id


def main(args):
    # read dataloader logs
    dataloader_log_paths = os.listdir(os.path.join(args.log_dir, "dataloader"))
    dataloader_log_paths = [
        os.path.join(args.log_dir, "dataloader", log)
        for log in dataloader_log_paths
    ]
    dataloader_logs = {}
    for log in dataloader_log_paths:
        rank, virtual_rank, worker_id = parse_rank_from_log_path(log)
        with open(log, "r") as f:
            dataloader_logs[(rank, virtual_rank, worker_id)] = f.readlines()
    planner_logs = {}
    worker_logs = defaultdict(dict)
    for rank, virtual_rank, worker_id in dataloader_logs.keys():
        if rank == 0 and virtual_rank == 0:
            planner_logs[worker_id] = dataloader_logs[
                (rank, virtual_rank, worker_id)
            ]
        else:
            worker_logs[(rank, virtual_rank)][worker_id] = dataloader_logs[
                (rank, virtual_rank, worker_id)
            ]
    last_generated_ep, _ = get_last_generated_ep(planner_logs)
    print(f"Last generated EP on planner: {last_generated_ep}")
    for (rank, virtual_rank), worker_dict in worker_logs.items():
        last_received_ep, _ = get_last_received_ep(worker_dict)
        print(
            f"Last received EP on worker r{rank}-vr{virtual_rank}: "
            f"{last_received_ep}"
        )


if __name__ == "__main__":
    main(parse_args())
