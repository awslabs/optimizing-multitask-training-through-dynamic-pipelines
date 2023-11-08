# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
from shift_trace_json import (
    construct_exec_time_dict,
    convert_to_multistream_comm,
)
from tqdm import tqdm

from dynapipe.model import (
    DynaPipeMicrobatch,
    DynaPipeMinibatch,
    get_uniform_cluster,
)
from dynapipe.schedule_opt.execution_planner import optimize_schedule


def get_hetero_minibatch(
    microbatch_multiplier, comm_factor=1
) -> DynaPipeMinibatch:
    fw_times = [4000] * 16
    memory_multiplier = microbatch_multiplier
    microbatches = []
    for i in range(len(microbatch_multiplier)):
        current_fw_times = [
            fw_times[j] * microbatch_multiplier[i]
            for j in range(len(fw_times))
        ]
        current_bw_times = [2 * t for t in current_fw_times]
        microbatch = DynaPipeMicrobatch(str(i))
        microbatch.set_fw_exec_times(current_fw_times)
        microbatch.set_bw_exec_times(current_bw_times)
        microbatch.set_fw_comm_size(
            [200 * comm_factor * microbatch_multiplier[i]]
            * (len(fw_times) - 1)
        )
        microbatch.set_bw_comm_size(
            [200 * comm_factor * microbatch_multiplier[i]]
            * (len(fw_times) - 1)
        )
        microbatch.set_model_state_memory([4000] * len(fw_times))
        microbatch.set_model_stored_activation_memory(
            [8000 * memory_multiplier[i]] * len(fw_times)
        )
        microbatch.set_model_peak_activation_memory(
            [16000 * memory_multiplier[i]] * len(fw_times)
        )
        microbatch.set_activation_shapes(
            [[(64, 128, 512)]] * (len(fw_times) // 2)
            + [[(64, 128, 512), (64, 128, 512)]] * (len(fw_times) // 2)
        )
        microbatches.append(microbatch)
    minibatch = DynaPipeMinibatch("test", microbatches)
    return minibatch


def gen_micro_batch_multipliers(n_iters, n_microbatches, std):
    rng = np.random.default_rng(seed=48)
    for _ in range(n_iters):
        m = np.clip(rng.normal(1, std, size=n_microbatches), 0.1, 10)
        normalized_m = m / (sum(m) / n_microbatches)
        yield normalized_m


def schedule_minibatch(
    n_stages,
    sch_type,
    n_iters,
    n_microbatches=16,
    std=0.1,
    multistream=False,
    comm_factor=1,
):
    nlayers = 16
    assert nlayers % n_stages == 0
    layers_per_stage = nlayers // n_stages
    device_assignment = []
    for i in range(n_stages):
        device_assignment += [i] * layers_per_stage
    cluster = get_uniform_cluster(n_stages, intra_node_bw=1e6)

    if sch_type == "1F1B":
        try_permutations = False
    else:
        try_permutations = True
    makespans = []
    for multiplier in gen_micro_batch_multipliers(
        n_iters, n_microbatches, std
    ):
        # multiplier = [1] * 16
        if multistream:
            try_permutations = False
        minibatch = get_hetero_minibatch(multiplier, comm_factor=comm_factor)
        (
            _,
            _,
            _,
            min_makespan,
            min_stats,
            min_instructions,
        ) = optimize_schedule(
            sch_type,
            minibatch,
            cluster,
            device_assignment,
            try_permutations=try_permutations,
            include_memory_stats=True,
            progress_bar=False,
            memory_limit=float("inf"),
        )
        if not multistream:
            makespans.append(min_makespan)
            continue
        # produce a reference trace
        ref_multiplier = [1] * 16
        ref_minibatch = get_hetero_minibatch(ref_multiplier)
        (
            _,
            _,
            _,
            ref_min_makespan,
            ref_min_stats,
            ref_min_instructions,
        ) = optimize_schedule(
            sch_type,
            ref_minibatch,
            cluster,
            device_assignment,
            try_permutations=False,
            include_memory_stats=True,
            progress_bar=False,
            memory_limit=float("inf"),
        )
        sch_trace = min_stats[-1]
        ref_trace = ref_min_stats[-1]
        sch_time_dict = construct_exec_time_dict(sch_trace)
        multistream_ref_trace, ref_makespan = convert_to_multistream_comm(
            ref_trace, sch_time_dict
        )
        multistream_sch_trace, makespan = convert_to_multistream_comm(
            sch_trace
        )
        makespans.append((makespan, ref_makespan))
        # trace_path = "test_wfcyclic_trace.json"
        # with open(trace_path, "w") as f:
        #     import json
        #     json.dump(min_stats[-1], f)
    return np.mean(np.array(makespans), axis=0)


if __name__ == "__main__":
    exists = os.path.isfile("./compare_schedule_multistream.csv")
    with open("./compare_schedule_multistream.csv", "a") as f:
        if not exists:
            f.write(
                "n_stages,sch_type,std,comm_factor,makespan,ref_makespan\n"
            )
        # for n_stages in tqdm([2, 4, 8, 16]):
        for n_stages in tqdm([16]):
            # for sch_type in tqdm(["1F1B", "wait-free-cyclic"], leave=False):
            for sch_type in tqdm(["wait-free-cyclic"], leave=False):
                for std in tqdm(
                    [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4], leave=False
                ):
                    for comm_factor in [0.2, 0.4, 0.8, 1.0, 1.2, 1.6, 3.2]:
                        makespan, ref_makespan = schedule_minibatch(
                            n_stages,
                            sch_type,
                            1,
                            std=std,
                            multistream=True,
                            comm_factor=comm_factor,
                        )
                        print(
                            f"{n_stages},{sch_type},{std},"
                            f"{comm_factor},{makespan},{ref_makespan}"
                        )
                        f.write(
                            f"{n_stages},{sch_type},{std},"
                            f"{comm_factor},{makespan},{ref_makespan}\n"
                        )
                        f.flush()
