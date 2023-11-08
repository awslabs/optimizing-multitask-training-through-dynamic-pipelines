# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle

import pytest

from dynapipe.model import (
    DynaPipeMicrobatch,
    DynaPipeMinibatch,
    get_uniform_cluster,
    get_uniform_microbatch,
)
from dynapipe.schedule_opt.execution_planner import (
    ExecutionPlan,
    optimize_schedule,
)


def hetero_minibatch() -> DynaPipeMinibatch:
    fw_times = [4000] * 4 + [2000] * 4  # 8 layer ende
    # fw_times = [4000] * 4 + [4000] * 4
    microbatch_multiplier = [1, 0.8, 1.2, 0.9, 1.1, 0.7, 1.4, 0.6]
    memory_multiplier = [1, 0.8, 1.2, 0.9, 1.1, 0.7, 1.4, 0.6]
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
            [2 * microbatch_multiplier[i]] * (len(fw_times) - 1)
        )
        microbatch.set_bw_comm_size(
            [2 * microbatch_multiplier[i]] * (len(fw_times) - 1)
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


@pytest.mark.parametrize("use_het_batch", [False, True])
@pytest.mark.parametrize(
    "device_assignment, sch_type, expected_file",
    [
        ([0, 0, 1, 1, 2, 2, 3, 3], "1F1B", "1f1b"),
        ([0, 0, 1, 1, 2, 2, 3, 3], "wait-free-cyclic", "wfcyclic"),
        ([0, 1, 2, 3, 0, 1, 2, 3], "interleaved-1F1B", "interleaved_1f1b"),
        ([0, 1, 2, 3, 0, 1, 2, 3], "wait-free-cyclic", "interleaved_wfcyclic"),
        ([0, 1, 2, 3, 3, 2, 1, 0], "wait-free-cyclic", "zigzag_wfcyclic"),
    ],
)
def test_minibatch(
    use_het_batch,
    device_assignment,
    sch_type,
    expected_file,
    try_permutations=False,
    memory_limit=float("inf"),
):
    cluster = get_uniform_cluster(4)
    if not use_het_batch:
        microbatches = []
        for i in range(8):
            microbatch = get_uniform_microbatch(8, comm_ratio=0.1)
            microbatch.name = str(i)
            microbatches.append(microbatch)
        minibatch = DynaPipeMinibatch("test", microbatches)
    else:
        minibatch = hetero_minibatch()
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
        memory_limit=memory_limit,
    )
    n_stages = (
        max([instr.stage for instrs in min_instructions for instr in instrs])
        + 1
    )
    eps = [
        ExecutionPlan(instrs, 8, 4, n_stages, i, [0, 1])
        for i, instrs in enumerate(min_instructions)
    ]
    serialized_eps = [ep.serialize() for ep in eps]
    prefix = "uniform" if not use_het_batch else "heter"
    expected_result_path = "{}_{}.pkl".format(prefix, expected_file)
    # Uncomment to get generated trace
    # trace_path = "{}_{}_trace.json".format(prefix, expected_file)
    # with open(trace_path, "w") as f:
    #     import json
    #     json.dump(min_stats[-1], f)
    # Uncomment to generate new expected results
    # with open(expected_result_path, "wb") as f:
    #     pickle.dump((min_makespan, min_stats[1], serialized_eps), f)
    with open(expected_result_path, "rb") as f:
        expected_result = pickle.load(f)
    assert expected_result == (min_makespan, min_stats[1], serialized_eps)


if __name__ == "__main__":
    pytest.main([__file__])
