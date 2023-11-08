# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pickle

import pytest

from dynapipe.pipe.instruction_optimizer import (
    InstructionOptimizer,
    _is_forward,
)
from dynapipe.pipe.instructions import *  # noqa: F403

eps_prefix = "./test_scheduler/"
eps_paths = [
    os.path.join(eps_prefix, x)
    for x in os.listdir(eps_prefix)
    if x.endswith(".pkl")
]


def load_eps(path):
    with open(path, "rb") as f:
        serialized_eps = pickle.load(f)[-1]
    eps = [ExecutionPlan.deserialize(ep) for ep in serialized_eps]
    return eps


@pytest.mark.parametrize("eps_path", eps_paths)
def test_inject_comm_finish_instrs(eps_path):
    input_eps = load_eps(eps_path)

    for ep in input_eps:
        input_instrs = ep.instructions
        optimizer = InstructionOptimizer([], n_stages=ep.nstages)
        output_instrs = optimizer._inject_comm_finish_instrs(input_instrs)
        start_keys = set()
        for instruction in output_instrs:
            if isinstance(instruction, CommunicationStartInstruction):
                key = (
                    instruction.microbatch,
                    instruction.stage,
                    _is_forward(instruction),
                )
                start_keys.add(key)
            elif isinstance(instruction, CommunicationFinishInsturction):
                key = (
                    instruction.microbatch,
                    instruction.stage,
                    _is_forward(instruction),
                )
                assert (
                    key in start_keys
                ), "Finish instruction without start instruction: {}".format(
                    instruction
                )
                start_keys.remove(key)
        assert (
            len(start_keys) == 0
        ), "Start instruction without finish instruction: {}".format(
            start_keys
        )


@pytest.mark.parametrize("eps_path", eps_paths)
def test_allocate_buffer(eps_path):
    input_eps = load_eps(eps_path)

    for ep in input_eps:
        input_instrs = ep.instructions
        optimizer = InstructionOptimizer([], n_stages=ep.nstages)
        output_instrs, n_buffer_slots = optimizer._allocate_buffers(
            input_instrs
        )
        buffer_slots = [None] * n_buffer_slots
        for instr in output_instrs:
            if isinstance(
                instr, (RecvActivationStart, RecvGradStart, LoadInput)
            ):
                # instructions that populates buffer slots
                for buffer_id, buffer_shape in zip(
                    instr.buffer_ids, instr.buffer_shapes
                ):
                    assert buffer_slots[buffer_id] is None
                    buffer_slots[buffer_id] = buffer_shape
            elif isinstance(
                instr,
                (
                    ForwardPass,
                    BackwardPass,
                    SendActivationStart,
                    SendGradStart,
                ),
            ):
                # instruction that reads buffer slots
                for buffer_id, buffer_shape in zip(
                    instr.buffer_ids, instr.buffer_shapes
                ):
                    assert buffer_slots[buffer_id] == buffer_shape
            elif isinstance(instr, FreeBuffer):
                # instruction that frees buffer slots
                for buffer_id in instr.buffer_ids:
                    assert buffer_slots[buffer_id] is not None
                    buffer_slots[buffer_id] = None
        assert all([buffer_slot is None for buffer_slot in buffer_slots])


if __name__ == "__main__":
    pytest.main([__file__])
