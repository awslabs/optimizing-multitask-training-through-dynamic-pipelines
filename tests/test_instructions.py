# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Union

import pytest

from dynapipe.pipe.instructions import *  # noqa: F403

comm_start_instrs = [
    SendActivationStart,
    SendGradStart,
    RecvActivationStart,
    RecvGradStart,
]
comm_finish_instrs = [
    SendActivationFinish,
    SendGradFinish,
    RecvActivationFinish,
    RecvGradFinish,
]
buffer_instrs = [LoadInput, FreeBuffer]


def _test_serialization(instr: Union[PipeInstruction, ExecutionPlan]):
    serialized = instr.serialize()
    assert isinstance(serialized, bytes)
    if isinstance(instr, PipeInstruction):
        deserialized, remaining_bytes = PipeInstruction.deserialize(serialized)
        assert len(remaining_bytes) == 0
    else:
        deserialized = ExecutionPlan.deserialize(serialized)
    assert instr == deserialized
    # test casting to str
    serialized_casted = (
        serialized.decode("iso-8859-1").encode().decode().encode("iso-8859-1")
    )
    if isinstance(instr, PipeInstruction):
        deserialized_casted, remaining_bytes = PipeInstruction.deserialize(
            serialized_casted
        )
        assert len(remaining_bytes) == 0
    else:
        deserialized_casted = ExecutionPlan.deserialize(serialized_casted)
    assert instr == deserialized_casted


@pytest.mark.parametrize("instr_cls", comm_start_instrs)
@pytest.mark.parametrize("n_tensors", [1, 2, 3])
@pytest.mark.parametrize("comm_dims", [0, 1, 2, 3])
def test_serialization_comm_start_same_dim(
    instr_cls: Type[CommunicationStartInstruction],
    n_tensors: int,
    comm_dims: int,
):
    buffer_ids = list(range(n_tensors))
    buffer_shapes = [
        tuple([2 for _ in range(comm_dims)]) for _ in range(n_tensors)
    ]
    instr = instr_cls(
        0, 1, peer=0, buffer_shapes=buffer_shapes, buffer_ids=buffer_ids
    )
    _test_serialization(instr)


@pytest.mark.parametrize("instr_cls", comm_start_instrs)
def test_serialization_comm_start_diff_dim(
    instr_cls: CommunicationStartInstruction,
):
    buffer_ids = [0, 1]
    buffer_shapes = [(2, 2, 2), (2, 2)]
    instr = instr_cls(
        0, 1, peer=0, buffer_shapes=buffer_shapes, buffer_ids=buffer_ids
    )
    _test_serialization(instr)


@pytest.mark.parametrize("instr_cls", comm_finish_instrs)
@pytest.mark.parametrize("n_buffers", [1, 2, 3])
def test_serialization_comm_finish(
    instr_cls: CommunicationFinishInsturction, n_buffers: int
):
    instr = instr_cls(0, 1, peer=0, buffer_ids=list(range(n_buffers)))
    _test_serialization(instr)


def test_serialization_forward():
    instr = ForwardPass(0, 1, [0, 1])
    _test_serialization(instr)


@pytest.mark.parametrize("first_bw_layer", [True, False])
def test_serialization_backward(first_bw_layer: bool):
    instr = BackwardPass(0, 1, [0, 1], first_bw_layer=first_bw_layer)
    _test_serialization(instr)


@pytest.mark.parametrize("instr_cls", buffer_instrs)
@pytest.mark.parametrize("n_buffers", [1, 2, 3])
@pytest.mark.parametrize("buffer_dims", [0, 1, 2, 3])
def test_serialization_buffer(
    instr_cls: Union[Type[LoadInput], Type[FreeBuffer]],
    n_buffers: int,
    buffer_dims: int,
):
    buffer_ids = list(range(n_buffers))
    buffer_shapes = [
        tuple([2 for _ in range(buffer_dims)]) for _ in range(n_buffers)
    ]
    if instr_cls == LoadInput:
        instr = instr_cls(
            0,
            1,
            buffer_shapes=buffer_shapes,
            buffer_ids=buffer_ids,
        )
    else:
        instr = instr_cls(buffer_ids=buffer_ids)
    _test_serialization(instr)


def test_serialization_exec_plan():
    instructions = [
        LoadInput(0, 0, buffer_shapes=[(2, 2, 2), (2, 2)], buffer_ids=[0, 1]),
        SendActivationStart(
            0, 1, peer=0, buffer_shapes=[(2, 2, 2), (2, 2)], buffer_ids=[0, 1]
        ),
        SendActivationFinish(0, 1, peer=0, buffer_ids=[0, 1]),
        RecvActivationStart(
            0, 1, peer=0, buffer_shapes=[(2, 2, 2), (2, 2)], buffer_ids=[0, 1]
        ),
        RecvActivationFinish(0, 1, peer=0, buffer_ids=[0, 1]),
        ForwardPass(0, 1, buffer_ids=[0, 1]),
        SendGradStart(
            0, 1, peer=0, buffer_shapes=[(2, 2, 2), (2, 2)], buffer_ids=[0, 1]
        ),
        SendGradFinish(0, 1, peer=0, buffer_ids=[0, 1]),
        RecvGradStart(
            0, 1, peer=0, buffer_shapes=[(2, 2, 2), (2, 2)], buffer_ids=[0, 1]
        ),
        RecvGradFinish(0, 1, peer=0, buffer_ids=[0, 1]),
        BackwardPass(0, 1, buffer_ids=[0, 1], first_bw_layer=True),
        BackwardPass(0, 1, buffer_ids=[0, 1], first_bw_layer=False),
        FreeBuffer(buffer_ids=[0, 1]),
    ]
    exec_plan = ExecutionPlan(
        instructions,
        micro_batches=8,
        nranks=4,
        nstages=4,
        rank=1,
        assigned_stages=[0, 1],
        recompute_method=RecomputeMethod.SELECTIVE,
        num_pipe_buffers=2,
    )
    _test_serialization(exec_plan)


if __name__ == "__main__":
    pytest.main([__file__])
