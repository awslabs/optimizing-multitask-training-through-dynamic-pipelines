# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from dataclasses import dataclass

from .instructions import *  # noqa: F403


@dataclass
class _Buffer:
    slot: int
    microbatch: int
    stage: int
    shape: Tuple[int, ...]
    life_start: int
    life_end: int


def _is_forward(instr):
    return isinstance(
        instr,
        (
            ForwardPass,
            SendActivationStart,
            SendActivationFinish,
            RecvActivationStart,
            RecvActivationFinish,
            LoadInput,
        ),
    )


def _is_recv_instr(instr):
    return isinstance(instr, (RecvActivationStart, RecvGradStart))


def _is_send_instr(instr):
    return isinstance(instr, (SendActivationStart, SendGradStart))


def _is_compute_instr(instr):
    return isinstance(instr, (ForwardPass, BackwardPass))


def _get_key(instr: PipeInstruction):
    return (instr.microbatch, instr.stage, _is_forward(instr))


def _fw_stage_to_bw_stage(stage: int, n_stages: int):
    return n_stages - 1 - stage


class InstructionOptimizer:
    """
    Inject buffer allocation/free and communication finish
    ops into the pipeline instructions.
    """

    def __init__(
        self,
        per_worker_instructions: List[List[PipeInstruction]],
        n_stages: int,
    ):
        self.per_worker_instructions = per_worker_instructions
        self.n_stages = n_stages

    def _inject_comm_finish_instrs(self, instrs: List[PipeInstruction]):
        # We assume that each rank has two communication streams,
        # one for communication with the previous rank and one for
        # the next rank. This gives better communication overlap
        # without the possibility to deadlock.
        #
        # For each RecvXXXStart, we need a RecvXXXFinish instr before
        # the instruction that uses the data, which is identified by
        # the corresponding microbatch and stage id.
        #
        # For each SendXXXStart, there is a trade-off between freeing the
        # memory early and unnecessary waiting if using static location for
        # SendXXXFinish. Therefore we dynamically query if the send is complete
        # during execution, and SendXXXFinish is added as late as possible,
        # only serving as constraints for correctness (in case dynamic query
        # fails).
        # We add SendActivationFinish only before the corresponding backward
        # pass, at which point the send must have completed. All SendGradFinish
        # are added at the end of the iteration.

        instr_map: Dict[
            Type[CommunicationStartInstruction],
            Type[CommunicationFinishInsturction],
        ] = {
            SendActivationStart: SendActivationFinish,
            RecvActivationStart: RecvActivationFinish,
            SendGradStart: SendGradFinish,
            RecvGradStart: RecvGradFinish,
        }
        _prepend_map = {}
        accumulated_send_activation_finish_instrs = defaultdict(list)
        accumulated_send_grad_finish_instrs = []
        new_instrs = []
        for instr in instrs:
            if _is_recv_instr(instr):
                key = _get_key(instr)
                assert key not in _prepend_map
                _prepend_map[key] = instr
            elif _is_send_instr(instr):
                instr: CommunicationStartInstruction
                # get the corresponding finish instr
                finish_instr = instr_map[type(instr)](
                    instr.microbatch, instr.stage, instr.peer
                )
                # append existing send finish instrs
                # new_instrs.extend(accumulated_send_finish_instrs[instr.peer].copy())
                # accumulated_send_finish_instrs[instr.peer].clear()
                if isinstance(instr, SendActivationStart):
                    accumulated_send_activation_finish_instrs[
                        (
                            instr.microbatch,
                            _fw_stage_to_bw_stage(instr.stage, self.n_stages),
                        )
                    ].append(finish_instr)
                elif isinstance(instr, SendGradStart):
                    accumulated_send_grad_finish_instrs.append(finish_instr)
                else:
                    raise RuntimeError(f"Unknown send instr: {instr}")
            elif _is_compute_instr(instr):
                key = _get_key(instr)
                if key in _prepend_map:
                    start_instr: CommunicationStartInstruction = _prepend_map[
                        key
                    ]
                    new_instrs.append(
                        instr_map[type(start_instr)](
                            start_instr.microbatch,
                            start_instr.stage,
                            start_instr.peer,
                        )
                    )
                if not _is_forward(instr):
                    # append existing send activation finish instrs
                    new_instrs.extend(
                        accumulated_send_activation_finish_instrs[
                            (instr.microbatch, instr.stage)
                        ].copy()
                    )
                    accumulated_send_activation_finish_instrs[
                        (instr.microbatch, instr.stage)
                    ].clear()
            new_instrs.append(instr)
        # append any remaining send finish instrs
        for (
            accumulated_send_finish_instrs
        ) in accumulated_send_activation_finish_instrs.values():
            assert len(accumulated_send_finish_instrs) == 0
        new_instrs.extend(accumulated_send_grad_finish_instrs)
        return new_instrs

    def _allocate_buffers(self, instrs: List[PipeInstruction]):
        # allcate: create new tensors (e.g. torch.zeros)
        # assign: assign a tensor to a buffer slot
        # Current assumptions:
        # 1. RecvXXXStart allocates its own buffers and writes to buffer_ids,
        #    so we are only assigning buffer slots here. This can be optimized
        #    by allocating buffers in advance if memory allocation issues
        #    arise.
        # 2. ForwardPass and BackwardPass reads and writes the same buffer_ids.
        #    SendXXXStart only reads but do not write to buffer_ids.
        #    RecvXXXStart creates new buffers. SendXXXFinish and RecvXXXFinish
        #    do not read or write to buffer_ids.
        buffer_slots: List[_Buffer] = []
        key_to_buffers: Dict[Any, List[_Buffer]] = defaultdict(list)

        def _allocate_buffer_slot(
            instr: BufferInstruction, shape, current_idx
        ) -> _Buffer:
            # find the first available buffer slot
            slot = len(buffer_slots)
            buffer = _Buffer(
                slot, instr.microbatch, instr.stage, shape, current_idx, None
            )
            buffer_slots.append(buffer)
            return buffer

        for instr_idx, instr in enumerate(instrs):
            if isinstance(
                instr,
                (
                    ForwardPass,
                    BackwardPass,
                    SendActivationStart,
                    SendGradStart,
                ),
            ):
                key = _get_key(instr)
                if isinstance(instr, BackwardPass) and instr.first_bw_layer:
                    # first backward layer directly uses forward pass buffers
                    assert key not in key_to_buffers
                    fw_key = (instr.microbatch, instr.stage - 1, True)
                    key_to_buffers[key] = key_to_buffers[fw_key].copy()
                assert (
                    key in key_to_buffers
                ), f"buffer not allocated for {instr}"
                buffers = key_to_buffers[key]
                # we only allow dropping buffers
                # allocation needs explicit instrs
                assert len(buffers) >= len(instr.buffer_shapes), (
                    f"buffer allocation mismatch for {instr}, "
                    f"expected less than {len(instr.buffer_shapes)}, "
                    f"got {len(buffers)}"
                )
                for buffer in buffers:
                    instr.buffer_ids.append(buffer.slot)
                    buffer.life_end = instr_idx
            elif isinstance(
                instr, (RecvActivationStart, RecvGradStart, LoadInput)
            ):
                # allocate new buffers
                key = _get_key(instr)
                for shape in instr.buffer_shapes:
                    buffer = _allocate_buffer_slot(instr, shape, instr_idx)
                    instr.buffer_ids.append(buffer.slot)
                    key_to_buffers[key].append(buffer)

        # now insert buffer free instructions
        new_instrs = []
        buffers_freed_at_idx = defaultdict(list)
        for buffer in buffer_slots:
            assert buffer.life_end is not None, f"buffer {buffer} not used. "
            buffers_freed_at_idx[buffer.life_end].append(buffer.slot)
        for instr_idx, instr in enumerate(instrs):
            new_instrs.append(instr)
            if instr_idx in buffers_freed_at_idx:
                new_instrs.append(
                    FreeBuffer(buffer_ids=buffers_freed_at_idx[instr_idx])
                )
        return new_instrs, len(buffer_slots)

    def optimize(self):
        result_instrs = []
        result_num_buffers = []
        for instrs in self.per_worker_instructions:
            instrs = self._inject_comm_finish_instrs(instrs)
            instrs, num_buffers = self._allocate_buffers(instrs)
            # check all needed buffers are allocated
            for instr in instrs:
                if isinstance(
                    instr,
                    (
                        ForwardPass,
                        BackwardPass,
                        SendActivationStart,
                        SendGradStart,
                        RecvActivationStart,
                        RecvGradStart,
                        LoadInput,
                    ),
                ):
                    assert len(instr.buffer_ids) >= len(
                        instr.buffer_shapes
                    ), f"buffer allocation mismatch for {instr}, "
                    f"expected {len(instr.buffer_shapes)}, "
                    f"got {len(instr.buffer_ids)}"
            result_instrs.append(instrs)
            result_num_buffers.append(num_buffers)
        return result_instrs, result_num_buffers
