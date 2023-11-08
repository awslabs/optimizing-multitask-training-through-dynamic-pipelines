# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Type

from dynapipe.utils.logger import create_logger

from .instructions import *  # noqa: F403


def _handle_free_buffer(exec: "PipelineExecutor", instr: FreeBuffer):
    # free buffer just removes the buffer from the buffer slots
    buffer_ids = instr.buffer_ids
    for buffer_id in buffer_ids:
        exec.buffer_slots[buffer_id] = None


class PipelineExecutor:
    """
    Executes the dynamic pipeline according to pipeline instructions.
    """

    def __init__(
        self,
        dp_rank: int = None,
        pp_rank: int = None,
        synchronous: bool = False,
    ):
        # rank is optional, only used for debugging
        self.buffer_slots = []
        self._instruction_handlers = {}
        self.dp_rank = dp_rank
        self.pp_rank = pp_rank
        # if synchronous, cuda device synchronization is performed after
        # each instruction, mainly used for debugging
        self.synchronous = synchronous
        # register default handlers
        self.register_handler(FreeBuffer, _handle_free_buffer)
        self.logger = create_logger(
            "PipelineExecutor",
            prefix=f"dRank {dp_rank} pRank {pp_rank}",
            log_file=f"executor/dr{dp_rank}_pr{pp_rank}.log",
        )
        self.instr_index = 0
        self.current_iteration = None
        self.is_last_micro_batch = False

    def execute(self, execution_plan: ExecutionPlan, iteration=None):
        self.execution_plan = execution_plan
        self.buffer_slots = [None] * execution_plan.num_pipe_buffers
        if iteration is not None:
            self.current_iteration = iteration
            self.logger.debug("Executing iteration %d", iteration)
        for instr_index, instruction in enumerate(execution_plan.instructions):
            self.logger.debug("Executing instruction: %s", instruction)
            self.instr_index = instr_index
            if (
                instruction.microbatch is not None
                and instruction.microbatch
                == execution_plan.num_micro_batches - 1
            ):
                self.is_last_micro_batch = True
            else:
                self.is_last_micro_batch = False
            self._execute_instruction(instruction)
        if iteration is not None:
            self.logger.debug("Finished executing iteration %d", iteration)

    def register_handler(
        self, instruction_type: Type[PipeInstruction], handler
    ):
        if not issubclass(instruction_type, PipeInstruction):
            raise TypeError(
                f"Instruction type must be a subclass of PipeInstruction, "
                f"got {instruction_type.__name__}"
            )
        if instruction_type in self._instruction_handlers:
            raise ValueError(
                f"Instruction handler for {instruction_type.__name__} "
                "already registered."
            )
        self._instruction_handlers[instruction_type] = handler

    @classmethod
    def _get_leaf_subclasses(cls, instruction_type: Type[PipeInstruction]):
        subclasses = instruction_type.__subclasses__()
        if subclasses:
            for subclass in subclasses:
                yield from cls._get_leaf_subclasses(subclass)
        else:
            yield instruction_type

    @classmethod
    def get_all_needed_handlers(cls):
        needed_handlers = set(
            cls._get_leaf_subclasses(PipeInstruction)
        ).difference([FreeBuffer])
        return [x.__name__ for x in needed_handlers]

    def check_all_handlers_registered(self):
        for instruction_type in self._get_leaf_subclasses(PipeInstruction):
            if instruction_type not in self._instruction_handlers:
                raise ValueError(
                    "No handler registered for instruction "
                    f"{instruction_type.__name__}"
                )

    def register_synchronization_handler(self, handler):
        self.synchronize = handler

    def synchronize(self):
        raise NotImplementedError("Synchronization handler not registered")

    def _execute_instruction(self, instruction: PipeInstruction):
        handler = self._instruction_handlers.get(type(instruction))
        if handler is None:
            raise ValueError(
                "No handler registered for instruction "
                f"{type(instruction).__name__}"
            )
        handler(self, instruction)
        if self.synchronous:
            self.synchronize()
