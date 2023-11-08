# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
from typing import Dict, List, Optional, Tuple

from .cyclic_schedule import CyclicScheduler
from .schedule_common import (
    ExecutorIndex,
    ScheduleExecutor,
    ScheduleOperation,
    SchedulerMinibatchSpec,
)
from .wait_free_schedule import WaitFreeExecutor, WaitFreeScheduler


class WaitFreeCyclicExecutor(WaitFreeExecutor):
    def __init__(
        self,
        executor_id: int,
        thread_id: int,
        n_orig_layers: int,
        assigned_stages: List[Tuple[int, float, bool]],
        is_comm_stage: bool = False,
        include_memory_stats: bool = True,
        parent_executor: Optional[ScheduleExecutor] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(
            executor_id,
            thread_id,
            n_orig_layers,
            assigned_stages,
            is_comm_stage,
            include_memory_stats,
            parent_executor,
            logger,
        )
        self.available_ops = set()
        self.next_op_idx = 0
        self.is_executing = False
        self.operator_order = None

    def set_operator_order(self, operator_order: List[ScheduleOperation]):
        self.operator_order = operator_order
        self.debug_print("Operator order: {}".format(operator_order))

    def reset(self):
        super().reset()
        self.available_ops.clear()
        self.next_op_idx = 0

    def add_operation(self, op: ScheduleOperation):
        if op.is_forward:
            assert (
                op.flattened_stage in self.fw_stages
            ), "Operation {} not in executor".format(op)
        else:
            assert (
                op.flattened_stage in self.bw_stages
            ), "Operation {} not in executor".format(op)
        self.available_ops.add(op)

    def try_execute(self, current_time):
        assert self.operator_order is not None, "Execution order not set"
        events = []
        if not self.is_executing and self.next_op_idx < len(
            self.operator_order
        ):
            self.debug_print(
                "Trying to execute next operation: {}".format(
                    self.operator_order[self.next_op_idx]
                )
            )
            next_op = self.operator_order[self.next_op_idx]
            if next_op in self.available_ops:
                self.next_op_idx += 1
                event = self.get_exec_event(
                    next_op,
                    current_time,
                    next_op.exec_time,
                )
                events.append(event)
                if not self.is_comm_stage and self.include_memory_stats:
                    # we add two memory events for each op
                    # one for the peak memory usage during the op
                    # one for the stored memory usage after the op
                    peak_time = current_time + next_op.exec_time / 2
                    finish_time = current_time + next_op.exec_time
                    memory_events = self.update_memory(
                        peak_time,
                        next_op.peak_memory,
                        finish_time,
                        next_op.stored_memory,
                    )
                    events += memory_events
                finish_time = current_time + next_op.exec_time
                self.is_executing = True
                return finish_time, next_op, events
        # Currently executing or no available operations
        return current_time, None, events

    def finish_execute(self):
        self.is_executing = False


class WaitFreeCyclicScheduler(WaitFreeScheduler):
    def __init__(
        self,
        minibatch_spec: SchedulerMinibatchSpec,
        include_memory_stats: bool = True,
        memory_limit: float = float("inf"),
        max_otf_microbatches: int = int(1e6),
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(
            minibatch_spec,
            include_memory_stats,
            memory_limit,
            logger=logger,
        )
        self.cyclic_scheduler = CyclicScheduler(
            minibatch_spec,
            self.include_memory_stats,
            memory_limit=memory_limit,
            max_otf_microbatches=max_otf_microbatches,
            logger=logger,
        )
        self.no_valid_schedule = False
        self.executors: Dict[ExecutorIndex, WaitFreeCyclicExecutor]

    def _get_executor(
        self,
        executor_id,
        thread_id,
        n_orig_layers,
        assigned_stages,
        is_comm_stage,
        include_memory_stats,
        memory_limit=float("inf"),
        parent_executor=None,
    ):
        # overrides Scheduler
        return WaitFreeCyclicExecutor(
            executor_id,
            thread_id=thread_id,
            n_orig_layers=n_orig_layers,
            assigned_stages=assigned_stages,
            is_comm_stage=is_comm_stage,
            include_memory_stats=include_memory_stats,
            parent_executor=parent_executor,
            logger=self.logger,
        )

    def _init_executors(self, n_microbatches, **kwargs):
        self.operator_order = self.cyclic_scheduler.get_operator_order()
        if self.operator_order is None:
            # no valid schedule found
            self.no_valid_schedule = True
            return False
        # overrides WaitFreeScheduler
        status = super()._init_executors(
            n_microbatches,
            **kwargs,
        )
        if not status:
            return False
        for executor_idx, executor in self.executors.items():
            ops = self.operator_order[executor_idx]

            def _create_new_op(op: ScheduleOperation):
                # we need to reset the op's executor. since op is immutable,
                #  we need to create a new op
                if op.next_executor is None:
                    next_executor = None
                else:
                    next_executor_id = ExecutorIndex(
                        op.next_executor.executor_id,
                        op.next_executor.thread_id,
                    )
                    next_executor = self.executors[next_executor_id]
                return dataclasses.replace(op, next_executor=next_executor)

            executor.set_operator_order([_create_new_op(op) for op in ops])
        return True

    def schedule(
        self, warmup=False, warmup_n_microbatches=-1, ofob=False, **kwargs
    ):
        # overrides WaitFreeScheduler
        n_microbatches = len(self.minibatch_spec.microbatches)
        if warmup:
            if warmup_n_microbatches == -1:
                warmup_n_microbatches = min(
                    self.n_flattened_stages - 1, n_microbatches
                )
                self.logger.warning(
                    "warmup_n_microbatches <= 0, "
                    "setting it to min(n_layers - 1, n_microbatches) "
                    f"({warmup_n_microbatches})"
                )
            else:
                assert (
                    warmup_n_microbatches <= n_microbatches
                ), "warmup_n_microbatches must be <= n_microbatches"
        return super().schedule(
            warmup=warmup,
            warmup_n_microbatches=warmup_n_microbatches,
            ofob=ofob,
            **kwargs,
        )
