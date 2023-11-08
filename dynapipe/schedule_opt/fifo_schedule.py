# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Optional, Tuple

from dynapipe.schedule_opt.wait_free_schedule import WaitFreeScheduler

from .schedule_common import ScheduleExecutor, ScheduleOperation
from .wait_free_schedule import WaitFreeExecutor


class FIFOExecutor(WaitFreeExecutor):
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
        self.is_executing = False
        self.last_executed_fw = True

    def reset(self):
        super().reset()
        self.available_queue = []

    def add_operation(self, op: ScheduleOperation):
        if op.is_forward:
            assert (
                op.flattened_stage in self.fw_stages
            ), "Operation {} not in executor".format(op)
        else:
            assert (
                op.flattened_stage in self.bw_stages
            ), "Operation {} not in executor".format(op)
        self.available_queue.append(op)

    def try_execute(self, current_time, ofob=False):
        events = []
        if self.available_queue and not self.is_executing:
            if ofob:
                op = None
                for idx, avail_op in enumerate(self.available_queue):
                    if avail_op.is_forward != self.last_executed_fw:
                        op = self.available_queue.pop(idx)
                        break
                if op is None:
                    return current_time, None, events
            else:
                op = self.available_queue.pop(0)
            event = self.get_exec_event(
                op,
                current_time,
                op.exec_time,
            )
            events.append(event)
            if not self.is_comm_stage and self.include_memory_stats:
                # we add two memory events for each op
                # one for the peak memory usage during the op
                # one for the stored memory usage after the op
                peak_time = current_time + op.exec_time / 2
                finish_time = current_time + op.exec_time
                memory_events = self.update_memory(
                    peak_time, op.peak_memory, finish_time, op.stored_memory
                )
                events += memory_events
            finish_time = current_time + op.exec_time
            self.is_executing = True
            self.last_executed_fw = op.is_forward
            return finish_time, op, events
        else:
            return current_time, None, events

    def finish_execute(self):
        self.is_executing = False


class FIFOScheduler(WaitFreeScheduler):
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
        return FIFOExecutor(
            executor_id,
            thread_id=thread_id,
            n_orig_layers=n_orig_layers,
            assigned_stages=assigned_stages,
            is_comm_stage=is_comm_stage,
            include_memory_stats=include_memory_stats,
            parent_executor=parent_executor,
            logger=self.logger,
        )
