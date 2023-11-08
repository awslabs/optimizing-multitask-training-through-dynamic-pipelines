# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Dict, List, Optional

from .schedule_common import (
    ExecutorIndex,
    ScheduleExecutor,
    ScheduleOperation,
    Scheduler,
    SchedulerMinibatchSpec,
)


@dataclass(order=True)
class CompleteEvent:
    completion_time: float
    op: ScheduleOperation = field(compare=False)
    executor: ScheduleExecutor = field(compare=False)


class WaitFreeExecutor(ScheduleExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.available_queue: List[ScheduleOperation] = []

    def add_operation(self, op: ScheduleOperation):
        raise NotImplementedError

    def try_execute(self, current_time):
        raise NotImplementedError

    def finish_execute(self):
        raise NotImplementedError


class WaitFreeScheduler(Scheduler):
    def __init__(
        self,
        minibatch_spec: SchedulerMinibatchSpec,
        include_memory_stats: bool = True,
        memory_limit: float = float("inf"),
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(
            minibatch_spec,
            include_memory_stats,
            memory_limit,
            logger=logger,
        )
        self._initialize()
        self._pending_events: PriorityQueue[CompleteEvent] = PriorityQueue()
        self.executors: Dict[ExecutorIndex, WaitFreeExecutor]

    def _get_executor(
        self,
        executor_id,
        thread_id,
        n_orig_layers,
        assigned_stages,
        is_comm_stage,
        include_memory_stats,
        memory_limit=float("inf"),
    ):
        # overrides Scheduler
        raise NotImplementedError

    def _init_executors(self, n_microbatches, **kwargs):
        for executor in self.executors.values():
            executor.reset()
        self.communication_executors = [
            executor
            for executor in self.executors.values()
            if executor.is_comm_stage
        ]
        self.computation_executors = [
            executor
            for executor in self.executors.values()
            if not executor.is_comm_stage
        ]
        return True

    def _inject_microbatches(
        self, microbatch_offset: int, n_microbatches: int
    ):
        for microbatch_id in range(
            microbatch_offset, microbatch_offset + n_microbatches
        ):
            executor = self.executors[
                self.minibatch_spec.flattened_executor_assignment[0]
            ]
            op = self._get_op(0, microbatch_id)
            executor.add_operation(op)

    def _on_op_finish(self, executor: WaitFreeExecutor, op: ScheduleOperation):
        executor.finish_execute()
        if op.flattened_stage < self.n_flattened_stages - 1:
            next_layer = op.flattened_stage + 1
            next_executor = self.minibatch_spec.flattened_executor_assignment[
                next_layer
            ]
            self.executors[next_executor].add_operation(
                self._get_op(next_layer, op.microbatch)
            )

    def _push_end_event(self, op, executor, end_time):
        self._pending_events.put(CompleteEvent(end_time, op, executor))

    def schedule(self, **kwargs):
        n_microbatches = len(self.minibatch_spec.microbatches)
        status = self._init_executors(n_microbatches, **kwargs)
        if not status:
            return None
        self._inject_microbatches(0, n_microbatches)
        trace_events = self._get_trace_events()
        current_time = 0

        def __try_execute():
            # priortize communication executors
            for executor in (
                self.communication_executors + self.computation_executors
            ):
                end_time, launched_op, events = executor.try_execute(
                    current_time
                )
                if launched_op:
                    self._push_end_event(launched_op, executor, end_time)
                    trace_events["traceEvents"].extend(events)

        while True:
            __try_execute()
            if self._pending_events.empty():
                break
            else:
                next_event = self._pending_events.get()
                current_time = next_event.completion_time
                ready_events = [next_event]
                while not self._pending_events.empty():
                    # try to process all events that finish at the same time
                    another_event = self._pending_events.get()
                    if another_event.completion_time <= current_time + 1e-6:
                        ready_events.append(another_event)
                    else:
                        self._pending_events.put(another_event)
                        break
                for event in ready_events:
                    self._on_op_finish(event.executor, event.op)
        self.makespan = current_time
        # make sure all executors are empty
        for executor_idx, executor in self.executors.items():
            if hasattr(executor, "available_queue"):
                assert len(executor.available_queue) == 0, (
                    f"Executor {executor_idx} has non-empty ready queue "
                    f"at end of scheduling: {executor.available_queue}"
                )
            if hasattr(executor, "next_op_idx"):
                assert executor.next_op_idx == len(executor.operator_order), (
                    f"Executor {executor_idx} has not finished all operations "
                    f"at end of scheduling: {executor.available_queue}"
                )
        return trace_events
