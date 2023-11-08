# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .schedule_common import (
    ExecutorIndex,
    ScheduleExecutor,
    ScheduleOperation,
    Scheduler,
    SchedulerMinibatchSpec,
)


class CyclicScheduler(Scheduler):
    def __init__(
        self,
        minibatch_spec: SchedulerMinibatchSpec,
        include_memory_stats: bool = True,
        memory_limit: float = float("inf"),
        max_otf_microbatches: int = int(1e6),
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(
            minibatch_spec,
            include_memory_stats,
            memory_limit,
            logger=logger,
        )
        self.max_otf_microbatches = max_otf_microbatches
        self._initialize()
        # calculate cycle time
        executor_exec_times = []
        max_stage_exec_times_acorss_mb = []
        for layer in range(self.n_flattened_stages):
            max_time = 0
            for microbatch in self.minibatch_spec.microbatches:
                max_time = max(
                    max_time, microbatch.flattened_exec_times[layer]
                )
            max_stage_exec_times_acorss_mb.append(max_time)
        for executor in self.executors.items():
            executor_exec_times.append(
                sum(
                    (max_stage_exec_times_acorss_mb[i])
                    for i in self.executor2stages[executor]
                )
            )
        self.cycle_time = max(executor_exec_times)
        self.fast_forward_comm = True
        self.executors: Dict[int, CyclicExecutor]

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
        return CyclicExecutor(
            executor_id,
            thread_id=thread_id,
            n_orig_layers=n_orig_layers,
            assigned_stages=assigned_stages,
            is_comm_stage=is_comm_stage,
            include_memory_stats=include_memory_stats,
            memory_limit=memory_limit,
            max_otf_microbatches=self.max_otf_microbatches,
            parent_executor=parent_executor,
            logger=self.logger,
        )

    def _init_executors(self, n_microbatches, **kwargs):
        for executor in self.executors.values():
            executor.reset()
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
            executor.forward_cycle()

    def _on_executed_ops(self, executed_ops: List[ScheduleOperation]):
        for op in executed_ops:
            if op.flattened_stage < self.n_flattened_stages - 1:
                next_stage = op.flattened_stage + 1
                next_executor = (
                    self.minibatch_spec.flattened_executor_assignment[
                        next_stage
                    ]
                )
                next_op = self._get_op(next_stage, op.microbatch)
                self.executors[next_executor].add_operation(next_op)

    def _get_global_instance_event(self, name, current_time):
        return {
            "name": name,
            "ph": "i",
            "ts": current_time,
            "pid": 0,
            "tid": 0,
            "s": "g",
        }

    def _schedule(
        self,
    ):
        n_microbatches = len(self.minibatch_spec.microbatches)
        status = self._init_executors(n_microbatches)
        if not status:
            return None, None
        self._inject_microbatches(0, n_microbatches)
        trace_events = self._get_trace_events()
        operator_execution_order: Dict[
            ExecutorIndex, list[ScheduleOperation]
        ] = defaultdict(list)
        current_time = 0
        executor_end_times = []
        # for _ in range(n_microbatches + self.n_layers):
        while True:
            has_progress = False
            for executor in self.executors.values():
                end_time, executed_ops, events = executor.exec_cycle(
                    current_time
                )
                executor_end_times.append(end_time)
                if executed_ops:
                    self._on_executed_ops(executed_ops)
                    trace_events["traceEvents"].extend(events)
                    operator_execution_order[
                        ExecutorIndex(executor.executor_id, executor.thread_id)
                    ] += executed_ops
                    has_progress = True
            for executor in self.executors.values():
                executor.forward_cycle()
            if self.fast_forward_comm:
                # execute an extra cycle to fast forward communication
                for executor in self.executors.values():
                    if executor.is_comm_stage:
                        end_time, executed_ops, events = executor.exec_cycle(
                            current_time
                        )
                        executor_end_times.append(end_time)
                        if executed_ops:
                            self._on_executed_ops(executed_ops)
                            trace_events["traceEvents"].extend(events)
                            operator_execution_order[
                                ExecutorIndex(
                                    executor.executor_id, executor.thread_id
                                )
                            ] += executed_ops
                            has_progress = True
                for executor in self.executors.values():
                    executor.forward_cycle()
            if not has_progress:
                break
            current_time = max(executor_end_times)
            trace_events["traceEvents"].append(
                self._get_global_instance_event("Cycle ended", current_time)
            )
        self.makespan = max(executor_end_times)
        for executor in self.executors.values():
            for buffer in executor.buffers.values():
                if len(buffer) != 0:
                    # unable to schedule all operations
                    self.makespan = -1
                    return None, None
        return trace_events, operator_execution_order

    def get_operator_order(
        self,
    ):
        _, operator_execution_order = self._schedule()
        return operator_execution_order

    def schedule(self):
        trace_events, _ = self._schedule()
        return trace_events


class CyclicExecutor(ScheduleExecutor):
    def __init__(
        self,
        executor_id: int,
        thread_id: int,
        n_orig_layers: int,
        assigned_stages: List[Tuple[int, float, bool]],
        is_comm_stage: bool = False,
        include_memory_stats: bool = True,
        memory_limit: float = float("inf"),
        max_otf_microbatches: int = int(1e6),
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
        self.memory_limit = memory_limit
        self.max_otf_microbatches = max_otf_microbatches
        self.buffers: Dict[int, List[ScheduleOperation]] = {}
        self.next_step_buffers: Dict[int, List[ScheduleOperation]] = {}
        for flattened_stage_id, _, _ in assigned_stages:
            self.buffers[flattened_stage_id] = []
            self.next_step_buffers[flattened_stage_id] = []
        self.exec_order = []
        for i in range(max(len(self.fw_stages), len(self.bw_stages))):
            if i < len(self.bw_stages):
                self.exec_order.append(self.bw_stages[i])
            if i < len(self.fw_stages):
                self.exec_order.append(self.fw_stages[i])
        self.executed_fw_microbatches = 0
        self.executed_bw_microbatches = 0

    def reset(self):
        super().reset()
        for key in self.buffers.keys():
            self.buffers[key] = []
            self.next_step_buffers[key] = []
        self.executed_fw_microbatches = 0
        self.executed_bw_microbatches = 0

    def add_operation(self, op: ScheduleOperation):
        if op.is_forward:
            assert (
                op.flattened_stage in self.fw_stages
            ), "Operation {} not in executor".format(op)
            self.next_step_buffers[op.flattened_stage].append(op)
        else:
            assert (
                op.flattened_stage in self.bw_stages
            ), "Operation {} not in executor".format(op)
            self.next_step_buffers[op.flattened_stage].append(op)

    def forward_cycle(self):
        # append next_step_buffer to buffer
        for key, ops in self.next_step_buffers.items():
            self.buffers[key] += ops
            self.next_step_buffers[key] = []

    def exec_cycle(self, current_time):
        total_exec_time_in_cycle = 0
        executed_ops = []
        events = []
        available_ops: List[ScheduleOperation] = []
        for stage_id in self.fw_stages + self.bw_stages:
            if len(self.buffers[stage_id]) > 0:
                available_ops.append(self.buffers[stage_id][0])
        available_bw_ops = sorted(
            [op for op in available_ops if not op.is_forward],
            key=lambda x: (x.microbatch, x.flattened_stage),
        )
        available_fw_ops = sorted(
            [op for op in available_ops if op.is_forward],
            key=lambda x: (x.microbatch, x.flattened_stage),
        )
        # merge
        available_ops: List[ScheduleOperation] = []
        for i in range(max(len(available_fw_ops), len(available_bw_ops))):
            if i < len(available_bw_ops):
                available_ops.append(available_bw_ops[i])
            if i < len(available_fw_ops):
                available_ops.append(available_fw_ops[i])
        for op in available_ops:
            # test if executing this op will exceed memory limit
            if (
                not self.is_comm_stage
                and op.is_forward
                and op.microbatch > self.executed_fw_microbatches
                and (
                    self.current_memory + op.peak_memory > self.memory_limit
                    or self.executed_fw_microbatches
                    - self.executed_bw_microbatches
                    >= self.max_otf_microbatches
                )
            ):
                # skip this op
                continue
            event = self.get_exec_event(
                op,
                current_time,
                op.exec_time,
            )
            events.append(event)
            if not self.is_comm_stage:
                # we add two memory events for each op
                # one for the peak memory usage during the op
                # one for the stored memory usage after the op
                peak_time = current_time + op.exec_time / 2
                finish_time = current_time + op.exec_time
                memory_events = self.update_memory(
                    peak_time, op.peak_memory, finish_time, op.stored_memory
                )
                if self.include_memory_stats:
                    events += memory_events
            executed_ops.append(op)
            total_exec_time_in_cycle += op.exec_time
            current_time += op.exec_time
            if op.is_forward:
                self.fw_count += 1
                if op.microbatch > self.executed_fw_microbatches:
                    self.executed_fw_microbatches = op.microbatch
            else:
                self.bw_count += 1
                if op.microbatch > self.executed_bw_microbatches:
                    self.executed_bw_microbatches = op.microbatch
            self.buffers[op.flattened_stage].pop(0)
        return current_time, executed_ops, events
