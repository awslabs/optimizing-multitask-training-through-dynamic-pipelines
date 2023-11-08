# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .schedule_common import (
    DEBUG_PRINT_EXECUTORS,
    ExecutorIndex,
    ScheduleExecutor,
    ScheduleOperation,
    SchedulerMinibatchSpec,
)
from .wait_free_schedule import WaitFreeExecutor, WaitFreeScheduler


class OFOBExecutor(WaitFreeExecutor):
    def __init__(
        self,
        executor_id: int,
        thread_id: int,
        n_orig_layers: int,
        assigned_stages: List[Tuple[int, float, bool]],
        n_executors: int,
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
        if not self.is_comm_stage:
            assert len(self.fw_stages) == len(
                self.bw_stages
            ), "Mismatched number of forward and backward layers"
        self.is_executing = False
        self.next_op = (0, 0, True)  # (microbatch, chunk_id, is_forward)
        self.n_executors = n_executors
        self.n_microbatches = None
        self.executed_fw_ops = 0
        self.executed_bw_ops = 0
        self._increment_next_op_fn = None
        self._try_execute_fn = None

    def register_increment_next_op_fn(self, fn):
        self._increment_next_op_fn = fn

    def register_try_execute_fn(self, fn):
        self._try_execute_fn = fn

    def reset(self):
        super().reset()
        self.available_queue = []
        self.next_op = (0, 0, True)
        self.n_microbatches = None
        self.executed_fw_ops = 0
        self.executed_bw_ops = 0

    def set_n_microbatches(self, n_microbatches):
        self.n_microbatches = n_microbatches

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

    def _increment_next_op(self):
        assert self._increment_next_op_fn is not None
        return self._increment_next_op_fn(self)

    def try_execute(self, current_time):
        assert self._try_execute_fn is not None
        if (
            self.executed_fw_ops == 0
            and self.is_comm_stage
            and len(self.fw_stages) == 0
        ):
            # no fw layers assigned, skip once
            self.executed_fw_ops = 1
            self._increment_next_op()
        return self._try_execute_fn(self, current_time)

    def finish_execute(self):
        self.is_executing = False

    def debug_print(self, *args):
        # overrides parent debug_print
        if self.executor_id in DEBUG_PRINT_EXECUTORS and self.logger:
            self.logger.info(
                "Executor {} thread {} - {}".format(
                    self.executor_id,
                    self.thread_id,
                    " ".join([str(x) for x in args]),
                )
            )


class OFOBScheduler(WaitFreeScheduler):
    def __init__(
        self,
        executor_factory,
        minibatch_spec: SchedulerMinibatchSpec,
        include_memory_stats: bool = True,
        memory_limit: float = float("inf"),
        logger: Optional[logging.Logger] = None,
    ):
        self._executor_factory = executor_factory
        super().__init__(
            minibatch_spec,
            include_memory_stats,
            memory_limit,
            logger=logger,
        )
        self.executors: Dict[ExecutorIndex, OFOBExecutor]

    def _init_executors(self, n_microbatches, **kwargs):
        status = super()._init_executors(n_microbatches, **kwargs)
        if not status:
            return False
        for executor in self.executors.values():
            executor.set_n_microbatches(n_microbatches)
        return True

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
        n_executors = len(
            set(
                [
                    x.executor_id
                    for x in self.minibatch_spec.flattened_executor_assignment
                ]
            )
        )
        return self._executor_factory(
            executor_id,
            thread_id,
            n_orig_layers,
            assigned_stages,
            n_executors,
            is_comm_stage,
            include_memory_stats,
            parent_executor,
            logger=self.logger,
        )


class ExtendedOFOBScheduler(OFOBScheduler):
    def extended_init(self):
        self.dependency_map = defaultdict(list)
        self.rev_dependency_map = defaultdict(list)
        self.pending_ops = set()

    def add_dependency(
        self, src_microbatch, src_layer, dst_microbatch, dst_layer
    ):
        self.dependency_map[(dst_microbatch, dst_layer)].append(
            (src_microbatch, src_layer)
        )
        self.rev_dependency_map[(src_microbatch, src_layer)].append(
            (dst_microbatch, dst_layer)
        )

    def _inject_microbatches(
        self, microbatch_offset: int, n_microbatches: int
    ):
        for microbatch_id in range(
            microbatch_offset, microbatch_offset + n_microbatches
        ):
            if self.dependency_map[microbatch_id, 0]:
                self.pending_ops.add((microbatch_id, 0))
            else:
                executor = self.executors[
                    self.minibatch_spec.flattened_executor_assignment[0]
                ]
                op = self._get_op(0, microbatch_id)
                executor.add_operation(op)

    def _on_op_finish(self, executor: OFOBExecutor, op: ScheduleOperation):
        executor.finish_execute()

        def _release_ops(mb, flattened_stage_id):
            next_executor = self.minibatch_spec.flattened_executor_assignment[
                flattened_stage_id
            ]
            self.executors[next_executor].add_operation(
                self._get_op(flattened_stage_id, mb)
            )

        for dst_microbatch, dst_layer in self.rev_dependency_map[
            (op.microbatch, op.flattened_stage)
        ]:
            # remove dependency
            self.dependency_map[(dst_microbatch, dst_layer)].remove(
                (op.microbatch, op.flattened_stage)
            )
            if not self.dependency_map[(dst_microbatch, dst_layer)]:
                if (dst_microbatch, dst_layer) in self.pending_ops:
                    self.pending_ops.remove((dst_microbatch, dst_layer))
                    _release_ops(dst_microbatch, dst_layer)

        if op.flattened_stage < self.n_flattened_stages - 1:
            next_layer = op.flattened_stage + 1
            if self.dependency_map[op.microbatch, next_layer]:
                self.pending_ops.add((op.microbatch, next_layer))
            else:
                _release_ops(op.microbatch, next_layer)


class OFOBSchedulerRegistry:
    increment_op_fn_registry = {}
    try_execute_fn_registry = {}
    dependency_policy_registry = {}

    @classmethod
    def register_increment_op_fn(cls, name):
        def wrapper(fn):
            if name in cls.increment_op_fn_registry:
                raise ValueError(
                    "Increment Op Fn registered twice: {}".format(name)
                )
            cls.increment_op_fn_registry[name] = fn
            return cls

        return wrapper

    @classmethod
    def register_try_execute_fn(cls, name):
        def wrapper(fn):
            if name in cls.try_execute_fn_registry:
                raise ValueError(
                    "Try Execute Fn registered twice: {}".format(name)
                )
            cls.try_execute_fn_registry[name] = fn
            return cls

        return wrapper

    @classmethod
    def register_dependency_policy(cls, name):
        def wrapper(fn):
            if name in cls.dependency_policy_registry:
                raise ValueError(
                    "Dependency Policy registered twice: {}".format(name)
                )
            cls.dependency_policy_registry[name] = fn
            return cls

        return wrapper

    @classmethod
    def get_scheduler_factory(
        cls,
        placement_type="linear",
        strictness="strict",
        dependency_policy=None,
    ):
        if placement_type not in cls.increment_op_fn_registry:
            raise ValueError(
                "Invalid placement type: {}".format(placement_type)
            )
        if strictness not in cls.try_execute_fn_registry:
            raise ValueError("Invalid strictness: {}".format(strictness))
        if (
            dependency_policy is not None
            and dependency_policy not in cls.dependency_policy_registry
        ):
            raise ValueError(
                "Invalid dependency policy: {}".format(dependency_policy)
            )

        scheduler_cls = (
            OFOBScheduler
            if dependency_policy is None
            else ExtendedOFOBScheduler
        )

        def create_executor(
            executor_id: int,
            thread_id: int,
            n_orig_layers: int,
            assigned_stages: List[Tuple[int, float, bool]],
            n_executors: int,
            is_comm_stage: bool = False,
            include_memory_stats: bool = True,
            parent_executor: Optional[ScheduleExecutor] = None,
            logger: Optional[logging.Logger] = None,
        ):
            executor = OFOBExecutor(
                executor_id,
                thread_id,
                n_orig_layers,
                assigned_stages,
                n_executors,
                is_comm_stage,
                include_memory_stats,
                parent_executor,
                logger=logger,
            )
            executor.register_increment_next_op_fn(
                cls.increment_op_fn_registry[placement_type]
            )
            executor.register_try_execute_fn(
                cls.try_execute_fn_registry[strictness]
            )
            return executor

        def create_scheduler(
            minibatch_spec: SchedulerMinibatchSpec,
            include_memory_stats: bool = True,
            memory_limit: float = float("inf"),
            logger: Optional[logging.Logger] = None,
        ):
            scheduler = scheduler_cls(
                create_executor,
                minibatch_spec,
                include_memory_stats,
                memory_limit,
                logger=logger,
            )
            if dependency_policy is not None:
                scheduler.extended_init()
                n_minibatches = len(minibatch_spec.microbatches)
                n_stages = max(minibatch_spec.device_assignment) + 1
                per_device_assignments = defaultdict(int)
                for device in minibatch_spec.device_assignment:
                    per_device_assignments[device] += 1
                n_chunks = per_device_assignments[0]
                for _, n_chunks_at_device in per_device_assignments.items():
                    assert (
                        n_chunks_at_device == n_chunks
                    ), "All devices must have the same number of chunks"
                dependencies = cls.dependency_policy_registry[
                    dependency_policy
                ](n_minibatches, n_stages, n_chunks)
                for src, dst in dependencies:
                    scheduler.add_dependency(*src, *dst)
            return scheduler

        return create_scheduler


@OFOBSchedulerRegistry.register_increment_op_fn("linear")
def increment_next_op_linear(self: OFOBExecutor):
    if not self.is_comm_stage:
        assert len(self.fw_stages) == len(self.bw_stages) == 1, (
            "Linear placement only supports 1 layer per executor, "
            "but got {} FW and {} BW stages".format(
                len(self.fw_stages), len(self.bw_stages)
            )
        )
    n_unflattened_stages = self.n_executors
    n_warmup_microbatches = n_unflattened_stages - self.executor_id
    n_warmup_microbatches = min(n_warmup_microbatches, self.n_microbatches)
    n_remaining_microbatches = self.n_microbatches - n_warmup_microbatches
    self.debug_print(
        f"n_unflattened_stages: {n_unflattened_stages}, "
        f"n_warmup_microbatches: {n_warmup_microbatches}, "
        f"n_remaining_microbatches: {n_remaining_microbatches}"
    )
    if self.executed_fw_ops < n_warmup_microbatches:
        # warmup stage
        self.next_op = (self.executed_fw_ops, 0, True)
        self.debug_print("In warmup stage: next op is", self.next_op)
    elif (
        self.executed_fw_ops < n_warmup_microbatches + n_remaining_microbatches
    ):
        # steady state stage
        if self.next_op[2]:
            # next is bw
            self.next_op = (self.executed_bw_ops, 0, False)
        else:
            # next is fw
            self.next_op = (self.executed_fw_ops, 0, True)
        self.debug_print("In steady state: next op is", self.next_op)
    else:
        # cooldown stage
        self.next_op = (self.executed_bw_ops, 0, False)
        self.debug_print("In cooldown stage: next op is", self.next_op)
    if self.thread_id == 1:
        # communication thread
        if (
            self.executor_id == n_unflattened_stages - 1
            and self.next_op[2] is True
            and self.next_op[0] <= self.n_microbatches - 1
        ):
            # FW of last layer have no comm
            self.executed_fw_ops += 1
            self._increment_next_op()
        if (
            self.executor_id == 0
            and self.next_op[2] is False
            and self.next_op[0] <= self.n_microbatches - 1
        ):
            # BW of first layer have no comm
            self.executed_bw_ops += 1
            self._increment_next_op()


@OFOBSchedulerRegistry.register_increment_op_fn("interleaved")
def increment_next_op_interleaved(self: OFOBExecutor):
    n_unflattened_stages = self.n_executors
    if (self.thread_id == 0) or self.executor_id != n_unflattened_stages - 1:
        n_chunks = len(self.fw_stages)
    else:
        n_chunks = len(self.fw_stages) + 1
    n_warmup_microbatches = (
        (n_unflattened_stages - self.executor_id - 1) * 2
    ) + ((n_chunks - 1) * n_unflattened_stages)
    n_warmup_microbatches = min(
        n_warmup_microbatches,
        (self.n_microbatches // n_unflattened_stages)
        * n_unflattened_stages
        * n_chunks,
    )
    if n_warmup_microbatches == 0:
        # num microbatches is less than n_stages
        n_warmup_microbatches = self.n_microbatches * n_chunks
    n_remaining_microbatches = (
        self.n_microbatches * n_chunks
    ) - n_warmup_microbatches
    self.debug_print(
        f"n_unflattened_stages: {n_unflattened_stages}, n_chunks: {n_chunks}, "
        f"n_warmup_microbatches: {n_warmup_microbatches}, "
        f"n_remaining_microbatches: {n_remaining_microbatches}"
    )

    def _locate_cycles(executed_ops):
        # executed_ops // (n_stages * n_chunks): number of full cycles
        # number of full cycles * n_stages:
        #   each full cycle has n_stages microbatches
        # executed_ops % n_stages: microbatch id within the current cycle
        num_full_cycles = executed_ops // (n_unflattened_stages * n_chunks)
        num_microbatches_in_full_cycles = (
            num_full_cycles * n_unflattened_stages
        )
        if (
            num_microbatches_in_full_cycles + n_unflattened_stages
        ) > self.n_microbatches:
            # last cycle, we may have less than n_stages microbatches
            last_cycle_size = (
                self.n_microbatches - num_microbatches_in_full_cycles
            )
            if last_cycle_size == 0:
                last_cycle_size = n_unflattened_stages
        else:
            last_cycle_size = n_unflattened_stages
        remainder_microbatches = (
            executed_ops - num_microbatches_in_full_cycles * n_chunks
        ) % n_unflattened_stages
        return (
            num_microbatches_in_full_cycles,
            last_cycle_size,
            remainder_microbatches,
        )

    def _get_microbatch(executed_ops):
        (
            num_microbatches_in_full_cycles,
            last_cycle_size,
            remainder,
        ) = _locate_cycles(executed_ops)
        if remainder >= last_cycle_size:
            # wait
            return None
        return num_microbatches_in_full_cycles + remainder

    def _get_chunk_id(executed_ops):
        num_microbatches_in_full_cycles, _, _ = _locate_cycles(executed_ops)
        chunk_id = (
            executed_ops - num_microbatches_in_full_cycles * n_chunks
        ) // n_unflattened_stages
        return chunk_id

    rounded_total_chunks = (
        (self.n_microbatches + n_unflattened_stages - 1)
        // n_unflattened_stages
        * n_unflattened_stages
    ) * n_chunks
    if not hasattr(self, "_first_mb_of_steady_state"):
        self._first_mb_of_steady_state = True
    if self.executed_fw_ops < n_warmup_microbatches:
        # warmup stage
        chunk_id = _get_chunk_id(self.executed_fw_ops)
        microbatch_id = _get_microbatch(self.executed_fw_ops)
        if microbatch_id is None:
            self.executed_fw_ops += 1
            return self._increment_next_op_fn(self)
        self.next_op = (microbatch_id, chunk_id, True)
        self.debug_print("In warmup stage: next op is", self.next_op)
    elif self.executed_fw_ops < rounded_total_chunks:
        # steady state stage
        if self.next_op[2] and not self._first_mb_of_steady_state:
            # next is bw
            chunk_id = _get_chunk_id(self.executed_bw_ops)
            microbatch_id = _get_microbatch(self.executed_bw_ops)
            self.next_op = (microbatch_id, chunk_id, False)
            if microbatch_id is None:
                self.executed_bw_ops += 1
                return self._increment_next_op_fn(self)
            self.debug_print(
                "In steady stage, first_mb_of_steady_state = "
                f"{self._first_mb_of_steady_state}, next op is",
                self.next_op,
            )
        else:
            # next is fw
            chunk_id = _get_chunk_id(self.executed_fw_ops)
            microbatch_id = _get_microbatch(self.executed_fw_ops)
            self.next_op = (microbatch_id, chunk_id, True)
            if microbatch_id is None:
                self.executed_fw_ops += 1
                return self._increment_next_op_fn(self)
            self.debug_print(
                "In steady stage, first_mb_of_steady_state = "
                f"{self._first_mb_of_steady_state}, next op is",
                self.next_op,
            )
        self._first_mb_of_steady_state = False
    else:
        # cooldown stage
        chunk_id = _get_chunk_id(self.executed_bw_ops)
        microbatch_id = _get_microbatch(self.executed_bw_ops)
        self.next_op = (microbatch_id, chunk_id, False)
        if microbatch_id is None:
            if self.executed_bw_ops >= rounded_total_chunks:
                return (None, 0, False)
            self.executed_bw_ops += 1
            return self._increment_next_op_fn(self)
        self.debug_print("In cooldown stage, next op is", self.next_op)
    if self.thread_id == 1:
        # communication thread
        if (
            self.executor_id == n_unflattened_stages - 1
            and self.next_op[2] is True
            and self.next_op[1] == n_chunks - 1
        ):
            # FW of last layer have no comm
            self.executed_fw_ops += 1
            self._increment_next_op()
        if (
            self.executor_id == 0
            and self.next_op[2] is False
            and self.next_op[1] == n_chunks - 1
        ):
            # BW of first layer have no comm
            self.executed_bw_ops += 1
            self._increment_next_op()


@OFOBSchedulerRegistry.register_try_execute_fn("strict")
def try_execute_strict(self: OFOBExecutor, current_time):
    assert self.n_microbatches is not None, "n_microbatches not set"
    events = []
    if self.available_queue and not self.is_executing:
        next_op = None
        for op in self.available_queue:
            if (
                op.microbatch == self.next_op[0]
                and op.is_forward == self.next_op[2]
            ):
                next_op = op
                break
        if next_op is not None:
            self.available_queue.remove(next_op)
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
            if next_op.is_forward:
                self.executed_fw_ops += 1
            else:
                self.executed_bw_ops += 1
            self._increment_next_op()
            return finish_time, next_op, events
    return current_time, None, events


def try_execute_relaxed_helper(
    self: OFOBExecutor, current_time, accumulation_limit
):
    assert self.n_microbatches is not None, "n_microbatches not set"
    events = []
    if self.available_queue and not self.is_executing:
        self.available_queue.sort(
            key=lambda op: (0 if not op.is_forward else 1, op.microbatch)
        )
        next_op = None
        for op in self.available_queue:
            if (
                not op.is_forward
                or self.executed_fw_ops - self.executed_bw_ops
                < accumulation_limit
            ):
                next_op = op
                break
        if next_op is not None:
            self.available_queue.remove(next_op)
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
            if next_op.is_forward:
                self.executed_fw_ops += 1
            else:
                self.executed_bw_ops += 1
            return finish_time, next_op, events
    return current_time, None, events


@OFOBSchedulerRegistry.register_try_execute_fn("relaxed")
def try_execute_relaxed(self: OFOBExecutor, current_time):
    return try_execute_relaxed_helper(self, current_time, self.n_executors)


@OFOBSchedulerRegistry.register_try_execute_fn("interleaved-relaxed")
def try_execute_relaxed_interleave(self: OFOBExecutor, current_time):
    return try_execute_relaxed_helper(
        self, current_time, len(self.fw_stages) * self.n_executors
    )


@OFOBSchedulerRegistry.register_dependency_policy("cyclic")
def cyclic_dependency_policy(n_microbatches, n_stages, n_chunks):
    fw_iterations = []
    bw_iterations = []
    for executor_id in range(n_stages):
        n_warmup_microbatches = ((n_stages - executor_id - 1) * 2) + (
            (n_chunks - 1) * n_stages
        )
        n_warmup_microbatches = min(
            n_warmup_microbatches, n_microbatches * n_chunks
        )
        n_remaining_microbatches = (
            n_microbatches * n_chunks
        ) - n_warmup_microbatches

        def _locate_cycles(executed_ops):
            # executed_ops // (self.n_stages * n_chunks): number of full cycles
            # number of full cycles * self.n_stages:
            #    each full cycle has self.n_stages microbatches
            # executed_ops % self.n_stages:
            #    microbatch id within the current cycle
            num_full_cycles = executed_ops // (n_stages * n_chunks)
            num_microbatches_in_full_cycles = num_full_cycles * n_stages
            if (num_microbatches_in_full_cycles + n_stages) > n_microbatches:
                # last cycle, we may have less than self.n_stages microbatches
                last_cycle_size = (
                    n_microbatches - num_microbatches_in_full_cycles
                )
                if last_cycle_size == 0:
                    last_cycle_size = n_stages
                # loop inside the last cycle
                remainder = executed_ops % last_cycle_size
            else:
                last_cycle_size = n_stages
                remainder = executed_ops % n_stages
            return num_microbatches_in_full_cycles, last_cycle_size, remainder

        def _get_layer_id(executed_ops, is_fw):
            (
                num_microbatches_in_full_cycles,
                last_cycle_size,
                _,
            ) = _locate_cycles(executed_ops)
            fw_layer_id = (
                (executed_ops - (num_microbatches_in_full_cycles * n_chunks))
                // last_cycle_size
            ) * n_stages + executor_id
            bw_layer_id = (
                n_stages * n_chunks
                + (
                    (
                        executed_ops
                        - (num_microbatches_in_full_cycles * n_chunks)
                    )
                    // last_cycle_size
                )
                * n_stages
                + (n_stages - executor_id - 1)
            )
            if is_fw:
                return fw_layer_id
            else:
                return bw_layer_id

        def _get_microbatch(executed_ops):
            num_microbatches_in_full_cycles, _, remainder = _locate_cycles(
                executed_ops
            )
            return num_microbatches_in_full_cycles + remainder

        total_iters = n_chunks * n_microbatches - n_stages
        last_cycle_size = n_microbatches % n_stages
        filler_microbatches = 0
        if last_cycle_size != 0:
            filler_microbatches = n_stages - last_cycle_size
        num_microbatches_in_full_cycles = n_microbatches // n_stages * n_stages
        fw_op_orders = list(
            reversed(
                [
                    (
                        _get_microbatch(n_warmup_microbatches - i - 1),
                        _get_layer_id(n_warmup_microbatches - i - 1, True),
                    )
                    for i in range((n_stages - executor_id - 1))
                ]
            )
        )
        bw_op_orders = [None for i in range((n_stages - executor_id - 1))]

        executed_fw_ops = n_warmup_microbatches
        executed_bw_ops = 0
        for _ in range(total_iters - (n_stages - executor_id - 1)):
            none_appended = False
            if (
                executed_fw_ops
                < n_warmup_microbatches + n_remaining_microbatches
            ):
                if (
                    executed_fw_ops
                    == num_microbatches_in_full_cycles * n_chunks
                    + last_cycle_size
                ):
                    for _ in range(filler_microbatches):
                        fw_op_orders.append(None)
                    none_appended = True
                fw_op_orders.append(
                    (
                        _get_microbatch(executed_fw_ops),
                        _get_layer_id(executed_fw_ops, True),
                    )
                )
                executed_fw_ops += 1
            else:
                fw_op_orders.append(None)
            if none_appended:
                if executor_id != n_stages - 1:
                    for _ in range(filler_microbatches):
                        bw_op_orders.append(None)
                else:
                    for _ in range(filler_microbatches):
                        bw_op_orders.insert(-1, None)
            bw_op_orders.append(
                (
                    _get_microbatch(executed_bw_ops),
                    _get_layer_id(executed_bw_ops, False),
                )
            )
            executed_bw_ops += 1

        fw_iterations.append(fw_op_orders)
        bw_iterations.append(bw_op_orders)

    dependencies = []
    len_iters = len(fw_iterations[0])
    for it in range(1, len_iters):
        it_fws = [
            fw_iterations[executor_id][it] for executor_id in range(n_stages)
        ]
        prev_it_bws = [
            bw_iterations[executor_id][it - 1]
            for executor_id in range(n_stages)
        ]
        for it_fw in it_fws:
            for prev_it_bw in prev_it_bws:
                if it_fw is not None and prev_it_bw is not None:
                    dependencies.append((prev_it_bw, it_fw))
    return dependencies
