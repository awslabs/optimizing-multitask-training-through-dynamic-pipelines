# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dynapipe.pipe.instructions import *  # noqa: F403

DEBUG_REPLACE_OP_NAMES_WITH_INSTRS = False
DEBUG_REPLACE_OP_NAMES_WITH_SCH_STATS = False
DEBUG_PRINT_EXECUTORS = []


# n_layers refers to the number of layers (en+de) in the original network
# flattened structure:
# no comm:
# stage: [fw, fw, fw, ..., fw, bw, bw, bw, ..., bw]
# index: [0,  1,  2,  ..., n-1, n, n+1,n+2, ..., 2n-1]
# with comm:
# stage: [fw, comm, fw, comm, ...,  fw,   bw, comm, bw,  comm, ...,  bw ]
# index: [0,    1,   2,   3,  ..., 2n-2, 2n-1, 2n, 2n+1, 2n+2, ..., 4n-3]
def _is_last_fw_or_bw_stage(
    flattened_stage_id, n_orig_layers, comm_added=True
):
    if not comm_added:
        return (flattened_stage_id == (n_orig_layers - 1)) or (
            flattened_stage_id == (2 * n_orig_layers - 1)
        )
    return (flattened_stage_id == (2 * n_orig_layers - 2)) or (
        flattened_stage_id == (4 * n_orig_layers - 3)
    )


# the following functions assumes comm stages are added
def _is_fw_stage(flattened_stage_id, n_orig_layers):
    return flattened_stage_id < 2 * n_orig_layers - 1


def _is_comm_stage(flattened_stage_id, n_orig_layers):
    if flattened_stage_id < 2 * n_orig_layers - 1:
        residual = 1
    else:
        residual = 0
    return flattened_stage_id % 2 == residual


def _is_first_bw_layer(flattened_stage_id, n_orig_layers):
    # communication of the first bw layer is ignored
    return flattened_stage_id == 2 * n_orig_layers - 1


# in instructions, we only count the computation stages for better readability
def _get_comp_only_stage_index(flattened_stage_id, n_orig_layers):
    if flattened_stage_id < 2 * n_orig_layers - 1:
        return flattened_stage_id // 2
    else:
        return (flattened_stage_id + 1) // 2


ExecutorIndex = namedtuple("ExecutorIndex", ["executor_id", "thread_id"])


class ExecutorThread:
    COMP_THREAD = 0
    COMM_THREAD = 1


@dataclass
class SchedulerMicrobatchSpec:
    """Microbatch specification for scheduler.
    `stage` in scheduler includes both communication and computation.
    """

    def __init__(
        self,
        name: str,
        fw_times: List[float],
        fw_comm_times: List[float],
        fw_stored_activation_size: List[float],
        fw_peak_activation_size: List[float],
        activation_shapes: List[List[Tuple[int, int, int]]],
        bw_times: List[int],
        bw_comm_times: List[int],
    ):
        self.name = name
        self._fw_times = fw_times
        # last forward and backward layer do not have communication time
        self._fw_comm_times = fw_comm_times
        self._fw_stored_activation_size = fw_stored_activation_size
        self._fw_peak_activation_size = fw_peak_activation_size
        self._activation_shapes = activation_shapes
        self._bw_times = bw_times
        self._bw_comm_times = bw_comm_times
        # number of layers in the original network
        self.n_orig_layers = len(fw_times)

        self._layers_merged = False
        self._raw_attrs = [
            "_fw_times",
            "_fw_comm_times",
            "_fw_stored_activation_size",
            "_fw_peak_activation_size",
            "_activation_shapes",
            "_bw_times",
            "_bw_comm_times",
        ]
        self._initialized = False
        self._validate_spec()

    def _validate_spec(self):
        # check if the spec is valid
        assert len(self._fw_times) == len(self._activation_shapes), (
            "fw_times must be of same length as activation_shapes, "
            "but got {} and {}".format(
                len(self._fw_times), len(self._activation_shapes)
            )
        )
        assert (
            len(self._fw_comm_times) == len(self._fw_times) - 1
        ), "Invalid comm times length, expected {} but got {}".format(
            len(self._fw_times) - 1, len(self._fw_comm_times)
        )
        assert len(self._bw_times) == len(
            self._fw_times
        ), "bw_times must be of same length as fw_times"
        assert len(self._bw_comm_times) == len(
            self._fw_comm_times
        ), "bw_comm_times must be of same length as fw_comm_times"
        assert len(self._fw_stored_activation_size) == len(
            self._fw_times
        ), "fw_stored_activation must be of same length as fw_times"
        assert len(self._fw_peak_activation_size) == len(
            self._fw_times
        ), "fw_peak_activation must be of same length as fw_times"

    def _merge_layers(self, merged2orig: Dict[int, int]):
        # merge consecutive layers within same executor,
        # so we don't need to waste extra cycles waiting
        # construct new argument arrays
        # merged2orig is a dict of {merged_layer_id: [orig_layer_ids]}
        merged_attrs = {attr: [] for attr in self._raw_attrs}
        for merged_layer_id in sorted(merged2orig.keys()):
            accum_attrs = {
                attr: 0 for attr in self._raw_attrs if "comm_times" not in attr
            }
            for orig_layer_id in merged2orig[merged_layer_id]:
                # for peak activation, we accumulate prior layers' stored
                # activation, and add the current layer's peak activation
                accum_attrs["_fw_peak_activation_size"] = max(
                    accum_attrs["_fw_stored_activation_size"]
                    + self._fw_peak_activation_size[orig_layer_id],
                    accum_attrs["_fw_peak_activation_size"],
                )
                # merge fw computation related attrs
                for attr in ["_fw_times", "_fw_stored_activation_size"]:
                    accum_attrs[attr] += getattr(self, attr)[orig_layer_id]
                # bw computation related attrs
                for attr in ["_bw_times"]:
                    # indexed in reverse
                    accum_attrs[attr] += getattr(self, attr)[
                        -(orig_layer_id + 1)
                    ]
                # we only keep the activation shape of the last layer on that
                # device
                for attr in ["_activation_shapes"]:
                    accum_attrs[attr] = getattr(self, attr)[orig_layer_id]
            # communication related attrs
            # last merged layer does not have communication time
            if merged_layer_id != len(merged2orig) - 1:
                for attr in ["_fw_comm_times", "_bw_comm_times"]:
                    # we only keep the communication time of the last layer
                    accum_attrs[attr] = getattr(self, attr)[
                        merged2orig[merged_layer_id][-1]
                    ]
            for attr in self._raw_attrs:
                if attr in accum_attrs:
                    merged_attrs[attr].append(accum_attrs[attr])
        # reverse bw related attrs
        for attr in ["_bw_times", "_bw_comm_times"]:
            merged_attrs[attr].reverse()
        # update self
        for attr in self._raw_attrs:
            setattr(self, attr, merged_attrs[attr])
        self.n_orig_layers = len(merged2orig)
        self._layers_merged = True

    def _flatten(self):
        # Flattening concatenates the backward pass to the forward ones
        # so each microbatch consists of a single list of 'jobs'
        # Communication also forms their own stages. There is no comm
        # stage between the last forward stage and the first backward stage.
        # e.g. a flattened 4 layer network
        # [fw1, comm1, fw2, comm2, fw3, comm3, fw4,
        #  bw4, comm3b, bw3, comm2b, bw2, comm1b, bw1]

        # assume the original (merged but not flattened) has n layers

        # flattened_stored_activation_sizes has length 2n
        # we use negative numbers to indicate memory freed in backward pass
        self.flattened_stored_activation_sizes = (
            self._fw_stored_activation_size
            + [-x for x in reversed(self._fw_stored_activation_size)]
        )
        # we don't count peak activations during backward pass
        self.flattened_peak_activation_sizes = (
            self._fw_peak_activation_size
            + [0 for _ in reversed(self._fw_peak_activation_size)]
        )
        # flattened_exec_times has length 2n
        self.flattened_exec_times = self._fw_times + self._bw_times

        backward_activation_shapes = list(reversed(self._activation_shapes))
        # if is ende, drop decoder gradients for the last backward of decoder
        if len(backward_activation_shapes[0]) > len(
            backward_activation_shapes[-1]
        ):
            for layer, shapes in enumerate(backward_activation_shapes):
                if len(shapes) == len(backward_activation_shapes[0]) and len(
                    backward_activation_shapes[layer + 1]
                ) == len(backward_activation_shapes[-1]):
                    backward_activation_shapes[layer] = shapes[
                        : len(backward_activation_shapes[-1])
                    ]
                    break
        # flattened_activation_shapes has length 2n
        self.flattened_activation_shapes = (
            self._activation_shapes + backward_activation_shapes
        )

        self._flattened_attrs = [
            "flattened_stored_activation_sizes",
            "flattened_peak_activation_sizes",
            "flattened_exec_times",
            "flattened_activation_shapes",
        ]
        # flattened_comm_times has length 2n, note that we pad
        # the non-existing comm times with 0

        # add comm stages into the flattened stats
        flat_attrs_w_comm = {attr: [] for attr in self._flattened_attrs}
        # append 0 to match flattened_stage_idx
        flattened_comm_times = (
            self._fw_comm_times + [0] + self._bw_comm_times + [0]
        )
        for flattened_stage_idx in range(len(self.flattened_exec_times)):
            # add the original stage
            for attr in self._flattened_attrs:
                flat_attrs_w_comm[attr].append(
                    getattr(self, attr)[flattened_stage_idx]
                )
            if not _is_last_fw_or_bw_stage(
                flattened_stage_idx, self.n_orig_layers, comm_added=False
            ):
                # add the comm stage
                flat_attrs_w_comm["flattened_exec_times"].append(
                    flattened_comm_times[flattened_stage_idx]
                )
                # comm stage don't have activation size
                flat_attrs_w_comm["flattened_stored_activation_sizes"].append(
                    0
                )
                flat_attrs_w_comm["flattened_peak_activation_sizes"].append(0)
                flat_attrs_w_comm["flattened_activation_shapes"].append(
                    self.flattened_activation_shapes[flattened_stage_idx]
                )
        # update self. new flattened attrs have length 4n - 2
        for attr in self._flattened_attrs:
            setattr(self, attr, flat_attrs_w_comm[attr])

    def initialize(self):
        self._flatten()
        self._initialized = True

    def is_layers_merged(self):
        return self._layers_merged


@dataclass
class SchedulerMinibatchSpec:
    name: str
    microbatches: List[SchedulerMicrobatchSpec]
    device_assignment: List[float]
    model_states: List[float]
    n_orig_layers: int = field(init=False, default=None, repr=True)
    _layers_merged: bool = field(init=False, default=False, repr=False)
    _initialized: bool = field(init=False, default=False, repr=False)

    def __post_init__(self):
        assert sorted(list(set(self.device_assignment))) == list(
            range(len(set(self.device_assignment)))
        ), (
            "device_assignment must be a list of consecutive integers "
            "starting from 0"
        )
        self.n_orig_layers = len(self.device_assignment)

    def _merge_layers(self):
        # merge consecutive layers within same executor,
        # so we don't need to waste extra cycles waiting
        merged2orig = {}
        orig2merged = {}
        for idx, executor in enumerate(self.device_assignment):
            if idx != 0 and executor == self.device_assignment[idx - 1]:
                # append current layer to previous layer's merged layer
                merged_layer_id = orig2merged[idx - 1]
                merged2orig[merged_layer_id].append(idx)
                orig2merged[idx] = merged_layer_id
            else:
                merged2orig[len(merged2orig)] = [idx]
                orig2merged[idx] = len(merged2orig) - 1
        # construct new argument arrays
        new_model_state = []
        new_device_assignment = []
        for merged_layer_id in sorted(merged2orig.keys()):
            merged_model_state = 0
            for orig_layer_id in merged2orig[merged_layer_id]:
                merged_model_state += self.model_states[orig_layer_id]
            new_model_state.append(merged_model_state)
            new_device_assignment.append(
                self.device_assignment[merged2orig[merged_layer_id][0]]
            )
        self.model_states = new_model_state
        self.device_assignment = new_device_assignment
        self.n_orig_layers = len(merged2orig)
        self.merged2original = merged2orig
        # call merge layers on each microbatch
        for microbatch in self.microbatches:
            microbatch._merge_layers(merged2orig)
        self._layers_merged = True

    def _flatten(self):
        self.flattened_model_states = self.model_states
        self.flattened_executor_assignment: List[ExecutorIndex] = []
        # add comm stages into the flattened stats
        # note that backward stages
        flat_state_w_comm = []
        for idx in range(self.n_orig_layers):
            flat_state_w_comm.append(self.flattened_model_states[idx])
            if idx < self.n_orig_layers - 1:
                flat_state_w_comm.append(0)
        # bw layers always have 0 model state
        for idx in range(self.n_orig_layers):
            flat_state_w_comm.append(0)
            if idx < self.n_orig_layers - 1:
                flat_state_w_comm.append(0)
        # fw bw concated and added comm, length 4n - 2
        self.flattened_model_states = flat_state_w_comm
        # modify device assignment as well
        # note that we use a tuple (executor_idx, executor_thread) to represent
        # an executor
        for layer_idx, fw_executor_id in enumerate(self.device_assignment):
            self.flattened_executor_assignment.append(
                ExecutorIndex(fw_executor_id, ExecutorThread.COMP_THREAD)
            )
            if layer_idx < len(self.device_assignment) - 1:
                self.flattened_executor_assignment.append(
                    ExecutorIndex(fw_executor_id, ExecutorThread.COMM_THREAD)
                )
        # use reversed device assignment for backward layers
        for bw_layer_idx, bw_executor_id in enumerate(
            reversed(self.device_assignment)
        ):
            self.flattened_executor_assignment.append(
                ExecutorIndex(bw_executor_id, ExecutorThread.COMP_THREAD)
            )
            if bw_layer_idx < len(self.device_assignment) - 1:
                self.flattened_executor_assignment.append(
                    ExecutorIndex(bw_executor_id, ExecutorThread.COMM_THREAD)
                )
        self.n_flattened_stages = 4 * self.n_orig_layers - 2

    def initialize(self):
        self._merge_layers()
        self._flatten()
        for microbatch in self.microbatches:
            microbatch.initialize()
        self._initialized = True


@dataclass(frozen=True, eq=True)
class ScheduleOperation:
    name: str
    microbatch: int
    flattened_stage: int
    exec_time: float
    is_forward: bool
    stored_memory: float
    peak_memory: float
    is_first_bw_layer: bool
    tensor_shape: Tuple[Tuple[int, int, int]]
    loads_data: Tuple[Tuple[int, int, int], ...] = tuple()
    next_executor: Optional["ScheduleExecutor"] = None

    def __repr__(self) -> str:
        return (
            "(name={}, m{}, fst{}, fw={}, shape={}, next_executor: {})".format(
                self.name,
                self.microbatch,
                self.flattened_stage,
                self.is_forward,
                self.tensor_shape,
                self.next_executor.executor_id if self.next_executor else None,
            )
        )

    @classmethod
    def from_operation(
        cls,
        operation: "ScheduleOperation",
        **kwargs,
    ):
        fields = dataclasses.asdict(operation)
        fields.update(kwargs)
        return cls(**fields)


class ScheduleExecutor:
    def __init__(
        self,
        executor_id: int,
        thread_id: int,
        n_orig_layers: int,
        assigned_stages: List[Tuple[int, float, bool]],
        is_comm_stage: bool = False,
        include_memory_stats: bool = True,
        parent_executor: Optional["ScheduleExecutor"] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.executor_id = executor_id
        self.fw_stages = []
        self.bw_stages = []
        self.n_orig_layers = n_orig_layers
        self.peak_memory = 0
        self.model_state_memory = 0
        for stage_id, memory, is_fw in assigned_stages:
            if is_fw:
                self.model_state_memory += memory
                self.fw_stages.append(stage_id)
            else:
                self.bw_stages.append(stage_id)
        self.fw_stages = sorted(self.fw_stages)
        self.bw_stages = sorted(self.bw_stages)
        if not is_comm_stage:
            assert len(self.fw_stages) == len(self.bw_stages), (
                "Unequal number of forward and backward stages "
                "({} vs {}) on executor {}.".format(
                    self.fw_stages, self.bw_stages, executor_id
                )
            )
        self.current_memory = self.model_state_memory
        self.fw_count = 0
        self.bw_count = 0
        self.is_comm_stage = is_comm_stage
        self.thread_id = thread_id
        self.include_memory_stats = include_memory_stats
        self.parent_executor = parent_executor
        self.traced_instructions = (
            []
            if parent_executor is None
            else parent_executor.traced_instructions
        )
        self.logger = logger

    def reset(self):
        self.current_memory = self.model_state_memory
        self.peak_memory = 0

    def process_name(self):
        return "Executor {}".format(self.executor_id)

    def thread_name(self):
        return "Compute" if not self.is_comm_stage else "Comm"

    def full_name(self):
        return "{} {}".format(self.process_name(), self.thread_name())

    def debug_print(self, *args):
        if self.executor_id in DEBUG_PRINT_EXECUTORS and self.logger:
            self.logger.info(
                "Executor {} thread {} - {}".format(
                    self.executor_id,
                    self.thread_id,
                    " ".join([str(x) for x in args]),
                )
            )

    def get_metadata(self):
        metadata = []
        if self.thread_id == 0:
            metadata += [
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": self.executor_id,
                    "tid": 0,
                    "args": {
                        "name": self.process_name(),
                    },
                },
                {
                    "name": "process_sort_index",
                    "ph": "M",
                    "pid": self.executor_id,
                    "tid": 0,
                    "args": {
                        "sort_index": self.executor_id,
                    },
                },
            ]
        metadata += [
            {
                "name": "thread_name",
                "ph": "M",
                "pid": self.executor_id,
                "tid": self.thread_id,
                "args": {
                    "name": self.thread_name(),
                },
            },
            {
                "name": "thread_sort_index",
                "ph": "M",
                "pid": self.executor_id,
                "tid": self.thread_id,
                "args": {
                    "sort_index": self.thread_id,
                },
            },
        ]
        return metadata

    def _get_duration_event(self, name, start_time, duration):
        return {
            "name": name,
            "ph": "X",
            "ts": start_time,
            "dur": duration,
            "pid": self.executor_id,
            "tid": self.thread_id,
            "args": {
                # anything helpful?
            },
        }

    def get_exec_event(
        self,
        op: ScheduleOperation,
        start_time,
        duration,
    ):
        if self.is_comm_stage:
            return self.get_comm_event(op, start_time, duration)
        # create a instruction
        self.debug_print("Current op: ", op)
        if op.loads_data:
            instruction = LoadInput(
                op.microbatch,
                _get_comp_only_stage_index(
                    op.flattened_stage, self.n_orig_layers
                ),
                buffer_shapes=list(op.loads_data),
            )
            self.traced_instructions.append(instruction)
            self.debug_print(
                "Appending instruction",
                instruction,
                "to executor",
                self.executor_id,
            )
        if op.is_forward:
            instruction = ForwardPass(
                op.microbatch,
                _get_comp_only_stage_index(
                    op.flattened_stage, self.n_orig_layers
                ),
                buffer_shapes=list(op.tensor_shape),
            )
        else:
            instruction = BackwardPass(
                op.microbatch,
                _get_comp_only_stage_index(
                    op.flattened_stage, self.n_orig_layers
                ),
                buffer_shapes=list(op.tensor_shape),
                first_bw_layer=op.is_first_bw_layer,
            )
        self.traced_instructions.append(instruction)
        self.debug_print(
            "Appending instruction",
            instruction,
            "to executor",
            self.executor_id,
        )
        if DEBUG_REPLACE_OP_NAMES_WITH_INSTRS:
            op_name = str(instruction)
        elif DEBUG_REPLACE_OP_NAMES_WITH_SCH_STATS:
            op_name = str(op)
        else:
            op_name = op.name + ("B" if not op.is_forward else "")
        return self._get_duration_event(op_name, start_time, duration)

    def get_comm_event(
        self,
        op: ScheduleOperation,
        start_time,
        duration,
    ):
        assert (
            op.next_executor is not None
        ), "No target executor for communication operation."
        if op.is_forward:
            send_instr = SendActivationStart(
                op.microbatch,
                _get_comp_only_stage_index(
                    op.flattened_stage, self.n_orig_layers
                ),
                op.next_executor.executor_id,
                buffer_shapes=list(op.tensor_shape),
            )
            # recv instr is linked with the next stage
            recv_instr = RecvActivationStart(
                op.microbatch,
                _get_comp_only_stage_index(
                    op.flattened_stage, self.n_orig_layers
                )
                + 1,
                self.executor_id,
                buffer_shapes=list(op.tensor_shape),
            )
        else:
            send_instr = SendGradStart(
                op.microbatch,
                _get_comp_only_stage_index(
                    op.flattened_stage, self.n_orig_layers
                ),
                op.next_executor.executor_id,
                buffer_shapes=list(op.tensor_shape),
            )
            # same here
            recv_instr = RecvGradStart(
                op.microbatch,
                _get_comp_only_stage_index(
                    op.flattened_stage, self.n_orig_layers
                )
                + 1,
                self.executor_id,
                buffer_shapes=list(op.tensor_shape),
            )
        self.debug_print("Current op: ", op)
        self.debug_print(
            "Appending instruction",
            send_instr,
            "to executor",
            self.executor_id,
        )
        self.traced_instructions.append(send_instr)
        self.debug_print(
            "Appending instruction",
            recv_instr,
            "to executor",
            op.next_executor.executor_id,
        )
        op.next_executor.traced_instructions.append(recv_instr)
        if DEBUG_REPLACE_OP_NAMES_WITH_INSTRS:
            op_name = str(send_instr)
        else:
            op_name = op.name + ("B" if not op.is_forward else "") + "_Comm"
        return self._get_duration_event(
            op_name,
            start_time,
            duration,
        )

    def get_memory_event(self, time, memory, event_type):
        return {
            "name": event_type,
            "ph": "C",
            "ts": time,
            "tid": 0,
            "pid": self.executor_id,
            "args": {
                event_type.lower(): memory,
            },
        }

    def update_memory(
        self, peak_time, peak_memory, stored_time, stored_memory
    ):
        peak_memory_during_op = self.current_memory + peak_memory
        # memory after op becomes current memory + stored memory
        self.current_memory += stored_memory
        self.peak_memory = max(self.peak_memory, peak_memory_during_op)
        device_memory_event_peak = self.get_memory_event(
            peak_time, peak_memory_during_op, "Memory"
        )
        device_peak_memory_event_peak = self.get_memory_event(
            peak_time, self.peak_memory, "Peak Memory"
        )
        device_memory_event_stored = self.get_memory_event(
            stored_time, self.current_memory, "Memory"
        )
        device_peak_memory_event_stored = self.get_memory_event(
            stored_time, self.peak_memory, "Peak Memory"
        )
        return [
            device_memory_event_peak,
            device_peak_memory_event_peak,
            device_memory_event_stored,
            device_peak_memory_event_stored,
        ]


class Scheduler:
    def __init__(
        self,
        minibatch_spec: SchedulerMinibatchSpec,
        include_memory_stats: bool = True,
        memory_limit: float = float("inf"),
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.minibatch_spec = minibatch_spec
        self.include_memory_stats = include_memory_stats
        self.memory_limit = memory_limit
        self.logger = logger

    def _get_executor(
        self,
        executor_id,
        thread_id,
        n_orig_layers,
        assigned_stages,
        is_comm_stage,
        include_memory_stats,
        memory_limit=float("inf"),
        separate_comm_stage=False,
        parent_executor=None,
    ):
        """
        Creates an executor, overridden in subclasses.
        Returns an instance of executor.
        """
        raise NotImplementedError

    def _initialize(self):
        self.minibatch_spec.initialize()
        self.n_orig_layers = self.minibatch_spec.n_orig_layers
        self.n_flattened_stages = self.minibatch_spec.n_flattened_stages
        self.executor2stages: Dict[ExecutorIndex, list[int]] = defaultdict(
            list
        )
        self.layer2chunkid = {}
        chunks_per_device = defaultdict(int)
        for flattened_stage_id, executor in enumerate(
            self.minibatch_spec.flattened_executor_assignment
        ):
            if not _is_comm_stage(flattened_stage_id, self.n_orig_layers):
                self.layer2chunkid[flattened_stage_id] = chunks_per_device[
                    executor
                ]
                chunks_per_device[executor] += 1
            else:
                self.layer2chunkid[flattened_stage_id] = self.layer2chunkid[
                    flattened_stage_id - 1
                ]
            self.executor2stages[executor].append(flattened_stage_id)

        # initiatiize executors
        self.executors: Dict[ExecutorIndex, ScheduleExecutor] = {}
        unique_executors_used = sorted(
            list(set(self.minibatch_spec.flattened_executor_assignment))
        )
        for executor in unique_executors_used:
            executor_stages = self.executor2stages[executor]
            stage_model_states = [
                self.minibatch_spec.flattened_model_states[stages]
                for stages in executor_stages
            ]
            stage_is_fw = [
                _is_fw_stage(stage_id, self.n_orig_layers)
                for stage_id in executor_stages
            ]
            self.executors[executor] = self._get_executor(
                executor_id=executor.executor_id,
                thread_id=executor.thread_id,
                n_orig_layers=self.n_orig_layers,
                assigned_stages=list(
                    zip(executor_stages, stage_model_states, stage_is_fw)
                ),
                is_comm_stage=executor.thread_id == ExecutorThread.COMM_THREAD,
                include_memory_stats=self.include_memory_stats,
                memory_limit=self.memory_limit,
                parent_executor=self.executors[
                    ExecutorIndex(
                        executor.executor_id, ExecutorThread.COMP_THREAD
                    )
                ]
                if executor.thread_id == ExecutorThread.COMM_THREAD
                else None,
            )
        self.makespan = None

    def _get_next_executor(self, flattened_stage_id):
        if flattened_stage_id == self.n_flattened_stages - 1:
            next_executor = None
        else:
            next_executor = self.minibatch_spec.flattened_executor_assignment[
                flattened_stage_id + 1
            ]
        if next_executor is not None:
            return self.executors[next_executor]
        return None

    def _get_op(self, flattened_stage_id: int, microbatch_id: int):
        tensor_shape = self.minibatch_spec.microbatches[
            microbatch_id
        ].flattened_activation_shapes[flattened_stage_id]
        if flattened_stage_id > 0:
            last_layer_tensor_shape = self.minibatch_spec.microbatches[
                microbatch_id
            ].flattened_activation_shapes[flattened_stage_id - 1]
        else:
            last_layer_tensor_shape = None
        loads_data = tuple()
        if last_layer_tensor_shape and len(last_layer_tensor_shape) < len(
            tensor_shape
        ):
            # encoder-decoder boundary, needs load data
            loads_data = tensor_shape[len(last_layer_tensor_shape) :]
        elif flattened_stage_id == 0:
            # first layer, needs load data
            loads_data = tensor_shape

        return ScheduleOperation(
            name=self.minibatch_spec.microbatches[microbatch_id].name,
            microbatch=microbatch_id,
            flattened_stage=flattened_stage_id,
            exec_time=self.minibatch_spec.microbatches[
                microbatch_id
            ].flattened_exec_times[flattened_stage_id],
            is_forward=_is_fw_stage(flattened_stage_id, self.n_orig_layers),
            stored_memory=self.minibatch_spec.microbatches[
                microbatch_id
            ].flattened_stored_activation_sizes[flattened_stage_id],
            peak_memory=self.minibatch_spec.microbatches[
                microbatch_id
            ].flattened_peak_activation_sizes[flattened_stage_id],
            is_first_bw_layer=_is_first_bw_layer(
                flattened_stage_id, self.n_orig_layers
            ),
            tensor_shape=tensor_shape,
            loads_data=tuple(loads_data),
            next_executor=self._get_next_executor(flattened_stage_id),
        )

    def _get_metadata(self):
        metadata = []
        for executor in self.executors.values():
            metadata += executor.get_metadata()
        return metadata

    def get_executor_peak_memory(self) -> Dict[str, float]:
        result = {}
        for executor in self.executors.values():
            result[executor.full_name()] = executor.peak_memory
        return result

    def get_makespan(self) -> float:
        return self.makespan

    def _get_trace_events(self) -> Dict[str, Any]:
        trace_events = {
            "traceEvents": self._get_metadata(),
            "displayTimeUnit": "ms",
        }
        return trace_events

    def schedule(self, **kwargs):
        raise NotImplementedError

    def get_instructions(self):
        instructions = [
            executor.traced_instructions
            for executor in self.executors.values()
            if executor.parent_executor is None
        ]
        return instructions
