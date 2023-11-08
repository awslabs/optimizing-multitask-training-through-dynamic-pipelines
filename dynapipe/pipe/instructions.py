# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Modified from DeepSpeed:
# deepspeed/runtime/pipe/schedule.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union


class RecomputeMethod:
    NONE = 0
    FULL = 1
    SELECTIVE = 2


_RECOMPUTE_METHOD_NAMES = {
    RecomputeMethod.NONE: "none",
    RecomputeMethod.FULL: "full",
    RecomputeMethod.SELECTIVE: "selective",
}

_NAME_TO_RECOMPUTE_METHOD = {
    "none": RecomputeMethod.NONE,
    "full": RecomputeMethod.FULL,
    "selective": RecomputeMethod.SELECTIVE,
}


def get_available_rc_types() -> List[str]:
    return list(_NAME_TO_RECOMPUTE_METHOD.keys())


def name_to_recompute_method(name: str) -> RecomputeMethod:
    if name in _NAME_TO_RECOMPUTE_METHOD:
        return _NAME_TO_RECOMPUTE_METHOD[name]
    raise ValueError(f"Unknown recompute method: {name}")


@dataclass
class SerializationConfig:
    BYTES_ENDIANNESS = "little"
    # length of the serialized fields in bytes
    # increase these the number to represent exceeds maximum
    INSTRUCTION_INDEX_BYTES = 1
    EXECUTION_PLAN_META_BYTES = 4
    SERIALIZED_SIZE_BYTES = 4
    TENSOR_META_BYTES = 1
    TENSOR_SHAPE_INDEX_BYTES = 32


class PipeInstruction:
    """Base class for all instructions to be executed by the pipeline engine.
    All keyword arguments are stored as members similar to a ``namedtuple``.
    These are then accessible to the PipeEngine during execution.
    Args:
        kwargs (optional): keyword arguments to store as members
    """

    # used to generate a unique index for each instruction class
    # for serialization
    _instr_index_to_cls: Dict[int, Type["PipeInstruction"]] = {}

    def __init__(self, microbatch, stage, **kwargs):
        self.name = self.__class__.__name__
        self.microbatch = microbatch
        self.stage = stage
        self.kwargs = kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __repr__(self):
        name = f"{self.name}(microbatch={self.microbatch}, stage={self.stage}"
        if self.kwargs:
            name += ", "
            name += ", ".join(
                f"{key}={repr(arg)}" for key, arg in self.kwargs.items()
            )
        name += ")"
        return name

    def __init_subclass__(cls) -> None:
        cls._instr_index = len(PipeInstruction._instr_index_to_cls)
        PipeInstruction._instr_index_to_cls[cls._instr_index] = cls

    def serialize(self, config=SerializationConfig()) -> bytes:
        """Serialize the instruction to a byte array."""
        return (
            self._instr_index.to_bytes(
                config.INSTRUCTION_INDEX_BYTES, config.BYTES_ENDIANNESS
            )
            + self.microbatch.to_bytes(
                config.EXECUTION_PLAN_META_BYTES, config.BYTES_ENDIANNESS
            )
            + self.stage.to_bytes(
                config.EXECUTION_PLAN_META_BYTES, config.BYTES_ENDIANNESS
            )
        )

    def _deserialize(
        bytes: bytes, config=SerializationConfig()
    ) -> Tuple[Dict[str, Any], bytes]:
        return {}, bytes

    @classmethod
    def deserialize(
        cls, bytes: bytes, config=SerializationConfig()
    ) -> Tuple["PipeInstruction", bytes]:
        """Deserialize the instruction from a byte array."""
        instr_index = int.from_bytes(
            bytes[: config.INSTRUCTION_INDEX_BYTES], config.BYTES_ENDIANNESS
        )
        bytes = bytes[config.INSTRUCTION_INDEX_BYTES :]
        microbatch = int.from_bytes(
            bytes[: config.EXECUTION_PLAN_META_BYTES], config.BYTES_ENDIANNESS
        )
        bytes = bytes[config.EXECUTION_PLAN_META_BYTES :]
        stage = int.from_bytes(
            bytes[: config.EXECUTION_PLAN_META_BYTES], config.BYTES_ENDIANNESS
        )
        bytes = bytes[config.EXECUTION_PLAN_META_BYTES :]
        kwargs, bytes = cls._instr_index_to_cls[instr_index]._deserialize(
            bytes, config=config
        )
        return (
            cls._instr_index_to_cls[instr_index](
                microbatch=microbatch, stage=stage, **kwargs
            ),
            bytes,
        )

    def __eq__(self, other: "PipeInstruction"):
        return (
            self.__class__ == other.__class__
            and self.microbatch == other.microbatch
            and self.stage == other.stage
            and self.kwargs == other.kwargs
        )


class BufferInstruction(PipeInstruction):
    """Base class for all instructions that access a pipeline buffer.
    Args:
        buffer_id (int): The index of the buffer to access.
    """

    def __init__(
        self,
        microbatch,
        stage,
        buffer_ids: Optional[List[int]] = None,
        buffer_shapes: List[Tuple[int, ...]] = None,
        **kwargs,
    ):
        # we cannot directly use a list as a default argument
        if buffer_ids is None:
            buffer_ids = []
        if buffer_shapes is None:
            buffer_shapes = []
        super().__init__(
            microbatch,
            stage,
            buffer_shapes=buffer_shapes,
            buffer_ids=buffer_ids,
            **kwargs,
        )
        self.buffer_ids: List[int]
        self.buffer_shapes: List[Tuple[int, ...]]

    def serialize(self, config=SerializationConfig()) -> bytes:
        serialized_buffer_ids = [
            buffer_id.to_bytes(
                config.EXECUTION_PLAN_META_BYTES, config.BYTES_ENDIANNESS
            )
            for buffer_id in self.buffer_ids
        ]
        return (
            super().serialize(config=config)
            + len(self.buffer_ids).to_bytes(
                config.EXECUTION_PLAN_META_BYTES, config.BYTES_ENDIANNESS
            )
            + b"".join(serialized_buffer_ids)
            + len(self.buffer_shapes).to_bytes(
                config.EXECUTION_PLAN_META_BYTES, config.BYTES_ENDIANNESS
            )
            + b"".join(
                [
                    serialize_tensor_shape(shape, config)
                    for shape in self.buffer_shapes
                ]
            )
        )

    @classmethod
    def _deserialize(
        cls, bytes: bytes, config=SerializationConfig()
    ) -> Tuple[Dict[str, Any], bytes]:
        """Partially deserialize the instruction from a byte array.
        Returns:
            buffer_ids: The list of buffer ids
            bytes: The remaining bytes after deserializing the buffer ids
        """
        kwargs, bytes = super()._deserialize(bytes, config=config)

        n_buffer_ids = int.from_bytes(
            bytes[: config.EXECUTION_PLAN_META_BYTES],
            config.BYTES_ENDIANNESS,
        )
        bytes = bytes[config.EXECUTION_PLAN_META_BYTES :]
        buffer_ids, bytes = deserialize_list_of_ints(
            bytes,
            n_buffer_ids,
            config.EXECUTION_PLAN_META_BYTES,
            config.BYTES_ENDIANNESS,
        )
        n_buffer_shapes = int.from_bytes(
            bytes[: config.EXECUTION_PLAN_META_BYTES],
            config.BYTES_ENDIANNESS,
        )
        bytes = bytes[config.EXECUTION_PLAN_META_BYTES :]
        buffer_shapes = []
        for _ in range(n_buffer_shapes):
            shape, bytes = deserialize_tensor_shape(
                bytes,
                config,
            )
            buffer_shapes.append(shape)
        kwargs.update(
            {"buffer_shapes": buffer_shapes, "buffer_ids": buffer_ids}
        )
        return kwargs, bytes


class CommunicationInstruction(BufferInstruction):
    def __init__(
        self,
        microbatch,
        stage,
        peer: int,
        buffer_ids: Optional[List[int]] = None,
        buffer_shapes: List[Tuple[int, ...]] = None,
        **kwargs,
    ):
        super().__init__(
            microbatch,
            stage,
            buffer_ids=buffer_ids,
            buffer_shapes=buffer_shapes,
            peer=peer,
            **kwargs,
        )
        self.peer: int

    def serialize(self, config=SerializationConfig()) -> bytes:
        return super().serialize(config=config) + self.peer.to_bytes(
            config.EXECUTION_PLAN_META_BYTES, config.BYTES_ENDIANNESS
        )

    @classmethod
    def _deserialize(
        cls, bytes, config=SerializationConfig()
    ) -> Tuple[Dict[str, Any], bytes]:
        kwargs, bytes = super()._deserialize(bytes, config=config)
        peer = int.from_bytes(
            bytes[: config.EXECUTION_PLAN_META_BYTES], config.BYTES_ENDIANNESS
        )
        bytes = bytes[config.EXECUTION_PLAN_META_BYTES :]
        kwargs.update({"peer": peer})
        return kwargs, bytes


class LoadInput(BufferInstruction):
    pass


class FreeBuffer(BufferInstruction):
    def __init__(
        self,
        buffer_ids: List[int],
        buffer_shapes: List[Tuple[int, ...]] = None,
        microbatch=0,
        stage=0,
    ):
        super().__init__(
            microbatch=microbatch,
            stage=stage,
            buffer_ids=buffer_ids,
            buffer_shapes=buffer_shapes,
        )


class CommunicationStartInstruction(CommunicationInstruction):
    """Base class for all instructions that initiates communication between
       pipeline stages.
       Note: All communication instructions are non-blocking.
             Correct scheduling is required to avoid deadlock.
    Args:
        buffer_ids: The indices of the buffer to access.
        buffer_shapes: The shape of the tensors to send.
    """

    pass


class CommunicationFinishInsturction(CommunicationInstruction):
    """Base class for all instructions that waits for communication to finish.
       Note: This instruction is blocking.
    Args:
        buffer_shapes: The shape of the tensors to send.
    """

    pass


class SendActivationStart(CommunicationStartInstruction):
    """Start to send activations to the next stage."""

    pass


class SendActivationFinish(CommunicationFinishInsturction):
    """Wait for sending activations to finish."""

    pass


class RecvActivationStart(CommunicationStartInstruction):
    """Start to receive activations from the previous stage."""

    pass


class RecvActivationFinish(CommunicationFinishInsturction):
    """Wait for receiving activations to finish."""

    pass


class SendGradStart(CommunicationStartInstruction):
    """Start to send gradients to the previous stage."""

    pass


class SendGradFinish(CommunicationFinishInsturction):
    """Wait for sending gradients to finish."""

    pass


class RecvGradStart(CommunicationStartInstruction):
    """Start to receive gradients from the next stage."""

    pass


class RecvGradFinish(CommunicationFinishInsturction):
    """Wait for receiving gradients to finish."""

    pass


class ForwardPass(BufferInstruction):
    """Execute the forward pass."""

    pass


class BackwardPass(BufferInstruction):
    """Execute the backward pass."""

    def __init__(
        self,
        microbatch,
        stage,
        buffer_ids: Optional[List[int]] = None,
        buffer_shapes: List[Tuple[int, ...]] = None,
        first_bw_layer: bool = False,
        **kwargs,
    ):
        super().__init__(
            microbatch,
            stage,
            buffer_ids,
            buffer_shapes,
            first_bw_layer=first_bw_layer,
            **kwargs,
        )
        self.first_bw_layer: bool

    def serialize(self, config=SerializationConfig()) -> bytes:
        return super().serialize(config=config) + int(
            self.first_bw_layer
        ).to_bytes(config.EXECUTION_PLAN_META_BYTES, config.BYTES_ENDIANNESS)

    @classmethod
    def _deserialize(
        cls, bytes, config=SerializationConfig()
    ) -> Tuple[Dict[str, Any], bytes]:
        kwargs, bytes = super()._deserialize(bytes, config=config)
        first_bw_layer = int.from_bytes(
            bytes[: config.EXECUTION_PLAN_META_BYTES], config.BYTES_ENDIANNESS
        )
        bytes = bytes[config.EXECUTION_PLAN_META_BYTES :]
        kwargs.update({"first_bw_layer": bool(first_bw_layer)})
        return kwargs, bytes


class ExecutionPlan:
    """
    Sequences of PipeInstructions to be executed by the PipeEngine, which
    defines the buffer allocation, the shape of the tensors and the pipeline
    schedule.

    The sequences of instructions must be executed in the exact order they are
    defined in the plan. No synchronization should be performed between
    instructions to avoid deadlock.

    Args:
        stages (int): The number of pipeline stages.
        stage_id (int): The stage that will execute the generated schedule.
    """

    def __init__(
        self,
        instructions: List[PipeInstruction],
        micro_batches: int,
        nranks: int,
        nstages: int,
        rank: int,
        assigned_stages: List[int],
        recompute_method: RecomputeMethod = RecomputeMethod.NONE,
        num_pipe_buffers: Optional[int] = 0,
    ):
        self.instructions = instructions
        self.micro_batches = micro_batches
        self.nranks = nranks
        self.nstages = nstages
        self.rank = rank
        self.assigned_stages = assigned_stages
        self.recompute_method = recompute_method
        self._valid_rank(rank)
        self.num_pipe_buffers = num_pipe_buffers

    def _valid_rank(self, rank):
        return 0 <= rank < self.nranks

    @property
    def num_micro_batches(self):
        """The number of total micro_batches in this schedule."""
        return self.micro_batches

    def __iter__(self):
        self.it = None
        return self

    def __next__(self):
        if self.it is None:
            self.it = self.steps()
        return next(self.it)

    def __repr__(self) -> str:
        return (
            "ExecutionPlan(micro_batches={}, nranks={}, nstages={}, rank={}, "
            "assigned_stages={}, recompute_method={}, "
            "num_pipe_buffers={}, instructions={})".format(
                self.micro_batches,
                self.nranks,
                self.nstages,
                self.rank,
                self.assigned_stages,
                _RECOMPUTE_METHOD_NAMES[self.recompute_method],
                self.num_pipe_buffers,
                self.instructions,
            )
        )

    def __str__(self):
        """Print the execution plan in a human readable format."""
        return (
            "ExecutionPlan(micro_batches={}, nranks={}, nstages={}, rank={}, "
            "assigned_stages={}, recompute_method={}, "
            "num_pipe_buffers={}, instructions=[\n\t".format(
                self.micro_batches,
                self.nranks,
                self.nstages,
                self.rank,
                self.assigned_stages,
                _RECOMPUTE_METHOD_NAMES[self.recompute_method],
                self.num_pipe_buffers,
            )
            + "\n\t".join([str(x) for x in self.instructions])
            + "\n])"
        )

    def __eq__(self, other: "ExecutionPlan"):
        if not isinstance(other, ExecutionPlan):
            return False
        return (
            self.micro_batches == other.micro_batches
            and self.nranks == other.nranks
            and self.nstages == other.nstages
            and self.rank == other.rank
            and self.assigned_stages == other.assigned_stages
            and self.recompute_method == other.recompute_method
            and self.num_pipe_buffers == other.num_pipe_buffers
            and self.instructions == other.instructions
        )

    def serialize(self, config=SerializationConfig()) -> bytes:
        """Serialize the execution plan to a byte array."""

        def _serialize_plan_meta(x: int):
            return x.to_bytes(
                config.EXECUTION_PLAN_META_BYTES, config.BYTES_ENDIANNESS
            )

        return (
            _serialize_plan_meta(self.micro_batches)
            + _serialize_plan_meta(self.nranks)
            + _serialize_plan_meta(self.nstages)
            + _serialize_plan_meta(self.rank)
            + _serialize_plan_meta(len(self.assigned_stages))
            + b"".join([_serialize_plan_meta(x) for x in self.assigned_stages])
            + _serialize_plan_meta(self.recompute_method)
            + _serialize_plan_meta(self.num_pipe_buffers)
            + len(self.instructions).to_bytes(
                config.EXECUTION_PLAN_META_BYTES, config.BYTES_ENDIANNESS
            )
            + b"".join(
                [instr.serialize(config) for instr in self.instructions]
            )
        )

    @classmethod
    def deserialize(
        cls, bytes, config=SerializationConfig()
    ) -> "ExecutionPlan":
        """Deserialize the execution plan from a byte array."""

        def _deserialize_plan_meta(bytes):
            return (
                int.from_bytes(
                    bytes[: config.EXECUTION_PLAN_META_BYTES],
                    config.BYTES_ENDIANNESS,
                ),
                bytes[config.EXECUTION_PLAN_META_BYTES :],
            )

        micro_batches, bytes = _deserialize_plan_meta(bytes)
        nranks, bytes = _deserialize_plan_meta(bytes)
        nstages, bytes = _deserialize_plan_meta(bytes)
        rank, bytes = _deserialize_plan_meta(bytes)
        n_assigned_stages, bytes = _deserialize_plan_meta(bytes)
        assigned_stages = []
        for _ in range(n_assigned_stages):
            assigned_stage, bytes = _deserialize_plan_meta(bytes)
            assigned_stages.append(assigned_stage)
        recompute_method, bytes = _deserialize_plan_meta(bytes)
        num_pipe_buffers, bytes = _deserialize_plan_meta(bytes)
        n_instructions = int.from_bytes(
            bytes[: config.EXECUTION_PLAN_META_BYTES], config.BYTES_ENDIANNESS
        )
        bytes = bytes[config.EXECUTION_PLAN_META_BYTES :]
        instructions = []
        for _ in range(n_instructions):
            instr, bytes = PipeInstruction.deserialize(bytes, config=config)
            instructions.append(instr)
        assert len(bytes) == 0
        return cls(
            instructions,
            micro_batches,
            nranks,
            nstages,
            rank,
            assigned_stages,
            recompute_method,
            num_pipe_buffers,
        )


def serialize_tensor_shape(
    tensor_shape: Tuple[int, ...], config=SerializationConfig()
):
    """Serialize the tensor shape to a byte array."""
    if not isinstance(tensor_shape, (tuple, list)):
        raise TypeError(
            "Expected a tuple or list for tensor_shape. "
            "Got {}.".format(tensor_shape)
        )
    tensor_dim = len(tensor_shape).to_bytes(
        config.TENSOR_META_BYTES, config.BYTES_ENDIANNESS
    )
    return tensor_dim + b"".join(
        [
            dim.to_bytes(
                config.TENSOR_SHAPE_INDEX_BYTES, config.BYTES_ENDIANNESS
            )
            for dim in tensor_shape
        ]
    )


def deserialize_list_of_ints(bytes, n_ints, n_bytes_per_int, endianess):
    """Deserialize a list of ints from a byte array."""
    return [
        int.from_bytes(
            bytes[i * n_bytes_per_int : (i + 1) * n_bytes_per_int],
            endianess,
        )
        for i in range(n_ints)
    ], bytes[n_ints * n_bytes_per_int :]


def deserialize_tensor_shape(
    bytes: bytes, config=SerializationConfig()
) -> Tuple[Tuple[int, ...], bytes]:
    """Deserialize the tensor shape from a byte array."""
    tensor_dim = int.from_bytes(
        bytes[: config.TENSOR_META_BYTES],
        config.BYTES_ENDIANNESS,
    )
    bytes = bytes[config.TENSOR_META_BYTES :]
    tensor_shape, bytes = deserialize_list_of_ints(
        bytes,
        tensor_dim,
        config.TENSOR_SHAPE_INDEX_BYTES,
        config.BYTES_ENDIANNESS,
    )
    return tuple(tensor_shape), bytes


def serialize_list_of_eps(
    eps: List[ExecutionPlan], config=SerializationConfig()
) -> bytes:
    """Serialize a list of execution plans to a byte array."""
    result = len(eps).to_bytes(
        config.EXECUTION_PLAN_META_BYTES, config.BYTES_ENDIANNESS
    )
    for ep in eps:
        ep_bytes = ep.serialize(config)
        ep_bytes_len = len(ep_bytes).to_bytes(
            config.SERIALIZED_SIZE_BYTES, config.BYTES_ENDIANNESS
        )
        result += ep_bytes_len + ep_bytes

    return result


def deserialize_list_of_eps(
    bytes: bytes, config=SerializationConfig(), deserialize_inner=True
) -> Tuple[List[Union[ExecutionPlan, bytes]]]:
    """Deserialize a list of execution plans from a byte array."""
    n_eps = int.from_bytes(
        bytes[: config.EXECUTION_PLAN_META_BYTES],
        config.BYTES_ENDIANNESS,
    )
    bytes = bytes[config.EXECUTION_PLAN_META_BYTES :]
    eps = []
    for _ in range(n_eps):
        ep_bytes_len = int.from_bytes(
            bytes[: config.SERIALIZED_SIZE_BYTES],
            config.BYTES_ENDIANNESS,
        )
        bytes = bytes[config.SERIALIZED_SIZE_BYTES :]
        ep_bytes = bytes[:ep_bytes_len]
        if deserialize_inner:
            ep = ExecutionPlan.deserialize(ep_bytes, config=config)
            eps.append(ep)
        else:
            eps.append(ep_bytes)
        bytes = bytes[ep_bytes_len:]
    assert len(bytes) == 0
    return eps
