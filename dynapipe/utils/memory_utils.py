# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Tuple, Union

from dynapipe.model import TransformerModelSpec

MAX_POSSIBLE_MICROBATCH_SIZE = 2**31 - 1


# utility functions to calculate transformer memory consumption
def get_transformer_output_memory(
    sequence_length, batch_size, hidden_dim, bytes_per_element
):
    # size is in MB (megabytes)
    return sequence_length * batch_size * hidden_dim * bytes_per_element / 1e6


def get_transformer_activation(
    sequence_length,
    batch_size,
    hidden_dim,
    num_attn_heads,
    mlp_hidden_dim,
    bytes_per_element,
    tp_size,
    is_decoder=False,
):
    # Estimates the activation memory needed for a transformer layer.
    # Size is in MB (megabytes)
    # Formula from Korthikanti et.al,
    # "Reducing Activation Recomputation in Large Transformer Models"
    # https://arxiv.org/abs/2205.05198
    result = 0
    sbh = sequence_length * batch_size * hidden_dim
    sbh_mlp = sequence_length * batch_size * mlp_hidden_dim
    as2b = num_attn_heads * sequence_length * sequence_length * batch_size
    attention = 0
    # QKV
    attention_input = sbh * bytes_per_element
    # QK^T
    attention += 2 * sbh * bytes_per_element
    # softmax
    attention += as2b * bytes_per_element
    # softmax dropout (one byte per element)
    attention += as2b
    # attn over values
    attention += (as2b + sbh) * bytes_per_element
    attention /= tp_size
    # attention input is not parallelised in tensor parallelism
    attention += attention_input

    result += attention
    # MLP (assume MLP hidden dim is same as hidden dim)
    # MLP input is not parallelised in tensor parallelism,
    # i.e., sbh * bytes_per_element
    # other parts are parallelised
    result += (
        sbh * bytes_per_element
        + (2 * sbh_mlp * bytes_per_element + sbh) / tp_size
    )
    # layernorm
    result += 2 * sbh * bytes_per_element
    if is_decoder:
        # cross-attention
        result += attention
        # encoder input
        result += sbh * bytes_per_element
    return result / 1e6


def get_transformer_model_state(
    hidden_dim,
    num_attn_heads,
    kv_channels,
    mlp_hidden_dim,
    bytes_per_element,
    optimizer_state_multiplier,
    tp_size,
    is_decoder=False,
):
    # Estimate transformer model state usage, assuming Adam optimizer
    # and fp16 training is used. Size is in MB (megabytes)
    # Note: optimizer state multiplier should already consider
    # the number bytes needed to store each element.
    # bytes_per_element is only used to calculate the size of
    # the model parameters and gradients.
    # Optimizer state multiplier is 12 for FP16 mixed precision Adam.
    # Reference: Rajbhandari et.al.,
    # "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
    # https://arxiv.org/abs/1910.02054
    n_params = 0
    attention = 0
    # layer norm x2
    n_params += 2 * 2 * hidden_dim
    # QKV
    attention += 3 * (hidden_dim * kv_channels + kv_channels)
    # projection
    attention += num_attn_heads * kv_channels * hidden_dim + hidden_dim
    attention /= tp_size
    n_params += attention
    # MLP
    n_params += (
        2 * hidden_dim * mlp_hidden_dim + hidden_dim + mlp_hidden_dim
    ) / tp_size
    if is_decoder:
        # cross-attention
        n_params += attention
        # layer norm
        n_params += 2 * hidden_dim
    # scale to model state
    result = n_params * (
        bytes_per_element + bytes_per_element + optimizer_state_multiplier
    )
    return result / 1e6


@dataclass
class TransformerMemoryModel(object):
    # See comments for get_transformer_activation and
    # get_transformer_model_state for details
    model_spec: TransformerModelSpec

    def get_output_memory(self, batch_size, sequence_length):
        return get_transformer_output_memory(
            sequence_length,
            batch_size,
            self.model_spec.hidden_dim,
            bytes_per_element=self.model_spec.bytes_per_element,
        )

    def get_activation_memory(
        self, batch_size, sequence_length, is_decoder=False
    ):
        return get_transformer_activation(
            sequence_length,
            batch_size,
            self.model_spec.hidden_dim,
            self.model_spec.num_attn_heads,
            mlp_hidden_dim=self.model_spec.mlp_hidden_dim,
            bytes_per_element=self.model_spec.bytes_per_element,
            tp_size=self.model_spec.tp_size,
            is_decoder=is_decoder,
        )

    def get_model_state_memory(self, is_decoder=False):
        return get_transformer_model_state(
            self.model_spec.hidden_dim,
            self.model_spec.num_attn_heads,
            kv_channels=self.model_spec.kv_channels,
            mlp_hidden_dim=self.model_spec.mlp_hidden_dim,
            bytes_per_element=self.model_spec.bytes_per_element,
            optimizer_state_multiplier=self.model_spec.optimizer_state_multiplier,  # noqa
            tp_size=self.model_spec.tp_size,
            is_decoder=is_decoder,
        )


@dataclass
class InvTransformerMemoryModel:
    n_encoders: int
    n_decoders: int
    model_spec: TransformerModelSpec
    _mem_model: Union[TransformerMemoryModel, None] = None

    def __post_init__(self):
        if self._mem_model is None:
            self._mem_model = TransformerMemoryModel(
                model_spec=self.model_spec,
            )

    def _get_memory(self, mbs: int, seq_len: int) -> float:
        return self.n_encoders * self._mem_model.get_activation_memory(
            mbs, seq_len
        ) + self.n_decoders * self._mem_model.get_activation_memory(
            mbs, seq_len, is_decoder=True
        )

    def set_max_memory(self, max_memory: float) -> None:
        self._ref_memory = max_memory

    def set_reference(self, mbs, seq_len) -> None:
        self._ref_memory = self._get_memory(mbs, seq_len)

    def _get_mbs_within_range(
        self, seq_len: int, mbs_range: Tuple[int, int]
    ) -> int:
        if mbs_range[1] >= mbs_range[0]:
            midpoint = (mbs_range[0] + mbs_range[1]) // 2
            mid_memory = self._get_memory(midpoint, seq_len)
            mid_plus_one_memory = self._get_memory(midpoint + 1, seq_len)
            if (
                mid_memory <= self._ref_memory
                and mid_plus_one_memory >= self._ref_memory
            ):
                return midpoint
            elif mid_memory <= self._ref_memory:
                return self._get_mbs_within_range(
                    seq_len, (midpoint + 1, mbs_range[1])
                )
            else:
                return self._get_mbs_within_range(
                    seq_len, (mbs_range[0], midpoint - 1)
                )
        else:
            return -1

    def get_microbatch_size(self, sequence_length):
        assert hasattr(
            self, "_ref_memory"
        ), "Must set memory reference or max memory first."
        return self._get_mbs_within_range(
            sequence_length, (1, MAX_POSSIBLE_MICROBATCH_SIZE)
        )
