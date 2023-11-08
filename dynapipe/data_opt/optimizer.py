# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import os

import elkai
import numpy as np
import prtpy
from sklearn.cluster import AgglomerativeClustering

from dynapipe.data_opt.cost_models import ProfileBasedCostModelWithRC
from dynapipe.model import TransformerModelSpec
from dynapipe.utils.logger import logger

__makefile_path = os.path.join(os.path.dirname(__file__), "Makefile")
if os.path.exists(__makefile_path):
    # not installed from wheel, try to compile the C++ extension
    import subprocess
    import sys
    import time

    start_time = time.time()
    logger.info(">>> compiling dp C++ extension ... <<<")
    path = os.path.abspath(os.path.dirname(__file__))
    ret = subprocess.run(["make", "-C", path])
    if ret.returncode != 0:
        logger.info("Making C++ DP extension module failed, exiting.")
        sys.exit(1)
    logger.info(
        ">>> Done with building C++ DP extension. Compilation time: {:.3f} "
        "seconds <<<".format(time.time() - start_time)
    )
    from dynapipe.data_opt.dp_helper import (  # type: ignore # noqa
        cpp_consecutive_partition_dp,
    )
else:
    try:
        from dynapipe.data_opt.dp_helper import (  # type: ignore # noqa
            cpp_consecutive_partition_dp,
        )
    except ImportError:
        logger.info(
            ">>> C++ DP extension not found and cannot be built. "
            "(missing Makefile). Please reinstall the package or download "
            "the source code and build the C++ extension manually."
        )
        raise


class DataAssignmentOptimizer(object):
    """Data assignment optimizer.

    Optimizes the assignment of a mini-batch of data into micro-batches.
    """

    def __init__(
        self,
        cost_model: ProfileBasedCostModelWithRC,
        model_spec: TransformerModelSpec,
        n_executors: int,
        n_layers_per_stage: int,
        n_chunks_per_device: int = 1,
        dp_size: int = 1,
        tp_size: int = 1,
        zero_stage: int = 0,
        device_memory_limit: float = float("inf"),
        round_seqlen_multiple=8,
        per_mb_memory_fraction=None,
        len_pack_sep_tokens=1,
        len_decoder_additional_tokens=2,
        seqlen_offset=0,
    ):
        """Optimizer for assigning data samples into micro-batches.
        cost_model: cost model for the model used
        model_spec: model specification
        n_executors: number of stages of the pipelined model
        n_layers_per_stage: number of layers per each pipeline stage
        n_chunks_per_device: number of chunks per device
                             (> 1 indicating interleaved schedule)
        dp_size: data parallelism degree
        tp_size: tensor parallelism degree
        zero_stage: stage of ZeRO optimizer
        device_memory_limit: memory limit in MB (MegaBytes)
        round_seqlen_multiple: always round sequence length to multiple of
                               this number, required for some kernels
                               default: 8
        len_pack_sep_tokens: number of tokens used to separate samples in the
                             packed sequence, only used when enable_packing
                             is True during optimization.
        len_decoder_additional_tokens: number of additional tokens added to
                                        the decoder sequence length other than
                                        the target sequence, e.g. <bos>, <eos>
        seqlen_offset: should be set 1 for decoder only models, whose input
                       and target sequences are data sequence length - 1
                       0 for encoder-decoder models.
        """
        self.cost_model = cost_model
        self.n_executors = n_executors
        self.n_layers_per_stage = n_layers_per_stage
        # create memory model
        self.model_spec = model_spec
        self.memory_limit = device_memory_limit
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.zero_stage = zero_stage
        self.round_seqlen_multiple = round_seqlen_multiple
        self.len_pack_sep_tokens = len_pack_sep_tokens
        self.len_decoder_additional_tokens = len_decoder_additional_tokens
        self.n_chunks_per_device = n_chunks_per_device
        self.per_mb_memory_fraction = per_mb_memory_fraction
        self.seqlen_offset = seqlen_offset

    def _round_seqlen(self, seqlen, decoder=False):
        if decoder:
            seqlen += self.len_decoder_additional_tokens
        seqlen -= self.seqlen_offset
        return (
            (seqlen + self.round_seqlen_multiple - 1)
            // self.round_seqlen_multiple
            * self.round_seqlen_multiple
            + self.seqlen_offset
        )

    def _solve_sample_order_tsp_problem(
        self,
        sample_sequence_lengths,
        decoder_sample_sequence_lengths,
        bottleneck_tsp=True,
        dist_function="sum",
        use_clustering=True,
        distance_threshold=16,
    ):
        """Solve the TSP problem to determine the sample order."""
        if dist_function == "sum":

            def _f_dist(x, y):
                return abs(int(x[0]) - int(y[0])) + abs(int(x[1]) - int(y[1]))

        elif dist_function == "max":

            def _f_dist(x, y):
                return max(
                    abs(int(x[0]) - int(y[0])), abs(int(x[1]) - int(y[1]))
                )

        elif dist_function == "square":

            def _f_dist(x, y):
                return (int(x[0]) - int(y[0])) ** 2 + (
                    int(x[1]) - int(y[1])
                ) ** 2

        else:
            raise ValueError(
                "Unknown distance function: {}".format(dist_function)
            )

        def _get_distance_matrix(points):
            # add a dummy point at the beginning
            # to transform it into an open TSP problem
            distance_matrix = [[0] * (len(points) + 1)]
            for x in points:
                row = [0]
                for y in points:
                    row.append(_f_dist(x, y))
                distance_matrix.append(row)
            return distance_matrix

        input_points = list(
            zip(sample_sequence_lengths, decoder_sample_sequence_lengths)
        )
        if use_clustering:
            vectors_np = np.array(input_points)
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                linkage="complete",
            ).fit(vectors_np)
            labels = clustering.labels_
            n_clusters = max(labels) + 1
            cluster_to_samples = [[] for _ in range(n_clusters)]
            cluster_to_data = [[] for _ in range(n_clusters)]
            for sample_idx, label in enumerate(labels):
                cluster_to_samples[label].append(sample_idx)
                cluster_to_data[label].append(input_points[sample_idx])
            # compute cluster centroids
            cluster_to_center = [None] * n_clusters
            for cluster_label, data in enumerate(cluster_to_data):
                cluster_to_center[cluster_label] = tuple(np.mean(data, axis=0))
            # compute tsp for cluster centroids
            distance_matrix = np.array(_get_distance_matrix(cluster_to_center))
            permutation = list(
                np.array(
                    elkai.solve_int_matrix(
                        distance_matrix, 1, bottleneck=bottleneck_tsp
                    )
                )
                - 1
            )[1:]
            # reconstruct orig order
            result = []
            for cluster_label in permutation:
                result += cluster_to_samples[cluster_label]
            # sanity check result is a valid permutation
            assert sorted(result) == list(range(len(result)))
            return result

        distance_matrix = np.array(_get_distance_matrix(input_points))
        permutation = list(
            np.array(
                elkai.solve_int_matrix(
                    distance_matrix, 1, bottleneck=bottleneck_tsp
                )
            )
            - 1
        )[1:]
        return permutation

    def _pack(
        self,
        sequence: list,
        current_enc_length,
        current_dec_length,
        target_enc_length,
        target_dec_length,
        next_idx,
        samples_with_ids,
        consumed,
    ):
        for j in range(next_idx, len(samples_with_ids)):
            if consumed[j]:
                continue
            (
                seqlen_to_pack,
                dec_seqlen_to_pack,
                sample_id_to_pack,
            ) = samples_with_ids[j]
            if (
                current_enc_length + seqlen_to_pack <= target_enc_length
                and current_dec_length + dec_seqlen_to_pack
                <= target_dec_length
            ):
                sequence.append(sample_id_to_pack)
                current_enc_length += seqlen_to_pack
                current_dec_length += dec_seqlen_to_pack
                consumed[j] = True
        return current_enc_length, current_dec_length

    def _uniform_partition(self, samples_with_ids, microbatch_size):
        max_sequence_length = max([x[0] for x in samples_with_ids])
        max_decoder_sequence_length = max([x[1] for x in samples_with_ids])

        # round sequence length to multiple of round_seqlen_multiple
        max_sequence_length = self._round_seqlen(max_sequence_length)
        max_decoder_sequence_length = self._round_seqlen(
            max_decoder_sequence_length, decoder=True
        )
        # pack all sequences into fixed sequence length
        target_src_seqlen = max_sequence_length
        target_tgt_seqlen = (
            max_decoder_sequence_length - self.len_decoder_additional_tokens
        )
        consumed = [False] * len(samples_with_ids)
        sequences = []
        for seqlen, dec_seqlen, idx in samples_with_ids:
            if consumed[idx]:
                continue
            curr_sequence = []
            curr_sequence_seqlen = seqlen
            curr_sequence_dec_seqlen = dec_seqlen
            curr_sequence.append(idx)
            curr_sequence_seqlen, curr_sequence_dec_seqlen = self._pack(
                curr_sequence,
                curr_sequence_seqlen,
                curr_sequence_dec_seqlen,
                target_src_seqlen,
                target_tgt_seqlen,
                idx + 1,
                samples_with_ids,
                consumed,
            )
            sequences.append(curr_sequence)
            consumed[idx] = True
        # divide sequences into microbatches
        microbatches = []
        for i in range(0, len(sequences), microbatch_size):
            microbatches.append(sequences[i : i + microbatch_size])
        return microbatches

    def _token_based_partition(self, samples_with_ids, microbatch_tokens):
        microbatches = []
        current_microbatch_tokens = 0
        current_microbatch = []
        for seqlen, dec_seqlen, idx in samples_with_ids:
            rounded_seqlen = self._round_seqlen(seqlen)
            rounded_dec_seqlen = self._round_seqlen(dec_seqlen, decoder=True)
            if (
                current_microbatch_tokens + rounded_seqlen + rounded_dec_seqlen
                > microbatch_tokens
            ):
                if len(current_microbatch) > 0:
                    microbatches.append(current_microbatch.copy())
                current_microbatch = []
                current_microbatch_tokens = 0
            current_microbatch.append([idx])
            current_microbatch_tokens += seqlen + dec_seqlen
        if len(current_microbatch) > 0:
            microbatches.append(current_microbatch)
        return microbatches

    def _subset_partition(self, micro_batch_costs):
        # partition the microbatches into subsets
        # create a mapping from microbatch index to its cost
        mb_cost_map = {}
        for i, mb in enumerate(micro_batch_costs):
            mb_cost_map[i] = mb
        return prtpy.partition(
            algorithm=prtpy.partitioning.kk,
            numbins=self.dp_size,
            items=mb_cost_map,
        )

    def generate_microbatches(
        self,
        sample_sequence_lengths,
        available_rc_types=None,
        decoder_sample_sequence_lengths=None,
        disable_tsp=False,
        bottleneck_tsp=False,
        tsp_dist_function="sum",
        tsp_use_clustering=True,
        tsp_cluster_distance_threshold=16,
        partition_method="dp",
        uniform_partition_batch_size=None,
        token_based_partition_mb_tokens=None,
        enable_packing=False,
    ):
        if available_rc_types is None:
            available_rc_types = ["none", "selective", "full"]
        if (
            self.n_chunks_per_device > 1
            and decoder_sample_sequence_lengths is None
        ):
            raise ValueError(
                "Interleaved schedule with non-encoder-decoder models "
                "are not supported yet."
            )
        # stage 1: determine the order of samples
        if decoder_sample_sequence_lengths is None:
            samples_with_ids = [
                (seqlen, 0, i)
                for i, seqlen in enumerate(sample_sequence_lengths)
            ]
            # single sequence, sorting suffices
            samples_with_ids.sort(reverse=True)
        else:
            if partition_method == "uniform":
                assert uniform_partition_batch_size is not None, (
                    "uniform_partition_batch_size must be specified "
                    "when partition_method is 'uniform'"
                )
                # uniform partitioning, don't need to solve TSP
                samples_with_ids = [
                    (seqlen, dec_seqlen, i)
                    for i, (seqlen, dec_seqlen) in enumerate(
                        zip(
                            sample_sequence_lengths,
                            decoder_sample_sequence_lengths,
                        )
                    )
                ]
            else:
                # multiple sequences, use TSP or 2 level sorting
                # to find the optimal order
                if disable_tsp:
                    samples_with_ids = [
                        (seqlen, dec_seqlen, i)
                        for i, (seqlen, dec_seqlen) in enumerate(
                            zip(
                                sample_sequence_lengths,
                                decoder_sample_sequence_lengths,
                            )
                        )
                    ]
                    # sort first by encoder sequence length, then by decoder
                    samples_with_ids.sort(reverse=True)
                else:
                    permutation = self._solve_sample_order_tsp_problem(
                        sample_sequence_lengths,
                        decoder_sample_sequence_lengths,
                        bottleneck_tsp=bottleneck_tsp,
                        dist_function=tsp_dist_function,
                        use_clustering=tsp_use_clustering,
                        distance_threshold=tsp_cluster_distance_threshold,
                    )
                    samples_with_ids = [
                        (
                            sample_sequence_lengths[i],
                            decoder_sample_sequence_lengths[i],
                            int(i),
                        )
                        for i in permutation
                    ]
        # stage 2: splitting and packing
        # we first calculate the model states memory and subtract it
        # from the memory limit
        # We assume that GPU0 is the bottleneck GPU, which holds Embedding
        # and Encoder of the model if not interleaved, and holds Embedding,
        # Encoder and Decoder of the model if interleaved.
        # rc_type doesn't matter here
        model_states_memory = self.cost_model.get_model_state(
            self.tp_size,
            "none",
            "Embedding",
            n_shards=self.dp_size,
            zero_stage=self.zero_stage,
        )
        encoder_model_state = self.cost_model.get_model_state(
            self.tp_size,
            "none",
            "Encoder",
            n_shards=self.dp_size,
            zero_stage=self.zero_stage,
        )
        if decoder_sample_sequence_lengths is not None:
            decoder_model_state = self.cost_model.get_model_state(
                self.tp_size,
                "none",
                "Decoder",
                n_shards=self.dp_size,
                zero_stage=self.zero_stage,
            )
        else:
            decoder_model_state = 0
        if self.n_chunks_per_device == 1:
            # not interleaved
            layer_states = max(encoder_model_state, decoder_model_state)
        else:
            # interleaved
            layer_states = encoder_model_state + decoder_model_state
            layer_states = layer_states * self.n_chunks_per_device / 2
        layer_states *= self.n_layers_per_stage
        model_states_memory += layer_states
        available_memory = self.memory_limit - model_states_memory

        if (
            self.per_mb_memory_fraction is not None
            and self.per_mb_memory_fraction > 0
        ):
            preferred_memory_limit = (
                self.per_mb_memory_fraction * available_memory
            )
        else:
            preferred_memory_limit = available_memory / self.n_executors
        for memory_type, memory_limit in [
            ("preferred", preferred_memory_limit),
            ("available", available_memory),
        ]:
            # first try to find a partition that do not need special schedule
            # if not found, only make sure that each single microbatch
            # fits in memory
            for rc_type in available_rc_types:
                if partition_method == "dp":
                    # use dynamic programming to find optimal
                    # sequential partition
                    (
                        objective_value,
                        microbatches,
                        microbatch_costs,
                    ) = cpp_consecutive_partition_dp(
                        self.cost_model.get_raw_cost_model(
                            self.tp_size, rc_type
                        ),
                        self.n_executors,
                        self.n_chunks_per_device,
                        self.n_layers_per_stage,
                        self.dp_size,
                        memory_limit,
                        available_memory,
                        samples_with_ids,
                        enable_packing=enable_packing,
                        round_seqlen_multiple=self.round_seqlen_multiple,
                        len_pack_sep_tokens=self.len_pack_sep_tokens,
                        len_decoder_additional_tokens=self.len_decoder_additional_tokens,  # noqa
                    )
                elif partition_method == "token_based":
                    assert token_based_partition_mb_tokens is not None, (
                        "token_based_partition_mb_tokens must be specified "
                        "when partition_method is 'token_based'"
                    )
                    # token based partitioning
                    microbatches = self._token_based_partition(
                        samples_with_ids, token_based_partition_mb_tokens
                    )
                    # dummy objective value, not used
                    objective_value = (
                        0,
                        0,
                        0,
                        [0] * len(microbatches),
                        [0] * len(microbatches),
                    )
                    # dummy microbatch costs
                    microbatch_costs = [0] * len(microbatches)
                elif partition_method == "uniform":
                    microbatches = self._uniform_partition(
                        samples_with_ids, uniform_partition_batch_size
                    )
                    # dummy objective value, not used
                    objective_value = (
                        0,
                        0,
                        0,
                        [0] * len(microbatches),
                        [0] * len(microbatches),
                    )
                    # dummy microbatch costs
                    microbatch_costs = [0] * len(microbatches)
                else:
                    raise ValueError(
                        "unknown partition method: {}".format(partition_method)
                    )
                if math.isinf(objective_value[0]) or math.isnan(
                    objective_value[0]
                ):
                    # memory limit is too small
                    continue
                # sanity check microbatches:
                # make sure that each index appears once and only once
                all_indices = set()
                for mb in microbatches:
                    for sample in mb:
                        for index in sample:
                            assert (
                                index not in all_indices
                            ), "index {} appears more than once".format(index)
                            all_indices.add(index)
                assert sorted(list(all_indices)) == list(
                    range(len(samples_with_ids))
                ), (
                    "not all indices appear in microbatches: "
                    "{} v.s. {}. Input seqlens: {}, target seqlens: {}".format(
                        len(all_indices),
                        len(samples_with_ids),
                        sample_sequence_lengths,
                        decoder_sample_sequence_lengths,
                    )
                )
                # partition microbatches into subsets, each for one data
                # parallel group
                if self.dp_size > 1:
                    partitioned_microbatch_ids = self._subset_partition(
                        microbatch_costs
                    )
                    partitioned_microbatches = []
                    for mb_ids in partitioned_microbatch_ids:
                        partitioned_microbatches.append(
                            [microbatches[i] for i in sorted(mb_ids)]
                        )
                else:
                    partitioned_microbatches = [microbatches]
                return (
                    objective_value,
                    partitioned_microbatches,
                    memory_type,
                    rc_type,
                    (available_memory, model_states_memory, memory_limit),
                )
        # no feasible microbatch split found
        return None, None, None, None, None
