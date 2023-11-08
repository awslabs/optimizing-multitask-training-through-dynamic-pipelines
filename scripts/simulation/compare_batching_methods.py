# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
from multiprocessing import Pool
from typing import Optional

import jsonlines
import numpy as np
from tqdm import tqdm

from dynapipe.data_opt.cost_models import ProfileBasedCostModelWithRC
from dynapipe.data_opt.optimizer import DataAssignmentOptimizer
from dynapipe.model import TransformerModelSpec


def parse_args():
    parser = argparse.ArgumentParser("Compare batching methods")
    parser.add_argument(
        "-t",
        "--method",
        type=str,
        choices=["none", "packing", "dynamic", "fixed_mbs", "fixed_tokens"],
        required=True,
        help="Micro-batching method to use.",
    )
    parser.add_argument(
        "-s",
        "--max-seqlen-range",
        type=str,
        default="2048",
        help="Range of maximum sequence length to simulate. "
        "Format as comma separated list of integers.",
    )
    parser.add_argument(
        "-di",
        "--input-dataset",
        type=str,
        required=True,
        help="Path to a Megatron-LM processed indexfile, "
        "which records the sequence length of samples in npy "
        "format. For input sequences.",
    )
    parser.add_argument(
        "-dt",
        "--target-dataset",
        type=str,
        required=True,
        help="Dataset path for target sequences.",
    )
    parser.add_argument(
        "-c",
        "--cost-model",
        type=str,
        required=True,
        help="Path to a cost model file, needed for dynamic " " batching.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        choices=["gpt", "t5"],
        help="Model to use.",
    )
    parser.add_argument(
        "-g",
        "--global-batch-size",
        type=int,
        default=65536,
        help="Global batch size.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="compare_batching_methods.jsonl",
        help="Output file.",
    )
    parser.add_argument(
        "-ml",
        "--mem-limit",
        type=float,
        default=float("inf"),
        help="Memory limit for the data assignment optimizer.",
    )
    parser.add_argument(
        "-ppr",
        "--pp-degree-range",
        type=str,
        default="1",
        help="Range of pipeline stages to simulate.",
    )
    parser.add_argument(
        "-tpd",
        "--tp-degree",
        type=int,
        default=1,
        help="TP degree to simulate.",
    )
    parser.add_argument(
        "-p",
        "--num-processes",
        type=int,
        default=64,
        help="Number of processes to use.",
    )

    args = parser.parse_args()
    args.max_seqlen_range = [int(x) for x in args.max_seqlen_range.split(",")]
    args.pp_degree_range = [int(x) for x in args.pp_degree_range.split(",")]
    return args


def get_powers_of_2_up_to(n):
    return [2**i for i in range(math.floor(math.log2(n)) + 1)]


def get_candidate_mbs(maxn=512):
    return get_powers_of_2_up_to(maxn)


def get_candidate_tokens(maxn=65536):
    return [x for x in get_powers_of_2_up_to(maxn) if x >= 32]


def get_sequence_lengths(dataset_path, max_seqlen):
    """Get the sequence lengths from a Megatron-LM processed dataset."""
    with open(dataset_path, "rb") as f:
        dataset = np.load(f)
    # dataset contains 3 columns: [start_id, end_id, sequence_length]
    # we only need the sequence length
    return np.clip(dataset[:, 2], 1, max_seqlen).astype(np.int32)[:100000]


def get_global_batches(input_seqlens, target_seqlens, gbs=65536):
    """Get the number of global batches for a given global batch size."""
    global_batches = []
    current_batch = []
    current_batch_size = 0
    for input_seqlen, target_seqlen in zip(input_seqlens, target_seqlens):
        if current_batch_size + input_seqlen + target_seqlen > gbs:
            global_batches.append(current_batch.copy())
            current_batch = []
            current_batch_size = 0
        current_batch.append((input_seqlen, target_seqlen))
        current_batch_size += input_seqlen + target_seqlen
    if current_batch:
        global_batches.append(current_batch.copy())
    return global_batches


def get_model_spec(pp_degree, model="gpt"):
    if model == "gpt":
        return TransformerModelSpec(4 * pp_degree, 0, 4096, 32, 16384, 128)
    elif model == "t5":
        return TransformerModelSpec(
            2 * pp_degree, 2 * pp_degree, 1024, 128, 65536, 128
        )
    else:
        raise ValueError("Unsupported model: {}".format(model))


def get_dataopt(
    pp_degree, cost_model, model="gpt", memlimit=float("inf"), tp_degree=1
):
    num_stages = pp_degree
    model_spec = get_model_spec(pp_degree, model)
    zero_stage = 0
    n_layers_per_stage = 4
    dp_size = 1
    dataopt = DataAssignmentOptimizer(
        cost_model,
        model_spec,
        num_stages,
        n_layers_per_stage,
        n_chunks_per_device=1,
        dp_size=dp_size,
        tp_size=tp_degree,
        zero_stage=zero_stage,
        device_memory_limit=memlimit,
        seqlen_offset=1 if model == "gpt" else 0,
    )
    return dataopt


def pack_sequences(enc_seqlens, dec_seqlens, max_seqlen, model):
    current_enc_seq_len = 0
    current_dec_seq_len = 0
    packed_enc_seqlens = []
    packed_dec_seqlens = []
    for enc_seqlen, dec_seqlen in zip(enc_seqlens, dec_seqlens):
        if (
            current_enc_seq_len + enc_seqlen > max_seqlen
            or current_dec_seq_len + dec_seqlen > max_seqlen
        ):
            packed_enc_seqlens.append(max_seqlen)
            if model == "gpt":
                packed_dec_seqlens.append(0)
            else:
                packed_dec_seqlens.append(max_seqlen)
            current_enc_seq_len = 0
            current_dec_seq_len = 0
        current_enc_seq_len += enc_seqlen
        current_dec_seq_len += dec_seqlen
    if current_enc_seq_len > 0:
        packed_enc_seqlens.append(max_seqlen)
        if model == "gpt":
            packed_dec_seqlens.append(0)
        else:
            packed_dec_seqlens.append(max_seqlen)
    return packed_enc_seqlens, packed_dec_seqlens


def get_microbatches(
    global_batch,
    method,
    model,
    mbs=None,
    dataopt: Optional[DataAssignmentOptimizer] = None,
):
    if method == "none":
        # no micro-batching, directly pad to max sequence length
        batch_np = np.array(global_batch)
        max_input_seqlen = np.max(batch_np[:, 0])
        max_target_seqlen = np.max(batch_np[:, 1])
        mbs = len(global_batch)
        return [(mbs, max_input_seqlen, max_target_seqlen)], "none"
    elif method == "dynamic":
        enc_seqlens = [x[0] for x in global_batch]
        dec_seqlens = [x[1] for x in global_batch]
        out = dataopt.generate_microbatches(
            enc_seqlens,
            decoder_sample_sequence_lengths=dec_seqlens
            if model == "t5"
            else None,
            bottleneck_tsp=False,
            partition_method="dp",
            enable_packing=False,
            tsp_dist_function="sum",
        )
        if out[0] is None:
            import pickle

            with open("dataopt.pkl", "wb") as f:
                pickle.dump(dataopt, f)
            with open("global_batch.pkl", "wb") as f:
                pickle.dump(global_batch, f)
            return None, None
        (
            objective_values,
            microbatches,
            memory_type,
            rc_type,
            (avail_mem, model_state, per_mb_memory_limit),
        ) = out
        micro_batch_shapes = []
        assert len(microbatches) == 1
        for microbatch in microbatches[0]:
            mbs = len(microbatch)
            max_enc_seqlen = 0
            max_dec_seqlen = 0
            for sequence in microbatch:
                assert len(sequence) == 1
                enc_seqlen = enc_seqlens[sequence[0]]
                dec_seqlen = dec_seqlens[sequence[0]]
                max_enc_seqlen = max(max_enc_seqlen, enc_seqlen)
                max_dec_seqlen = max(max_dec_seqlen, dec_seqlen)
            micro_batch_shapes.append((mbs, max_enc_seqlen, max_dec_seqlen))
        return micro_batch_shapes, rc_type
    elif method in ["fixed_mbs", "packing"]:
        assert mbs is not None
        microbatches = []
        sorted_batch = sorted(global_batch, reverse=True)
        for i in range(0, len(sorted_batch), mbs):
            batch_np = np.array(sorted_batch[i : i + mbs])
            max_input_seqlen = np.max(batch_np[:, 0])
            max_target_seqlen = np.max(batch_np[:, 1])
            microbatches.append(
                (len(batch_np), max_input_seqlen, max_target_seqlen)
            )
        return microbatches, "none"
    elif method == "fixed_tokens":
        assert mbs is not None
        enc_seqlens = [x[0] for x in global_batch]
        dec_seqlens = [x[1] for x in global_batch]
        out = dataopt.generate_microbatches(
            enc_seqlens,
            decoder_sample_sequence_lengths=dec_seqlens
            if args.model == "t5"
            else None,
            bottleneck_tsp=False,
            partition_method="token_based",
            enable_packing=False,
            token_based_partition_mb_tokens=mbs,
            tsp_dist_function="sum",
        )
        (
            objective_values,
            microbatches,
            memory_type,
            rc_type,
            (avail_mem, model_state, per_mb_memory_limit),
        ) = out
        if out[0] is None:
            return None, None
        micro_batch_shapes = []
        assert len(microbatches) == 1
        for microbatch in microbatches[0]:
            mbs = len(microbatch)
            max_enc_seqlen = 0
            max_dec_seqlen = 0
            for sequence in microbatch:
                assert len(sequence) == 1
                enc_seqlen = enc_seqlens[sequence[0]]
                dec_seqlen = dec_seqlens[sequence[0]]
                max_enc_seqlen = max(max_enc_seqlen, enc_seqlen)
                max_dec_seqlen = max(max_dec_seqlen, dec_seqlen)
            micro_batch_shapes.append((mbs, max_enc_seqlen, max_dec_seqlen))
        return micro_batch_shapes, rc_type
    else:
        raise ValueError(
            "Unsupported micro-batching method: {}".format(method)
        )


def count_tokens(microbatches):
    total_tokens = 0
    for mbs, enc_seqlen, dec_seqlen in microbatches:
        total_tokens += mbs * (enc_seqlen + dec_seqlen)
    return total_tokens


def get_execution_time_and_memory(
    microbatches,
    rc_type,
    cost_model: ProfileBasedCostModelWithRC,
    pp_degree,
    tp_degree=1,
    model="gpt",
):
    sum_execution_time = 0
    max_execution_time = 0
    peak_memory = 0
    memory_sliding_window_size = pp_degree
    memory_sliding_window = []

    def round_up(x, multiple, extra=0):
        return (x + extra + multiple - 1) // multiple * multiple

    for mbs, enc_seqlen, dec_seqlen in microbatches:
        enc_seqlen = round_up(enc_seqlen, 8)
        if dec_seqlen > 0:
            dec_seqlen = round_up(dec_seqlen, 8, extra=2)
        enc_time = (
            cost_model.get_cost(
                tp_degree, rc_type, "Encoder FW", enc_seqlen, mbs
            )
            + cost_model.get_cost(
                tp_degree, rc_type, "Encoder BW", enc_seqlen, mbs
            )
        ) * (4 if model == "gpt" else 2)
        if model == "t5":
            dec_time = (
                cost_model.get_cost(
                    tp_degree,
                    rc_type,
                    "Decoder FW",
                    (enc_seqlen, dec_seqlen),
                    mbs,
                )
                + cost_model.get_cost(
                    tp_degree,
                    rc_type,
                    "Decoder BW",
                    (enc_seqlen, dec_seqlen),
                    mbs,
                )
            ) * 2
            if dec_time < 0:
                dec_time = 0
            if pp_degree > 1:
                time = max(enc_time, dec_time) * 2
            else:
                time = enc_time + dec_time
        else:
            time = enc_time
        memory = max(
            cost_model.get_peak_activation(
                tp_degree, rc_type, "Encoder", enc_seqlen, mbs
            )
            * (4 if model == "gpt" else 2),
            cost_model.get_stored_activation(
                tp_degree, rc_type, "Encoder", enc_seqlen, mbs
            )
            * (4 if model == "gpt" else 2),
        )
        if model == "t5":
            if pp_degree == 1:
                memory += max(
                    cost_model.get_peak_activation(
                        tp_degree,
                        rc_type,
                        "Decoder",
                        (enc_seqlen, dec_seqlen),
                        mbs,
                    )
                    * 2,
                    cost_model.get_stored_activation(
                        tp_degree,
                        rc_type,
                        "Decoder",
                        (enc_seqlen, dec_seqlen),
                        mbs,
                    )
                    * 2,
                )
            else:
                memory *= 2
        sum_execution_time += time
        max_execution_time = max(max_execution_time, time)
        memory_sliding_window.append(memory)
        if len(memory_sliding_window) > memory_sliding_window_size:
            memory_sliding_window.pop(0)
        peak_memory = max(peak_memory, sum(memory_sliding_window))
    return (
        sum_execution_time + max_execution_time * (pp_degree - 1),
        peak_memory,
    )


def eval_global_batch_process(args):
    global_batch, method, model, mbs, pp_degree, tp_degree = args
    microbatches, rc_type = get_microbatches(
        global_batch, method, model, mbs, dataopt
    )
    if microbatches is None:
        return None, None, None
    execution_time, peak_memory = get_execution_time_and_memory(
        microbatches,
        rc_type,
        cost_model,
        pp_degree,
        tp_degree=tp_degree,
        model=model,
    )
    return count_tokens(microbatches), execution_time, peak_memory


if __name__ == "__main__":
    args = parse_args()
    print("PP Degree range: {}".format(args.pp_degree_range))
    print("TP Degree: {}".format(args.tp_degree))
    print("Method: {}".format(args.method))
    print("Input dataset: {}".format(args.input_dataset))
    print("Target dataset: {}".format(args.target_dataset))
    print("Cost model: {}".format(args.cost_model))
    print("Model: {}".format(args.model))
    print("Max sequence length range: {}".format(args.max_seqlen_range))
    print("PP Degree range: {}".format(args.pp_degree_range))
    cost_model = ProfileBasedCostModelWithRC.load(args.cost_model)
    with jsonlines.open(args.output, "a") as out_file:
        pp_degree_iterator = tqdm(args.pp_degree_range)
        for pp_degree in pp_degree_iterator:
            pp_degree_iterator.set_description(
                "PP Degree: {}".format(pp_degree)
            )
            if args.method in ["dynamic", "fixed_tokens"]:
                dataopt = get_dataopt(
                    pp_degree,
                    cost_model,
                    args.model,
                    args.mem_limit,
                    args.tp_degree,
                )
            else:
                dataopt = None
            seqlen_iterator = tqdm(args.max_seqlen_range, leave=False)
            for max_seqlen in seqlen_iterator:
                seqlen_iterator.set_description(
                    "Max sequence length: {}".format(max_seqlen)
                )
                input_sequence_lengths = get_sequence_lengths(
                    args.input_dataset, max_seqlen
                )
                target_sequence_lengths = get_sequence_lengths(
                    args.target_dataset, max_seqlen
                )
                if args.model == "gpt":
                    input_sequence_lengths = np.clip(
                        np.array(input_sequence_lengths)
                        + np.array(target_sequence_lengths),
                        1,
                        max_seqlen,
                    )
                    target_sequence_lengths = [0] * len(
                        target_sequence_lengths
                    )
                total_actual_tokens = np.sum(input_sequence_lengths) + np.sum(
                    target_sequence_lengths
                )
                if args.method == "packing":
                    (
                        input_sequence_lengths,
                        target_sequence_lengths,
                    ) = pack_sequences(
                        input_sequence_lengths,
                        target_sequence_lengths,
                        max_seqlen,
                        args.model,
                    )

                if args.method in ["fixed_mbs", "packing"]:
                    mbs_candidates = get_candidate_mbs()
                elif args.method == "fixed_tokens":
                    mbs_candidates = get_candidate_tokens()
                else:
                    mbs_candidates = [None]
                mbs_iterator = tqdm(mbs_candidates, leave=False)
                for mbs in mbs_iterator:
                    mbs_iterator.set_description(
                        "Micro-batch size: {}".format(mbs)
                    )
                    total_exec_time = 0
                    total_peak_memory = 0
                    total_padded_tokens = 0
                    should_abort = False
                    global_batches = get_global_batches(
                        input_sequence_lengths, target_sequence_lengths
                    )
                    pool = Pool(processes=args.num_processes)

                    def arg_generator():
                        for global_batch in global_batches:
                            yield (
                                global_batch,
                                args.method,
                                args.model,
                                mbs,
                                pp_degree,
                                args.tp_degree,
                            )

                    pbar = tqdm(
                        total=len(global_batches),
                        leave=False,
                        desc="Global batch",
                    )
                    # for arg in arg_generator():
                    #     eval_global_batch_process(arg)
                    #     exit(0)
                    for (
                        padded_tokens,
                        execution_time,
                        peak_memory,
                    ) in pool.imap_unordered(
                        eval_global_batch_process, arg_generator()
                    ):
                        if padded_tokens is None:
                            should_abort = True
                        else:
                            total_padded_tokens += padded_tokens
                            total_exec_time += execution_time
                            total_peak_memory = max(
                                total_peak_memory, peak_memory
                            )
                        pbar.update()
                    pbar.close()
                    pool.close()
                    if not should_abort:
                        throughput = total_actual_tokens / (
                            total_exec_time / 1e3
                        )
                        if throughput > 1 or args.method == "none":
                            result_item = {
                                "pp_degree": pp_degree,
                                "max_seqlen": max_seqlen,
                                "mbs": mbs,
                                "method": args.method,
                                "throughput": throughput,
                                "padding_efficiency": total_actual_tokens
                                / total_padded_tokens,
                                "peak_memory": total_peak_memory,
                            }
                            out_file.write(result_item)
