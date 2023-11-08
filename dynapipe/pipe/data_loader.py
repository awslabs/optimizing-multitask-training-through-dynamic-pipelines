# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass, field, fields
from queue import Empty
from typing import List, Optional

import torch
from torch.utils.data import DataLoader as PTDataLoader

from dynapipe.data_opt.cost_models import ProfileBasedCostModelWithRC
from dynapipe.data_opt.optimizer import DataAssignmentOptimizer
from dynapipe.model import DynaPipeCluster, TransformerModelSpec
from dynapipe.pipe.instructions import (
    deserialize_list_of_eps,
    serialize_list_of_eps,
)
from dynapipe.schedule_opt.execution_planner import ExecutionPlanner
from dynapipe.utils.logger import create_logger, logger

from .kv_redis import RedisKVStore
from .utils import validate_device_assignment

MANAGER_PROCESS_TIMEOUT = 1
RECEIVER_PROCESS_TIMEOUT = 1
KVSTORE_TIMEOUT = 1800  # 30 minutes

# ONLY USED FOR DEBUG PURPOSES
DEBUG_USE_DUMMY_EP = False
DEBUG_DUMP_EP_STATS = os.getenv(
    "DYNAPIPE_DEBUG_DUMP_EP_STATS", "False"
).lower() in ("true", "1", "t")
DEBUG_DUMP_EP_PREFIX = os.environ.get("DYNAPIPE_DEBUG_DUMP_EP_PREFIX", None)
if DEBUG_DUMP_EP_STATS and DEBUG_DUMP_EP_PREFIX is None:
    raise ValueError(
        "DYNAPIPE_DEBUG_DUMP_EP_PREFIX must be set if "
        "DYNAPIPE_DEBUG_DUMP_EP_STATS is set."
    )

_kvstore_handle = None


def _init_kv_store(is_master, logger=None):
    host = os.environ.get("DYNAPIPE_KV_HOST", "localhost")
    port = os.environ.get("DYNAPIPE_KV_PORT", 29500)
    if logger is not None:
        logger.debug(
            "Init kv store, is_master: {}, host: {}, port: {}".format(
                is_master, host, port
            )
        )
    # kv_store = torch.distributed.TCPStore(
    #     "127.0.0.1",
    #     port,
    #     is_master=is_master,
    #     timeout=timedelta(seconds=KVSTORE_TIMEOUT),
    # )
    kv_store = RedisKVStore(host, port, is_master=is_master)
    return kv_store, host, port


def _checked_delete_key(kv_store: RedisKVStore, key: str, logger=None):
    result = kv_store.delete_key(key)
    if not result:
        raise RuntimeError(
            "Internal error: failed to delete key " "{}.".format(key)
        )
    if logger is not None:
        logger.debug("Deleted key: {}".format(key))


def _get_from_shared_kv_store(
    kv_store: RedisKVStore,
    key: str,
    reader_idx: int,
    n_total_readers: int,
    decode: bool = True,
    logger=None,
):
    reader_count_key = key + "_rc"
    reader_ack_key = key + "_r{}_ack".format(reader_idx)
    # wait for reader ack
    if logger is not None:
        logger.debug("Waiting for reader ack key: {}".format(reader_ack_key))
    kv_store.get(reader_ack_key)
    if logger is not None:
        logger.debug(
            "Got reader ack key: {}, waiting for data key: {}".format(
                reader_ack_key, key
            )
        )
    data = kv_store.get(key)
    if logger is not None:
        logger.debug("Removing reader ack key: {}".format(reader_ack_key))
    # remove reader ack
    _checked_delete_key(kv_store, reader_ack_key, logger=logger)
    # get reader count
    reader_count = kv_store.add(reader_count_key, 1)
    if reader_count == n_total_readers:
        if logger is not None:
            logger.debug(
                "Last reader, reset reader count: {}".format(reader_count_key)
            )
        # reset reader count
        result_readers = kv_store.add(reader_count_key, -n_total_readers)
        assert result_readers == 0
        if logger is not None:
            logger.debug("Last reader, remove data key: {}".format(key))
        # remove data key
        _checked_delete_key(kv_store, key, logger=logger)
        if logger is not None:
            logger.debug("Last reader, set ack key: {}".format(key + "_ack"))
        # set all reader ack keys
        keys_to_reset = [
            key + "_r{}_ack".format(i) for i in range(n_total_readers)
        ]
        if logger is not None:
            logger.debug("Last reader, reset keys: {}".format(keys_to_reset))
        for reset_key in keys_to_reset:
            val = kv_store.add(reset_key, 1)
            # make sure the key is set
            got_val = int(kv_store.get(reset_key).decode())
            if not val == got_val:
                raise RuntimeError(
                    "Failed to set reader ack key: {}".format(reset_key)
                )
            if logger is not None:
                logger.debug("Set reader ack key: {}".format(reset_key))
        # set data ack key
        kv_store.add(key + "_ack", 1)

    if decode:
        return data.decode()
    return data


def _put_to_shared_kv_store(
    kv_store: RedisKVStore, key: str, data, logger=None
):
    # put execution plan into local kv store
    ack_key = key + "_ack"
    if logger is not None:
        logger.debug("Wait for data ack key: {}".format(ack_key))
    # wait for ack key
    kv_store.get(ack_key)
    # remove ack key
    _checked_delete_key(kv_store, ack_key, logger=logger)
    if logger is not None:
        logger.debug("Set data key: {}".format(key))
    # set data key
    kv_store.set(key, data)


@dataclass
class WorkerData:
    round_seqlen_multiple: Optional[int] = None
    logger: Optional[logging.Logger] = None
    kv_store: Optional[RedisKVStore] = None
    processed_batches: Optional[int] = None
    kv_buffer_size: Optional[int] = None
    seqlen_offset: Optional[int] = 0

    def check_initialized(self):
        cls_fields = fields(self.__class__)
        for fld in cls_fields:
            if getattr(self, fld.name) is None:
                raise RuntimeError(
                    "Worker data not initialized: {}".format(fld.name)
                )


@dataclass
class PreprocessingWorkerData(WorkerData):
    # required at initialization:
    node_rank: Optional[int] = None
    profile_path: Optional[str] = None
    # filled later in worker init:
    dataopt: Optional[DataAssignmentOptimizer] = None
    exec_planner: Optional[ExecutionPlanner] = None
    partition_method: Optional[str] = None
    token_based_partition_mbs: Optional[int] = None
    disable_tsp: Optional[bool] = None
    schedule_method: Optional[str] = None
    disable_mb_permutation: Optional[bool] = None
    disable_scheduler_memory_limit: Optional[bool] = None
    enable_packing: Optional[bool] = None
    n_layers_per_stage: Optional[int] = None
    assigned_iters_per_node: Optional[int] = None
    node_size: Optional[int] = None

    def __post_init__(self):
        if self.node_rank is None:
            raise RuntimeError("node_rank must be set at initialization.")
        if self.profile_path is None:
            raise RuntimeError("profile_path must be set at initialization.")


@dataclass
class DataloaderWorkerData(WorkerData):
    # required at initialization:
    dp_rank: Optional[int] = None
    pp_rank: Optional[int] = None
    virtual_pp_rank: Optional[int] = None
    # filled later in worker init:
    dp_size: Optional[int] = None
    pp_size: Optional[int] = None
    virtual_pp_size: Optional[int] = None

    def __post_init__(self):
        if self.dp_rank is None:
            raise RuntimeError("dp_rank must be set at initialization.")
        if self.pp_rank is None:
            raise RuntimeError("pp_rank must be set at initialization.")
        if self.virtual_pp_rank is None:
            raise RuntimeError(
                "virtual_pp_rank must be " "set at initialization."
            )


class KVStoreMetaKeys:
    DP_SIZE = "data_parallel_size"
    TP_SIZE = "tensor_parallel_size"
    PP_SIZE = "pipeline_parallel_size"
    VIRTUAL_PP_SIZE = "virtual_pipeline_parallel_size"
    ZERO_STAGE = "zero_stage"
    NODE_SIZE = "node_size"
    MODEL_SPEC = "model_spec"
    N_EXECS = "n_executors"
    N_LAYERS_PER_STAGE = "n_layers_per_stage"
    N_CHUNKS_PER_DEVICE = "n_chunks_per_device"
    DEVICE_MEMORY_LIMIT = "device_memory_limit"
    PARTITION_METHOD = "partition_method"
    TOKEN_BASED_PARTITION_MBS = "token_based_partition_mbs"
    DISABLE_TSP = "disable_tsp"
    SCHEDULE_METHOD = "schedule_method"
    DISABLE_MB_PERMUTATION = "disable_mb_permutation"
    DISABLE_SCHEDULER_MEMORY_LIMIT = "disable_scheduler_memory_limit"
    ENABLE_PACKING = "enable_packing"
    PER_MB_MEM_FRAC = "per_mb_memory_fraction"
    CLUSTER_SPEC = "cluster_spec"
    DEV_ASSIGNMENT = "device_assignment"
    KV_BUFFER_SIZE = "kv_buffer_size"
    ROUND_SEQLEN_MULT = "round_seqlen_multiple"
    ASSIGNED_ITER_PER_NODE = "assigned_iters_per_node"
    SEQLEN_OFFSET = "seqlen_offset"
    MODEL_TYPE = "model_type"
    # used outside dataloader
    N_ITERS = "n_iters"


@dataclass
class TrainingSpec:
    cm_path: str
    cluster_spec: DynaPipeCluster
    model_spec: TransformerModelSpec
    data_parallel_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    zero_stage: int
    device_assignment: List[int]
    device_memory_limit: int
    partition_algo: str = "dp"
    token_based_partition_mbs: int = 0
    disable_tsp: bool = False
    schedule_method: str = "dynamic"
    disable_mb_permutation: bool = False
    disable_scheduler_memory_limit: bool = False
    enable_packing: bool = False
    per_mb_memory_fraction: float = -1.0
    round_seqlen_multiple: int = 8
    seqlen_offset: int = 0
    limit_rc_type: Optional[List[str]] = None
    model_type: str = "gpt"
    n_executors: int = field(init=False)
    n_layers_per_stage: int = field(init=False)
    n_chunks_per_device: int = field(init=False)

    def __post_init__(self):
        self.n_executors = max(self.device_assignment) + 1
        (
            _,
            _,
            self.n_layers_per_stage,
            self.n_chunks_per_device,
        ) = validate_device_assignment(
            self.model_spec, self.cluster_spec, self.device_assignment
        )


def _preprocessing_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    worker_data: PreprocessingWorkerData = dataset.worker_data
    # local information such as rank should be stored in the dataset
    node_rank = worker_data.node_rank

    logger = create_logger(
        "preprocess_worker",
        prefix=f"Node {node_rank} | " f"Preprocessing Worker {worker_id}",
        log_file="preprocessing/" f"nr{node_rank}_w{worker_id}.log",
    )
    # init kv store first since we need to get data from it
    kv_store, host, port = _init_kv_store(is_master=False, logger=logger)

    node_size = int(kv_store.get(KVStoreMetaKeys.NODE_SIZE).decode())
    assigned_iters_per_node = int(
        kv_store.get(KVStoreMetaKeys.ASSIGNED_ITER_PER_NODE).decode()
    )
    seqlen_offset = int(kv_store.get(KVStoreMetaKeys.SEQLEN_OFFSET).decode())
    model_type = kv_store.get(KVStoreMetaKeys.MODEL_TYPE).decode()
    worker_data.node_size = node_size
    worker_data.assigned_iters_per_node = assigned_iters_per_node
    worker_data.seqlen_offset = seqlen_offset
    round_seqlen_multiple = int(
        kv_store.get(KVStoreMetaKeys.ROUND_SEQLEN_MULT).decode()
    )
    worker_data.round_seqlen_multiple = round_seqlen_multiple

    worker_data.logger = logger
    worker_data.logger.debug("Subprocess started.")
    worker_data.kv_store = kv_store
    worker_data.processed_batches = 0

    # create data opt
    profile_path = worker_data.profile_path
    cm = ProfileBasedCostModelWithRC.load(profile_path)
    model_spec_bytes = (
        kv_store.get(KVStoreMetaKeys.MODEL_SPEC).decode().encode("iso-8859-1")
    )
    model_spec = TransformerModelSpec.deserialize(model_spec_bytes)
    data_parallel_size = int(kv_store.get(KVStoreMetaKeys.DP_SIZE).decode())
    tensor_parallel_size = int(kv_store.get(KVStoreMetaKeys.TP_SIZE).decode())
    zero_stage = int(kv_store.get(KVStoreMetaKeys.ZERO_STAGE).decode())
    n_executors = int(kv_store.get(KVStoreMetaKeys.N_EXECS).decode())
    n_layers_per_stage = int(
        kv_store.get(KVStoreMetaKeys.N_LAYERS_PER_STAGE).decode()
    )
    n_chunks_per_device = int(
        kv_store.get(KVStoreMetaKeys.N_CHUNKS_PER_DEVICE).decode()
    )
    device_memory_limit = int(
        kv_store.get(KVStoreMetaKeys.DEVICE_MEMORY_LIMIT).decode()
    )
    per_mb_memory_fraction = float(
        kv_store.get(KVStoreMetaKeys.PER_MB_MEM_FRAC).decode()
    )
    if per_mb_memory_fraction < 0:
        per_mb_memory_fraction = None
    # Special handling for t5 with no pipelining. Our dp algorithm
    # assumes encoder and decoder are on different devices, which
    # is only true for pp_size > 1. For pp_size = 1, we trick the
    # algorithm by using n_chunks_per_device = 2
    if model_type == "t5" and n_executors == 1:
        assert n_layers_per_stage % 2 == 0
        assert n_chunks_per_device == 1
        n_chunks_per_device_for_dp = 2
        n_layers_per_stage_for_dp = n_layers_per_stage // 2
    else:
        n_chunks_per_device_for_dp = n_chunks_per_device
        n_layers_per_stage_for_dp = n_layers_per_stage
    dataopt = DataAssignmentOptimizer(
        cm,
        model_spec,
        n_executors,
        n_layers_per_stage_for_dp,
        n_chunks_per_device=n_chunks_per_device_for_dp,
        dp_size=data_parallel_size,
        tp_size=tensor_parallel_size,
        zero_stage=zero_stage,
        device_memory_limit=device_memory_limit,
        round_seqlen_multiple=round_seqlen_multiple,
        per_mb_memory_fraction=per_mb_memory_fraction,
        seqlen_offset=seqlen_offset,
    )
    worker_data.dataopt = dataopt
    # get other args
    partition_method = kv_store.get(KVStoreMetaKeys.PARTITION_METHOD).decode()
    token_based_partition_mbs = int(
        kv_store.get(KVStoreMetaKeys.TOKEN_BASED_PARTITION_MBS).decode()
    )
    disable_tsp = kv_store.get(KVStoreMetaKeys.DISABLE_TSP).decode()
    if disable_tsp.lower() == "true":
        disable_tsp = True
    else:
        disable_tsp = False
    schedule_method = kv_store.get(KVStoreMetaKeys.SCHEDULE_METHOD).decode()
    disable_mb_permutation = kv_store.get(
        KVStoreMetaKeys.DISABLE_MB_PERMUTATION
    ).decode()
    if disable_mb_permutation.lower() == "true":
        disable_mb_permutation = True
    else:
        disable_mb_permutation = False
    disable_scheduler_memory_limit = kv_store.get(
        KVStoreMetaKeys.DISABLE_SCHEDULER_MEMORY_LIMIT
    ).decode()
    if disable_scheduler_memory_limit.lower() == "true":
        disable_scheduler_memory_limit = True
    else:
        disable_scheduler_memory_limit = False
    enable_packing = kv_store.get(KVStoreMetaKeys.ENABLE_PACKING).decode()
    if enable_packing.lower() == "true":
        enable_packing = True
    else:
        enable_packing = False
    worker_data.partition_method = partition_method
    worker_data.token_based_partition_mbs = token_based_partition_mbs
    worker_data.disable_tsp = disable_tsp
    worker_data.schedule_method = schedule_method
    worker_data.disable_mb_permutation = disable_mb_permutation
    worker_data.disable_scheduler_memory_limit = disable_scheduler_memory_limit
    worker_data.enable_packing = enable_packing
    worker_data.n_layers_per_stage = n_layers_per_stage
    # create exec planner
    cluster_spec_json = kv_store.get(KVStoreMetaKeys.CLUSTER_SPEC).decode()
    cluster_spec = DynaPipeCluster.loads(cluster_spec_json)
    device_assignment = [
        int(x)
        for x in kv_store.get(KVStoreMetaKeys.DEV_ASSIGNMENT)
        .decode()
        .split(",")
    ]
    exec_planner = ExecutionPlanner(
        cluster_spec,
        model_spec,
        device_assignment,
        device_memory_limit,
        cm,
        dp_size=data_parallel_size,
        tp_size=tensor_parallel_size,
        zero_stage=zero_stage,
        logger=logger,
    )
    worker_data.exec_planner = exec_planner

    kv_buffer_size = int(kv_store.get(KVStoreMetaKeys.KV_BUFFER_SIZE).decode())
    worker_data.kv_buffer_size = kv_buffer_size
    worker_data.check_initialized()
    worker_data.logger.debug("Exiting init function.")


def _worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    worker_data: DataloaderWorkerData = dataset.worker_data
    # local information such as rank should be stored in the dataset
    dp_rank = worker_data.dp_rank
    pp_rank = worker_data.pp_rank
    virtual_pp_rank = worker_data.virtual_pp_rank

    logger = create_logger(
        "worker",
        prefix=f"dRank {dp_rank} pRank "
        f"{pp_rank} vpRank {virtual_pp_rank} | "
        f"Dataloader Worker {worker_id}",
        log_file="dataloader/"
        f"dr{dp_rank}_pr{pp_rank}_vr{virtual_pp_rank}_w{worker_id}.log",
    )
    # init kv store first since we need to get data from it
    kv_store, host, port = _init_kv_store(is_master=False, logger=logger)

    data_parallel_size = int(kv_store.get(KVStoreMetaKeys.DP_SIZE).decode())
    pipeline_parallel_size = int(
        kv_store.get(KVStoreMetaKeys.PP_SIZE).decode()
    )
    virtual_pipeline_parallel_size = int(
        kv_store.get(KVStoreMetaKeys.VIRTUAL_PP_SIZE).decode()
    )
    round_seqlen_multiple = int(
        kv_store.get(KVStoreMetaKeys.ROUND_SEQLEN_MULT).decode()
    )
    seqlen_offset = int(kv_store.get(KVStoreMetaKeys.SEQLEN_OFFSET).decode())
    worker_data.dp_size = data_parallel_size
    worker_data.pp_size = pipeline_parallel_size
    worker_data.virtual_pp_size = virtual_pipeline_parallel_size
    worker_data.round_seqlen_multiple = round_seqlen_multiple
    worker_data.seqlen_offset = seqlen_offset
    worker_data.logger = logger
    worker_data.logger.debug("Subprocess started.")
    worker_data.kv_store = kv_store
    worker_data.processed_batches = 0
    kv_buffer_size = int(kv_store.get(KVStoreMetaKeys.KV_BUFFER_SIZE).decode())
    worker_data.kv_buffer_size = kv_buffer_size
    worker_data.check_initialized()
    worker_data.logger.debug("Exiting init function.")


def _collate_samples(
    batch,
    indices: List[List[List[int]]],
    pack_fn,
    constructor_fn,
    collate_fn,
    round_seqlen_multiple=8,
    seqlen_offset=0,
    encoder_key="text_enc",
    decoder_key="text_dec",
    len_decoder_additional_tokens=2,
):
    # pack microbatches
    microbatches = []
    batch_shapes = []
    input_only = False
    for microbatch in indices:
        encoder_inputs = []
        encoder_extras = []
        decoder_inputs = []
        decoder_extras = []
        for sample in microbatch:
            encoder_input = []
            decoder_input = []
            for sequence_index in sample:
                sample_data = batch[sequence_index]
                if isinstance(sample_data, tuple):
                    if len(sample_data) == 2:
                        enc_data, dec_data = sample_data
                    else:
                        enc_data = sample_data[0]
                        dec_data = []
                        input_only = True
                elif isinstance(sample_data, dict):
                    enc_data = sample_data[encoder_key]
                    if decoder_key in sample_data:
                        dec_data = sample_data[decoder_key]
                    else:
                        dec_data = []
                        input_only = True
                else:
                    raise ValueError(
                        "Sample data must be a tuple or dict, got "
                        f"{type(sample_data)}."
                    )
                if input_only:
                    assert len(dec_data) == 0
                encoder_input.append(enc_data)
                decoder_input.append(dec_data)
            encoder_input, encoder_extra = pack_fn(encoder_input)
            encoder_inputs.append(encoder_input)
            encoder_extras.append(encoder_extra)
            if not input_only:
                decoder_input, decoder_extra = pack_fn(decoder_input)
                decoder_inputs.append(decoder_input)
                decoder_extras.append(decoder_extra)
        encoder_seqlen = max([len(x) for x in encoder_inputs])
        if not input_only:
            decoder_seqlen = (
                max([len(x) for x in decoder_inputs])
                + len_decoder_additional_tokens
            )
        else:
            decoder_seqlen = 0
        # round seqlen
        encoder_seqlen -= seqlen_offset
        encoder_seqlen = (
            (encoder_seqlen + round_seqlen_multiple - 1)
            // round_seqlen_multiple
            * round_seqlen_multiple
        )
        if not input_only:
            decoder_seqlen -= seqlen_offset
        decoder_seqlen = (
            (decoder_seqlen + round_seqlen_multiple - 1)
            // round_seqlen_multiple
            * round_seqlen_multiple
        )
        batch_shapes.append(
            (
                len(microbatch),
                encoder_seqlen,
                decoder_seqlen,
            )
        )
        constructed_samples = []
        if not input_only:
            for (
                encoder_input,
                encoder_extra,
                decoder_input,
                decoder_extra,
            ) in zip(
                encoder_inputs, encoder_extras, decoder_inputs, decoder_extras
            ):
                constructed_samples.append(
                    constructor_fn(
                        encoder_input,
                        encoder_extra,
                        decoder_input,
                        decoder_extra,
                        encoder_seqlen + seqlen_offset,
                        decoder_seqlen + seqlen_offset,
                    )
                )
        else:
            for encoder_input, encoder_extra in zip(
                encoder_inputs, encoder_extras
            ):
                constructed_samples.append(
                    constructor_fn(
                        encoder_input,
                        encoder_extra,
                        None,
                        None,
                        encoder_seqlen + seqlen_offset,
                        None,
                    )
                )
        constructed_microbatch = collate_fn(constructed_samples)
        microbatches.append(constructed_microbatch)
    return microbatches, batch_shapes


def get_preprocessing_collate_fn(
    pack_fn,
    encoder_key="text_enc",
    decoder_key="text_dec",
    limit_rc_type=None,
):
    def _collate_fn(batch):
        # get states from variables set in worker_init_fn
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        worker_data: PreprocessingWorkerData = dataset.worker_data
        try:
            # needed by both planner and worker
            kv_store = worker_data.kv_store
            kv_buffer_size = worker_data.kv_buffer_size
            # processed_batches is local to each worker since dataset
            # is replicated
            processed_batches = worker_data.processed_batches

            assigned_iters_per_node = worker_data.assigned_iters_per_node
            # pytorch assigns batch to workers in round-robin fashion
            # so we can use worker_info.id to determine the current batch index
            # TODO: find a better way to do this since it is not documented
            #       and depends on implementation
            current_batch_idx = (
                worker_info.num_workers * processed_batches + worker_info.id
            )
            assigned_node_id = (
                current_batch_idx // assigned_iters_per_node
            ) % (worker_data.node_size)
            # increase processed_batches
            worker_info.dataset.worker_data.processed_batches += 1
            if assigned_node_id != worker_data.node_rank:
                # directly return
                worker_data.logger.debug(
                    f"Skipped generating EP for iteration {current_batch_idx}."
                )
                return None

            buffer_slot = current_batch_idx % kv_buffer_size
            indices_key = f"ind_{buffer_slot}"
            ep_key = f"ep_{buffer_slot}"

            dataopt: DataAssignmentOptimizer = worker_data.dataopt
            partition_method: str = worker_data.partition_method
            token_based_partition_mbs: int = (
                worker_data.token_based_partition_mbs
            )
            disable_tsp: bool = worker_data.disable_tsp
            schedule_method: str = worker_data.schedule_method
            disable_mb_permutation: bool = worker_data.disable_mb_permutation
            disable_scheduler_memory_limit: bool = (
                worker_data.disable_scheduler_memory_limit
            )
            enable_packing: bool = worker_data.enable_packing
            exec_planner: ExecutionPlanner = worker_data.exec_planner
            # calculate exec plans on planner
            input_seqlens = []
            target_seqlens = []
            input_only = False
            for sample in batch:
                if isinstance(sample, tuple):
                    if len(sample) == 2:
                        encoder_seq, decoder_seq = sample
                    else:
                        encoder_seq = sample[0]
                        decoder_seq = []
                        input_only = True
                elif isinstance(sample, dict):
                    assert encoder_key in sample, (
                        f"encoder_key '{encoder_key}' not found in sample. "
                        f"Available keys: {sample.keys()}"
                    )
                    encoder_seq = sample[encoder_key]
                    if decoder_key in sample:
                        decoder_seq = sample[decoder_key]
                    else:
                        decoder_seq = []
                        input_only = True
                else:
                    raise ValueError(
                        "sample must be either a tuple or a dict"
                        " but got {}".format(type(sample))
                    )
                if input_only:
                    assert len(decoder_seq) == 0
                encoder_seqlen = len(encoder_seq)
                input_seqlens.append(encoder_seqlen)
                decoder_seqlen = len(decoder_seq)
                target_seqlens.append(decoder_seqlen)
            worker_data.logger.debug(
                f"Generating EP for iteration {current_batch_idx}."
            )

            t_start = time.time()
            eps_per_dp_group = []
            batch_shapes_per_dp_group = []
            indices_per_dp_group = []
            best_stats_per_dp_group = []
            best_rcs_per_dp_group = []
            best_schs_per_dp_group = []
            if hasattr(dataset, "_dynapipe_dummy_ep"):
                # use dummy EP for debugging
                (
                    _,
                    indices,
                    exec_plans,
                    best_rc,
                    best_stats,
                ) = dataset._dynapipe_dummy_ep
                eps_per_dp_group = [exec_plans] * dataopt.dp_size
                indices_per_dp_group = [indices] * dataopt.dp_size
                best_stats_per_dp_group = [best_stats] * dataopt.dp_size
                best_rcs_per_dp_group = [best_rc] * dataopt.dp_size
            else:
                # generate microbatches
                t_gen_batch_start = time.time()
                (
                    obj_values,
                    indices,
                    mb_split_memory_type,
                    mb_split_rc_type,
                    mem_stats,
                ) = dataopt.generate_microbatches(
                    input_seqlens,
                    available_rc_types=limit_rc_type,
                    decoder_sample_sequence_lengths=target_seqlens
                    if not input_only
                    else None,
                    partition_method=partition_method,
                    token_based_partition_mb_tokens=token_based_partition_mbs,
                    disable_tsp=disable_tsp,
                    bottleneck_tsp=False,
                    enable_packing=enable_packing,
                )
                if indices is None:
                    # no valid microbatches found
                    worker_data.logger.error(
                        "Failed to generate microbatches for "
                        f"iteration {current_batch_idx}. "
                        "Consider using other parallelization scheme."
                    )
                    raise RuntimeError("Failed to generate microbatches.")
                (avail_mem, model_state, per_mb_memory_limit) = mem_stats
                t_gen_batch_end = time.time()
                # debug only
                peak_memory_consumption = max(obj_values[-1])
                peak_memory_consumption += model_state

                peak_stored_memory_consumption = max(obj_values[-2])
                peak_stored_memory_consumption *= (
                    dataopt.n_executors
                    if mb_split_memory_type == "preferred"
                    else 1
                )
                peak_stored_memory_consumption += model_state
                worker_data.logger.debug(
                    "Micro-batch generation for iteration "
                    "{} took {} seconds. Using memory limit type {} "
                    "and assume recomputation {}, "
                    "estimated peak memory before last layer: {} MB, "
                    "stored peak memory: {} MB".format(
                        current_batch_idx,
                        t_gen_batch_end - t_gen_batch_start,
                        mb_split_memory_type,
                        mb_split_rc_type,
                        peak_memory_consumption,
                        peak_stored_memory_consumption,
                    )
                )

                for dp_group_idx, per_dp_group_indices in enumerate(indices):
                    _, batch_shapes = _collate_samples(
                        batch,
                        per_dp_group_indices,
                        pack_fn,
                        lambda *args: None,
                        lambda *args: None,
                        round_seqlen_multiple=worker_data.round_seqlen_multiple,  # noqa: E501
                        seqlen_offset=worker_data.seqlen_offset,
                        encoder_key=encoder_key,
                        decoder_key=decoder_key,
                    )
                    t_gen_sch_start = time.time()
                    # generate execution plans
                    (
                        exec_plans,
                        _,
                        best_stats,
                        best_rc,
                        best_schedule,
                    ) = exec_planner.generate_execution_plan(
                        batch_shapes,
                        schedule_method=schedule_method,
                        disable_permute_microbatches=disable_mb_permutation,
                        disable_scheduler_memory_limit=disable_scheduler_memory_limit,  # noqa: E501
                        limit_rc_type=limit_rc_type,
                        current_batch_idx=current_batch_idx,
                    )
                    t_gen_sch_end = time.time()
                    worker_data.logger.debug(
                        "Schedule generation for DP group {} "
                        "iteration {} took {} seconds.".format(
                            dp_group_idx,
                            current_batch_idx,
                            t_gen_sch_end - t_gen_sch_start,
                        )
                    )
                    # reorder microbatches based on permutation
                    perm = best_stats[0]
                    per_dp_group_indices = [
                        per_dp_group_indices[i] for i in perm
                    ]
                    if DEBUG_USE_DUMMY_EP:
                        dataset._dynapipe_dummy_ep = (
                            None,
                            per_dp_group_indices,
                            exec_plans,
                            best_rc,
                            best_stats,
                        )
                    eps_per_dp_group.append(exec_plans)
                    indices_per_dp_group.append(per_dp_group_indices)
                    batch_shapes_per_dp_group.append(batch_shapes)
                    best_stats_per_dp_group.append(best_stats)
                    best_rcs_per_dp_group.append(best_rc)
                    best_schs_per_dp_group.append(best_schedule)

            if DEBUG_DUMP_EP_STATS:
                import pickle

                exec_plan_dir = os.path.join(
                    DEBUG_DUMP_EP_PREFIX, "per_iter_exec_plans"
                )
                # save execution plans
                if not os.path.exists(exec_plan_dir):
                    os.makedirs(exec_plan_dir, exist_ok=True)
                for i, exec_plans in enumerate(eps_per_dp_group):
                    with open(
                        os.path.join(
                            exec_plan_dir,
                            "dpg{}_{}.pkl".format(i, current_batch_idx),
                        ),
                        "wb",
                    ) as f:
                        pickle.dump(exec_plans, f)
                # save original sequence lengths in the batch
                orig_seqlens_dir = os.path.join(
                    DEBUG_DUMP_EP_PREFIX, "orig_seq_lens"
                )
                if not os.path.exists(orig_seqlens_dir):
                    os.makedirs(orig_seqlens_dir, exist_ok=True)
                with open(
                    os.path.join(
                        orig_seqlens_dir,
                        "batch_{}.pkl".format(current_batch_idx),
                    ),
                    "wb",
                ) as f:
                    pickle.dump((input_seqlens, target_seqlens), f)
                # save computed microbatch shapes
                per_iter_mb_shapes_dir = os.path.join(
                    DEBUG_DUMP_EP_PREFIX, "per_iter_mb_shapes"
                )
                if not os.path.exists(per_iter_mb_shapes_dir):
                    os.makedirs(per_iter_mb_shapes_dir, exist_ok=True)
                for i, batch_shapes in enumerate(batch_shapes_per_dp_group):
                    with open(
                        os.path.join(
                            per_iter_mb_shapes_dir,
                            "dpg{}_{}.pkl".format(i, current_batch_idx),
                        ),
                        "wb",
                    ) as f:
                        pickle.dump(batch_shapes, f)
                # save memory estimation and simulated traces
                estimated_memory_dir = os.path.join(
                    DEBUG_DUMP_EP_PREFIX, "estimated_memory"
                )
                if not os.path.exists(estimated_memory_dir):
                    os.makedirs(estimated_memory_dir, exist_ok=True)
                simulated_traces_dir = os.path.join(
                    DEBUG_DUMP_EP_PREFIX, "per_iter_simulated_traces"
                )
                if not os.path.exists(simulated_traces_dir):
                    os.makedirs(simulated_traces_dir, exist_ok=True)
                for i, best_stats in enumerate(best_stats_per_dp_group):
                    with open(
                        os.path.join(
                            estimated_memory_dir,
                            "estimated_memory_stats_{}.txt".format(
                                worker_info.id
                            ),
                        ),
                        "a",
                    ) as f:
                        f.write(
                            "{} : {} : {}\n".format(
                                str(i),
                                str(current_batch_idx),
                                str(best_stats[1]),
                            )
                        )
                    with open(
                        os.path.join(
                            simulated_traces_dir,
                            "dpg{}_{}.json".format(i, current_batch_idx),
                        ),
                        "w",
                    ) as f:
                        json.dump(best_stats[-1], f)

            worker_data.logger.debug(
                "Finished generating EP for iteration "
                f"{current_batch_idx}, "
                f"used schedules: {best_schs_per_dp_group}, "
                f"recomputations: {best_rcs_per_dp_group}."
            )
            for dp_group_idx, eps in enumerate(eps_per_dp_group):
                serialized = serialize_list_of_eps(eps)
                indices = indices_per_dp_group[dp_group_idx]
                bytes_as_str = serialized.decode("iso-8859-1")
                worker_data.logger.debug(
                    f"Pushing DPG {dp_group_idx} "
                    f"EP {current_batch_idx} to local store."
                )
                # put execution plan into kv store
                _put_to_shared_kv_store(
                    kv_store,
                    f"dpg{dp_group_idx}_" + indices_key,
                    json.dumps(indices),
                    logger=worker_data.logger,
                )
                _put_to_shared_kv_store(
                    kv_store,
                    f"dpg{dp_group_idx}_" + ep_key,
                    bytes_as_str,
                    logger=worker_data.logger,
                )
            worker_data.logger.debug(
                "Successfully pushed EP "
                f"{current_batch_idx} to shared kv store."
            )
            t_end = time.time()
            worker_data.logger.debug(
                "EP generation for iteration {} took {} seconds.".format(
                    current_batch_idx, t_end - t_start
                )
            )

        except Exception as e:
            # explicitly log exception here since it will be swallowed by
            # multiprocessing
            worker_data.logger.error(
                "Exception in worker process: {}".format(e)
            )
            worker_data.logger.error(traceback.format_exc())
            raise e
        return None

    return _collate_fn


def get_collate_fn(
    pack_fn,
    constructor_fn,
    collate_fn=torch.utils.data.default_collate,
    encoder_key="text_enc",
    decoder_key="text_dec",
):
    # pack_fn takes a list of tensors to pack and returns a single packed
    #     tensor along with any additional information that will be
    #     fed into constructor_fn)
    # constructor_fn has the following signiture:
    #     (encoder_input, encoder_extra,
    #      decoder_input, decoder_extra,
    #      encoder_seqlen, decoder_seqlen)
    #     where input and extra are the output of pack_fn
    #     Its outputs will be fed into collate_fn and will be returned
    # collate_fn defaults to torch.utils.data.dataloader.default_collate

    def _collate_fn(batch):
        # get states from variables set in worker_init_fn
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        worker_data: DataloaderWorkerData = dataset.worker_data
        try:
            dp_rank = worker_data.dp_rank
            pp_rank = worker_data.pp_rank
            pp_size = worker_data.pp_size
            virtual_pp_rank = worker_data.virtual_pp_rank
            virtual_pp_size = worker_data.virtual_pp_size

            kv_store = worker_data.kv_store
            kv_buffer_size = worker_data.kv_buffer_size
            # processed_batches is local to each worker since dataset
            # is replicated
            processed_batches = worker_data.processed_batches

            # pytorch assigns batch to workers in round-robin fashion
            # so we can use worker_info.id to determine the current batch index
            # TODO: find a better way to do this since it is not documented
            #       and depends on implemtation
            current_batch_idx = (
                worker_info.num_workers * processed_batches + worker_info.id
            )
            buffer_slot = current_batch_idx % kv_buffer_size
            indices_key = f"dpg{dp_rank}_ind_{buffer_slot}"
            ep_key = f"dpg{dp_rank}_ep_{buffer_slot}"

            worker_data.logger.debug(
                "Waiting for EP for iteration {}.".format(current_batch_idx)
            )
            # there are virtual_pp_size * pp_size readers per dp group
            # index is assigned pp rank first, then virtual pp rank
            n_readers = virtual_pp_size * pp_size
            reader_idx = pp_rank * virtual_pp_size + virtual_pp_rank
            indices = json.loads(
                _get_from_shared_kv_store(
                    kv_store,
                    indices_key,
                    reader_idx=reader_idx,
                    n_total_readers=n_readers,
                    decode=False,
                    logger=worker_data.logger,
                )
            )
            serialized_ep = _get_from_shared_kv_store(
                kv_store,
                ep_key,
                reader_idx=reader_idx,
                n_total_readers=n_readers,
                logger=worker_data.logger,
            ).encode("iso-8859-1")
            worker_data.logger.debug(
                "Got data for iteration {}.".format(current_batch_idx)
            )
            # received ep is actually a list of eps for each pp_rank
            execution_plans = deserialize_list_of_eps(serialized_ep)
            exec_plan = execution_plans[pp_rank]

            microbatches, _ = _collate_samples(
                batch,
                indices,
                pack_fn,
                constructor_fn,
                collate_fn,
                round_seqlen_multiple=worker_data.round_seqlen_multiple,
                seqlen_offset=worker_data.seqlen_offset,
                encoder_key=encoder_key,
                decoder_key=decoder_key,
            )
            worker_data.logger.debug(
                "Generated data for iteration {}.".format(current_batch_idx)
            )
            # increment processed batches
            worker_info.dataset.worker_data.processed_batches += 1
        except Exception as e:
            # explicitly log exception here since it will be swallowed by
            # multiprocessing
            worker_data.logger.error(
                "Exception in worker process: {}".format(e)
            )
            worker_data.logger.error(traceback.format_exc())
            raise e
        return microbatches, exec_plan

    return _collate_fn


class DataloaderArgs:
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        sampler,
        batch_sampler,
        num_workers,
        pack_fn,
        drop_last,
        prefetch_factor,
        persistent_workers,
        *args,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.pack_fn = pack_fn
        self.drop_last = drop_last
        self.args = args
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers


def _preprocessor_poller(
    control_queue: mp.Queue,
    dataloader_args: DataloaderArgs,
    training_spec: TrainingSpec,
    node_rank,
    node_size,
    is_kv_host,
    assigned_iters_per_node,
    encoder_key,
    decoder_key,
):
    # this process runs only once per physical node, responsible for
    # initializing and polling the preprocessor processes
    logger = create_logger(
        "poller",
        prefix="Poller",
        log_file=f"poller/poller_r{node_rank}.log",
    )
    kv_buffer_size = assigned_iters_per_node * node_size
    logger.debug("Starting poller process.")
    if is_kv_host:
        logger.debug("Starting kvstore server.")
        kv_store, _, _ = _init_kv_store(is_master=True, logger=logger)
        # set up kv_store values for workers
        kv_store.set(
            KVStoreMetaKeys.DP_SIZE,
            str(training_spec.data_parallel_size),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.TP_SIZE,
            str(training_spec.tensor_parallel_size),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.PP_SIZE,
            str(training_spec.pipeline_parallel_size),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.VIRTUAL_PP_SIZE,
            str(training_spec.n_chunks_per_device),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.ZERO_STAGE,
            str(training_spec.zero_stage),
            logger=logger,
        )
        kv_store.set(KVStoreMetaKeys.NODE_SIZE, str(node_size), logger=logger)
        kv_store.set(
            KVStoreMetaKeys.MODEL_SPEC,
            training_spec.model_spec.serialize().decode("iso-8859-1"),
        )
        kv_store.set(
            KVStoreMetaKeys.N_EXECS,
            str(training_spec.n_executors),
            logger=logger,
        )

        kv_store.set(
            KVStoreMetaKeys.N_LAYERS_PER_STAGE,
            str(training_spec.n_layers_per_stage),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.N_CHUNKS_PER_DEVICE,
            str(training_spec.n_chunks_per_device),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.DEVICE_MEMORY_LIMIT,
            str(training_spec.device_memory_limit),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.PARTITION_METHOD,
            training_spec.partition_algo,
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.TOKEN_BASED_PARTITION_MBS,
            str(training_spec.token_based_partition_mbs),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.DISABLE_TSP,
            str(training_spec.disable_tsp),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.SCHEDULE_METHOD,
            str(training_spec.schedule_method),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.DISABLE_MB_PERMUTATION,
            str(training_spec.disable_mb_permutation),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.DISABLE_SCHEDULER_MEMORY_LIMIT,
            str(training_spec.disable_scheduler_memory_limit),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.ENABLE_PACKING,
            str(training_spec.enable_packing),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.PER_MB_MEM_FRAC,
            str(training_spec.per_mb_memory_fraction),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.CLUSTER_SPEC, training_spec.cluster_spec.dumps()
        )
        kv_store.set(
            KVStoreMetaKeys.DEV_ASSIGNMENT,
            ",".join([str(x) for x in training_spec.device_assignment]),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.KV_BUFFER_SIZE,
            str(kv_buffer_size),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.ROUND_SEQLEN_MULT,
            str(training_spec.round_seqlen_multiple),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.ASSIGNED_ITER_PER_NODE,
            str(assigned_iters_per_node),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.SEQLEN_OFFSET,
            str(training_spec.seqlen_offset),
            logger=logger,
        )
        kv_store.set(
            KVStoreMetaKeys.MODEL_TYPE,
            training_spec.model_type,
            logger=logger,
        )

        # init all ack keys
        for dp_idx in range(training_spec.data_parallel_size):
            for i in range(kv_buffer_size):
                kv_store.add(f"dpg{dp_idx}_ind_{i}_ack", 1)
                kv_store.add(f"dpg{dp_idx}_ep_{i}_ack", 1)
            # set reader ack keys
            for i in range(
                training_spec.pipeline_parallel_size
                * training_spec.n_chunks_per_device
            ):
                for buffer_idx in range(kv_buffer_size):
                    for key in ["ind", "ep"]:
                        kv_store.add(
                            f"dpg{dp_idx}_{key}_{buffer_idx}_r{i}_ack", 1
                        )
    # create dataloader
    dataset = dataloader_args.dataset
    # add worker data to it
    preprocess_worker_data = PreprocessingWorkerData(
        node_rank=node_rank, profile_path=training_spec.cm_path
    )
    dataset.worker_data = preprocess_worker_data
    prefetch_data_loader = PTDataLoader(
        dataset,
        dataloader_args.batch_size,
        dataloader_args.shuffle,
        dataloader_args.sampler,
        dataloader_args.batch_sampler,
        dataloader_args.num_workers,
        get_preprocessing_collate_fn(
            dataloader_args.pack_fn,
            limit_rc_type=training_spec.limit_rc_type,
            encoder_key=encoder_key,
            decoder_key=decoder_key,
        ),
        False,
        dataloader_args.drop_last,
        0,
        _preprocessing_worker_init_fn,
        *dataloader_args.args,
        prefetch_factor=dataloader_args.prefetch_factor,
        persistent_workers=dataloader_args.persistent_workers,
    )
    # start prefetching
    for idx, _ in enumerate(prefetch_data_loader):
        # try to see if we receive a exit signal
        logger.debug(f"Poller polled for iteration {idx}.")
        try:
            item = control_queue.get_nowait()
            if item == "exit":
                logger.debug("Got exit signal! Poller exiting.")
                break
        except Empty:
            pass
    if is_kv_host:
        kv_store.set(KVStoreMetaKeys.N_ITERS, str(idx + 1), logger=logger)
    logger.debug("No more data to prefetch. Poller exiting.")


def get_num_iters():
    global _kvstore_handle
    if _kvstore_handle is None:
        kv_store, _, _ = _init_kv_store(is_master=False)
        _kvstore_handle = kv_store
    n_iters = _kvstore_handle.get(KVStoreMetaKeys.N_ITERS, wait=False)
    if n_iters is None:
        return None
    return int(n_iters.decode())


class DynaPipeDataLoader:
    """
    A wrapper around PyTorch DataLoader, which automatically generates
    execution plans for each batch and returns the execution plan along
    with the batch of data.

    On local rank 0 of each node, it starts a poller process which creates
    a Torch DataLoader wrapping the user provided dataset and prefetches data.
    Each worker in the Torch DataLoader is instructed to compute the execution
    plan for assigned batches and pushes the execution plan to a shared kv
    store. On the node where kv store is hosted, it is also responsible for kv
    store initialization.

    On all ranks, it creates a torch DataLoader wrapping the user dataset.
    In addition to the data, it also returns the execution plan for the batch,
    fetched from the shared kv store.
    """

    def __init__(
        self,
        training_spec: TrainingSpec,
        dataset,
        pack_fn,
        constructor_fn,
        is_kv_host,
        node_rank=0,
        node_local_rank=0,
        node_size=1,
        dp_rank=0,
        pp_rank=0,
        virtual_pp_rank=0,
        collate_fn=torch.utils.data.default_collate,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        num_preprocess_workers=64,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        *args,
        prefetch_factor=2,
        persistent_workers=False,
        encoder_key="text_enc",
        decoder_key="text_dec",
    ):
        self.node_rank = node_rank
        self.node_local_rank = node_local_rank
        self.dp_rank = dp_rank
        self.pp_rank = pp_rank
        assert pp_rank < training_spec.pipeline_parallel_size, (
            f"pp_rank ({pp_rank}) should be smaller than "
            f"pipeline_parallel_size ({training_spec.pipeline_parallel_size})"
            "in training_spec."
        )
        # virtual rank is similar to virtual_pipeline_model_parallel_rank
        # in Megatron-LM, where multiple data loaders are created for
        # interleaved scheduling.
        self.virtual_pp_rank = virtual_pp_rank
        assert virtual_pp_rank < training_spec.n_chunks_per_device, (
            f"virtual_pp_rank ({virtual_pp_rank}) should be smaller than "
            f"n_chunks_per_device ({training_spec.n_chunks_per_device})"
            "in training_spec, calculated from device assignment."
        )
        # create queues
        self.poller_control_queue = mp.Queue()
        self.num_preprocess_workers = num_preprocess_workers

        if self.node_local_rank == 0 and self.virtual_pp_rank == 0:
            dataloader_args = DataloaderArgs(
                dataset,
                batch_size,
                shuffle,
                sampler,
                batch_sampler,
                num_preprocess_workers,
                pack_fn,
                drop_last,
                prefetch_factor,
                persistent_workers,
                *args,
            )
            assigned_iters_per_node = num_preprocess_workers * prefetch_factor
            self.poller_process = mp.Process(
                target=_preprocessor_poller,
                args=(
                    self.poller_control_queue,
                    dataloader_args,
                    training_spec,
                    node_rank,
                    node_size,
                    is_kv_host,
                    assigned_iters_per_node,
                    encoder_key,
                    decoder_key,
                ),
            )
            self.poller_process.start()
        # create torch dataloader
        worker_data = DataloaderWorkerData(
            dp_rank=dp_rank,
            pp_rank=pp_rank,
            virtual_pp_rank=virtual_pp_rank,
        )
        dataset.worker_data = worker_data
        self.data_loader = PTDataLoader(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            get_collate_fn(
                pack_fn,
                constructor_fn,
                collate_fn,
                encoder_key=encoder_key,
                decoder_key=decoder_key,
            ),
            pin_memory,
            drop_last,
            timeout,
            _worker_init_fn,
            *args,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    def __del__(self):
        if hasattr(self, "poller_process"):
            if self.poller_process.is_alive():
                self.poller_control_queue.put("exit")
                self.poller_process.join()

    def __iter__(self):
        yield from self.data_loader

    def __len__(self):
        return self.data_loader.__len__()

    def check_worker_number_rationality(self):
        if self.num_preprocess_workers == 0:
            logger.warn(
                "DynaPipeDataLoader should be used with a large number of "
                "preprocessing workers to achieve good performance. "
                "Current setting is num_preprocess_workers=0."
            )
        self.data_loader.check_worker_number_rationality()
