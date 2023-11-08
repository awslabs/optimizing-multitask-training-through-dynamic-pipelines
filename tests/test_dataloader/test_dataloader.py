# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Note: this test requires torch
# to run this test, exec:
# If running hanging tests or multi-node tests:
# DYNAPIPE_DEBUG=DEBUG DYNAPIPE_LOGGING_DEBUG_DIR=./test_debug \
# torchrun --standalone --nnodes=1 --nproc_per_node=4 test_dataloader.py
# Others:
# DYNAPIPE_DEBUG=DEBUG DYNAPIPE_LOGGING_DEBUG_DIR=./test_debug \
# torchrun --standalone --nnodes=1 --nproc_per_node=2 test_dataloader.py

import os

import pytest
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from dynapipe.model import TransformerModelSpec, get_uniform_cluster
from dynapipe.pipe.data_loader import DynaPipeDataLoader, TrainingSpec
from dynapipe.pipe.instructions import ExecutionPlan, ForwardPass

torch.manual_seed(42)


@pytest.fixture(scope="module", autouse=True)
def init_torch_distributed():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("gloo")


class DummyDataset(Dataset):
    def __init__(self, size, inputs_only=False):
        self.size = size
        torch.manual_seed(42)
        # pre-generate all data
        self.enc_seqlen = []
        self.dec_seqlen = []
        self.data = []
        for _ in range(size):
            enc_seqlen, dec_seqlen = torch.randint(24, 512, (2,))
            self.enc_seqlen.append(enc_seqlen)
            if not inputs_only:
                self.dec_seqlen.append(dec_seqlen)
                result = {
                    "text_enc": list(
                        torch.randint(0, 100, (enc_seqlen,)).numpy()
                    ),
                    "text_dec": list(
                        torch.randint(0, 100, (dec_seqlen,)).numpy()
                    ),
                }
            else:
                result = {
                    "text": list(torch.randint(0, 100, (enc_seqlen,)).numpy()),
                }
            self.data.append(result)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]


def dummy_pack_fn(tensors):
    # (input, extra)
    if len(tensors) == 0:
        return [], 0
    if isinstance(tensors[0], list):
        concated_list = []
        for t in tensors:
            concated_list.extend(t)
        return concated_list, 0
    return torch.cat(tensors, dim=0), 0


def dummy_constructor_fn(
    encoder_input,
    encoder_extra,
    decoder_input,
    decoder_extra,
    encoder_seqlen,
    decoder_seqlen,
):
    encoder_padding_len = encoder_seqlen - len(encoder_input)
    if decoder_input is not None:
        decoder_padding_len = decoder_seqlen - len(decoder_input)
    encoder_input = torch.tensor(encoder_input, dtype=torch.long)
    if decoder_input is not None:
        decoder_input = torch.tensor(decoder_input, dtype=torch.long)
    encoder_padded = torch.cat(
        [
            encoder_input,
            torch.zeros(
                encoder_padding_len,
                dtype=encoder_input.dtype,
                device=encoder_input.device,
            ),
        ],
        dim=0,
    )
    if decoder_input is not None:
        decoder_padded = torch.cat(
            [
                decoder_input,
                torch.zeros(
                    decoder_padding_len,
                    dtype=decoder_input.dtype,
                    device=decoder_input.device,
                ),
            ],
            dim=0,
        )
        return {
            "text_enc": encoder_padded,
            "text_dec": decoder_padded,
        }
    else:
        return {
            "text": encoder_padded,
        }


def get_mb_shape_from_ep(ep: ExecutionPlan):
    fw_shapes = []
    for instr in ep.instructions:
        if isinstance(instr, ForwardPass):
            fw_shapes.append(instr.buffer_shapes)
    return fw_shapes


def test_joint_data_loader(inputs_only=False):
    cluster_spec = get_uniform_cluster(2)
    if inputs_only:
        train_spec = TrainingSpec(
            "test_cm.pkl",
            cluster_spec,
            TransformerModelSpec(8, 0, 1024, 128, 65536, 128),
            1,
            2,
            0,
            [0, 0, 0, 0, 1, 1, 1, 1],
            800000,  # ignore memory limit for this test
            prefetch_buffer_size=2,
        )
    else:
        train_spec = TrainingSpec(
            "test_cm.pkl",
            cluster_spec,
            TransformerModelSpec(4, 4, 1024, 128, 65536, 128),
            1,
            2,
            0,
            [0, 0, 0, 0, 1, 1, 1, 1],
            800000,  # ignore memory limit for this test
            prefetch_buffer_size=2,
            model_type="t5",
        )
    rank = dist.get_rank()
    is_kv_host = rank == 0
    data_loader = DynaPipeDataLoader(
        train_spec,
        DummyDataset(256 * 10, inputs_only=inputs_only),
        pack_fn=dummy_pack_fn,
        constructor_fn=dummy_constructor_fn,
        is_kv_host=is_kv_host,
        node_rank=0,
        node_local_rank=rank,
        dp_rank=0,
        pp_rank=rank,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        num_preprocess_workers=2,
        pin_memory=True,
        encoder_key="text_enc" if not inputs_only else "text",
        decoder_key="text_dec" if not inputs_only else None,
    )
    batch_idx = 0
    for batch, ep in data_loader:
        if rank == 0:
            assert batch is not None
            ep_shapes = get_mb_shape_from_ep(ep)
            assert len(ep_shapes) == len(batch)
            for microbatch, ep_shape in zip(batch, ep_shapes):
                if not inputs_only:
                    enc_seqlen, dec_seqlen = (
                        microbatch["text_enc"].shape[1],
                        microbatch["text_dec"].shape[1],
                    )
                    enc_mbs, dec_mbs = (
                        microbatch["text_enc"].shape[0],
                        microbatch["text_dec"].shape[0],
                    )
                else:
                    enc_seqlen = microbatch["text"].shape[1]
                    enc_mbs = microbatch["text"].shape[0]
                    dec_mbs = enc_mbs
                    dec_seqlen = 0
                assert enc_mbs == dec_mbs
                # encoder only have ep_shape size 1
                assert len(ep_shape) == 1
                # test shape rounding
                assert enc_seqlen % 8 == 0
                assert dec_seqlen % 8 == 0
                mbs_from_ep = ep_shape[0][0]
                enc_seqlen_from_ep = ep_shape[0][1]
                assert mbs_from_ep == enc_mbs
                assert enc_seqlen_from_ep == enc_seqlen
                # get enc and decoder len from rank 1
                mbs_rank1_ep_tensor = torch.empty(1, dtype=torch.int64)
                encoder_ep_seqlen_tensor = torch.empty(1, dtype=torch.int64)
                decoder_ep_seqlen_tensor = torch.empty(1, dtype=torch.int64)
                dist.recv(tensor=mbs_rank1_ep_tensor, src=1)
                dist.recv(tensor=encoder_ep_seqlen_tensor, src=1)
                dist.recv(tensor=decoder_ep_seqlen_tensor, src=1)
                mbs_rank1_ep = mbs_rank1_ep_tensor.item()
                encoder_ep_seqlen = encoder_ep_seqlen_tensor.item()
                decoder_ep_seqlen = decoder_ep_seqlen_tensor.item()
                assert mbs_rank1_ep == enc_mbs
                assert dec_seqlen == decoder_ep_seqlen
                assert enc_seqlen == encoder_ep_seqlen
            print(f"batch {batch_idx} passed")
            batch_idx += 1
        else:
            assert batch is not None
            assert ep is not None
            ep_shapes = get_mb_shape_from_ep(ep)
            for ep_shape in ep_shapes:
                if not inputs_only:
                    assert len(ep_shape) == 2
                    assert ep_shape[0][0] == ep_shape[1][0]
                    mbs_from_ep = ep_shape[0][0]
                    enc_seqlen_from_ep = ep_shape[0][1]
                    dec_seqlen_from_ep = ep_shape[1][1]
                else:
                    assert len(ep_shape) == 1
                    mbs_from_ep = ep_shape[0][0]
                    enc_seqlen_from_ep = ep_shape[0][1]
                    dec_seqlen_from_ep = 0
                mbs_tensor = torch.tensor(mbs_from_ep, dtype=torch.int64)
                enc_seqlen_tensor = torch.tensor(
                    enc_seqlen_from_ep, dtype=torch.int64
                )
                dec_seqlen_tensor = torch.tensor(
                    dec_seqlen_from_ep, dtype=torch.int64
                )
                dist.send(tensor=mbs_tensor, dst=0)
                dist.send(tensor=enc_seqlen_tensor, dst=0)
                dist.send(tensor=dec_seqlen_tensor, dst=0)
    dist.barrier()


def test_joint_data_loader_hanging():
    cluster_spec = get_uniform_cluster(4)
    train_spec = TrainingSpec(
        "test_cm.pkl",
        cluster_spec,
        TransformerModelSpec(4, 4, 1024, 128, 65536, 128),
        1,
        4,
        0,
        [0, 0, 1, 1, 2, 2, 3, 3],
        800000,  # ignore memory limit for this test
        prefetch_buffer_size=32,
        model_type="t5",
    )
    rank = dist.get_rank()
    data_loader = DynaPipeDataLoader(
        train_spec,
        DummyDataset(256 * 1000),
        pack_fn=dummy_pack_fn,
        constructor_fn=dummy_constructor_fn,
        is_kv_host=rank == 0,
        node_rank=0,
        node_local_rank=rank,
        dp_rank=0,
        pp_rank=rank,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        num_preprocess_workers=32,
        pin_memory=True,
    )
    for idx, (batch, ep) in enumerate(data_loader):
        if rank == 0:
            print("Progress: Iteration {}".format(idx))
        dist.barrier()
    dist.barrier()


def test_joint_data_loader_multiple_nodes():
    cluster_spec = get_uniform_cluster(4)
    train_spec = TrainingSpec(
        "test_cm.pkl",
        cluster_spec,
        TransformerModelSpec(4, 4, 1024, 128, 65536, 128),
        1,
        4,
        0,
        [0, 0, 1, 1, 2, 2, 3, 3],
        800000,  # ignore memory limit for this test
        prefetch_buffer_size=32,
        model_type="t5",
    )
    rank = dist.get_rank()
    data_loader = DynaPipeDataLoader(
        train_spec,
        DummyDataset(256 * 1000),
        pack_fn=dummy_pack_fn,
        constructor_fn=dummy_constructor_fn,
        is_kv_host=rank == 0,
        node_rank=rank // 2,
        node_local_rank=rank % 2,
        node_size=2,
        dp_rank=0,
        pp_rank=rank,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        num_preprocess_workers=32,
        pin_memory=True,
    )
    for idx, (batch, ep) in enumerate(data_loader):
        if rank == 0:
            print("Progress: Iteration {}".format(idx))
        dist.barrier()
    dist.barrier()


def test_joint_data_loader_with_virtual_ranks():
    cluster_spec = get_uniform_cluster(2)
    train_spec = TrainingSpec(
        "test_cm.pkl",
        cluster_spec,
        TransformerModelSpec(4, 4, 1024, 128, 65536, 128),
        1,
        2,
        0,
        [0, 0, 1, 1, 0, 0, 1, 1],
        800000,  # ignore memory limit for this test
        prefetch_buffer_size=2,
        model_type="t5",
    )
    rank = dist.get_rank()
    data_loader_0 = DynaPipeDataLoader(
        train_spec,
        DummyDataset(256 * 10),
        pack_fn=dummy_pack_fn,
        constructor_fn=dummy_constructor_fn,
        is_kv_host=True if rank == 0 else False,
        node_rank=0,
        node_local_rank=rank,
        dp_rank=0,
        pp_rank=rank,
        virtual_pp_rank=0,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        num_preprocess_workers=2,
        pin_memory=True,
    )
    data_loader_1 = DynaPipeDataLoader(
        train_spec,
        DummyDataset(256 * 10),
        pack_fn=dummy_pack_fn,
        constructor_fn=dummy_constructor_fn,
        is_kv_host=False,
        node_rank=0,
        node_local_rank=rank,
        node_size=1,
        dp_rank=0,
        pp_rank=rank,
        virtual_pp_rank=1,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    for it, ((batch0, ep0), (batch1, ep1)) in enumerate(
        zip(data_loader_0, data_loader_1)
    ):
        assert len(batch0) == len(
            batch1
        ), "batch size mismatch ({}, {}) at iter {}".format(
            len(batch0), len(batch1), it
        )
        for mb0, mb1 in zip(batch0, batch1):
            assert torch.equal(mb0["encoder_input"], mb1["encoder_input"])
            assert torch.equal(mb0["decoder_input"], mb1["decoder_input"])
        assert ep0 == ep1
    dist.barrier()


os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
torch.distributed.init_process_group("gloo")
# test hanging issue
# test_joint_data_loader_hanging()
# test multi-node preprocessing
# test_joint_data_loader_multiple_nodes()
# test without virtual ranks
test_joint_data_loader(inputs_only=True)
# test with virtual ranks
# test_joint_data_loader_with_virtual_ranks()
