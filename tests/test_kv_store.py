# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Note: this test requires torch
# to run this test, exec:
# DYNAPIPE_DEBUG=DEBUG DYNAPIPE_LOGGING_DEBUG_DIR=./test_debug \
# torchrun --standalone --nnodes=1 --nproc_per_node=1 test_kv_store.py

import multiprocessing as mp

from dynapipe.pipe.data_loader import (
    _get_from_shared_kv_store,
    _init_kv_store,
    _put_to_shared_kv_store,
)


def _producer_process(max_iters, buffer_size=32):
    try:
        kv_store, _, _ = _init_kv_store(is_master=True)
        # set all ack keys
        for i in range(buffer_size):
            kv_store.set(f"key_{i}_ack".format(i), "1")
            kv_store.set(f"key_{i}_r0_ack".format(i), "1")
        for i in range(max_iters):
            key = "key_{}".format(i % buffer_size)
            payload = str(i)
            _put_to_shared_kv_store(kv_store, key, payload)
            print("[producer] put key: {}".format(key), flush=True)
        import time

        time.sleep(2)
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise e


def _consumer_process(max_iters, buffer_size=32):
    try:
        kv_store, _, _ = _init_kv_store(is_master=False)
        for i in range(max_iters):
            key = "key_{}".format(i % buffer_size)
            payload = _get_from_shared_kv_store(
                kv_store, key, 0, 1, decode=True
            )
            assert payload == str(i)
            print("[consumer] got key: {}".format(key), flush=True)
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise e


def test_kv_store():
    max_iters = 1000
    buffer_size = 32
    producer = mp.Process(
        target=_producer_process, args=(max_iters, buffer_size)
    )
    consumer = mp.Process(
        target=_consumer_process, args=(max_iters, buffer_size)
    )
    producer.start()
    consumer.start()
    consumer.join()
    producer.join()


if __name__ == "__main__":
    test_kv_store()
