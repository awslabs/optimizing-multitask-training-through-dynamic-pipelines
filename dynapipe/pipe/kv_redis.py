# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import atexit
import datetime
import os
import subprocess
import time

import redis

REDIS_CMD = os.environ.get("DYNAPIPE_REDIS_CMD", "redis-server")
KVREDIS_POLLING_INTERVAL = float(
    os.environ.get("DYNAPIPE_KVREDIS_POLLING_INTERVAL", "0.05")
)
KVREDIS_CONNECT_TIMEOUT = float(
    os.environ.get("DYNAPIPE_KVREDIS_CONNECT_TIMEOUT", 30)
)


class RedisKVStore(object):
    # a blocking redis client
    def __init__(self, host, port, is_master=False):
        self.is_master = is_master
        self.host = host
        self.port = port
        if self.is_master:
            self.server = self._run_redis_server()
        # wait for redis server to start
        t = time.time()
        while True:
            try:
                self.client = redis.Redis(host=host, port=port, db=0)
                self.client.ping()
                break
            except redis.exceptions.ConnectionError:
                time.sleep(KVREDIS_POLLING_INTERVAL)
                if time.time() - t > KVREDIS_CONNECT_TIMEOUT:
                    raise RuntimeError(
                        "WARNING: Cannot connect to KV Server. "
                        "Is DYNAPIPE_KV_HOST and "
                        "DYNAPIPE_KV_PORT set correctly?"
                    )
                continue
        # register cleanup
        atexit.register(self.__del__)

    def __del__(self):
        if self.is_master:
            if self.server.poll() is not None:
                return
            self.server.send_signal(subprocess.signal.SIGINT)
            self.server.wait()

    def _run_redis_server(self):
        # run a redis server
        p = subprocess.Popen(
            [
                REDIS_CMD,
                "--save",
                "",
                "--port",
                str(self.port),
                "--bind",
                str(self.host),
            ],
            shell=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        return p

    def wait(self, keys, timeout=None):
        # wait for a key to be set
        time_start = datetime.datetime.now()
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        while True:
            if self.client.exists(*keys):
                break
            if (
                timeout is not None
                and datetime.datetime.now() - time_start > timeout
            ):
                # match torch kvstore behavior
                raise RuntimeError("Timeout")
            time.sleep(KVREDIS_POLLING_INTERVAL)

    def get(self, key, wait=True):
        if wait:
            self.wait(key)
        return self.client.get(key)

    def set(self, key, value: str, logger=None):
        # match torch kvstore behavior
        value_bytes = value.encode()
        self.client.set(key, value_bytes)
        if logger:
            logger.debug("KVStore: set {} to {}".format(key, value))

    def add(self, key, value: int):
        # match torch kvstore behavior
        return self.client.incr(key, value)

    def delete_key(self, key):
        return self.client.delete(key)
