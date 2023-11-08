# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys
from logging import Handler
from typing import List

_logging_lvl_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
_logging_lvl_from_env = os.environ.get("DYNAPIPE_DEBUG", "INFO")
if _logging_lvl_from_env not in _logging_lvl_map:
    raise ValueError(
        f"Invalid logging level from env detected: {_logging_lvl_from_env}. "
        f"Valid options are {list(_logging_lvl_map.keys())}"
    )
_default_logging_level = _logging_lvl_map[_logging_lvl_from_env]

if _default_logging_level == logging.DEBUG:
    debug_dir = os.environ.get("DYNAPIPE_LOGGING_DEBUG_DIR", None)
    if debug_dir is None:
        raise ValueError(
            "DYNAPIPE_LOGGING_DEBUG_DIR must be set when "
            "DYNAPIPE_DEBUG is set to DEBUG"
        )
    # create output dir for executor logs
    os.makedirs(debug_dir, exist_ok=True)
    _debug_log_dir = debug_dir
    # create subdirs
    _dataloader_log_dir = os.path.join(_debug_log_dir, "dataloader")
    os.makedirs(_dataloader_log_dir, exist_ok=True)
    _preprocessing_log_dir = os.path.join(_debug_log_dir, "preprocessing")
    os.makedirs(_preprocessing_log_dir, exist_ok=True)
    _executor_log_dir = os.path.join(_debug_log_dir, "executor")
    os.makedirs(_executor_log_dir, exist_ok=True)
    _poller_log_dir = os.path.join(_debug_log_dir, "poller")
    os.makedirs(_poller_log_dir, exist_ok=True)
else:
    # not used
    _debug_log_dir = "./"


# modified from https://stackoverflow.com/a/56944256
class DynaPipeFormatter(logging.Formatter):
    white = "\x1b[38;20m"
    grey = "\x1b[38;5;8m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    color_mapping = {
        logging.DEBUG: grey,
        logging.INFO: white,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red,
    }

    def __init__(self, prefix=None, distributed_rank=None, colored=True):
        self.prefix = prefix
        self.distributed_rank = distributed_rank
        self.colored = colored

    def _get_fmt_colored(self, level):
        color = self.color_mapping[level]
        fmt = (
            self.grey
            + "[%(asctime)s] "
            + self.reset
            + color
            + "[%(levelname)s] "
            + self.reset
            + self.grey
            + "[%(filename)s:%(lineno)d] "
            + self.reset
        )
        fmt += color
        if self.prefix is not None:
            fmt += "[" + self.prefix + "] "
        if self.distributed_rank is not None:
            fmt += "[Rank " + str(self.distributed_rank) + "] "
        fmt += "%(message)s"
        fmt += self.reset
        return fmt

    def _get_fmt(self):
        fmt = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] "
        if self.prefix is not None:
            fmt += "[" + self.prefix + "] "
        if self.distributed_rank is not None:
            fmt += "[Rank " + str(self.distributed_rank) + "] "
        fmt += "%(message)s"
        return fmt

    def format(self, record):
        if self.colored:
            log_fmt = self._get_fmt_colored(record.levelno)
        else:
            log_fmt = self._get_fmt()
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# modified from https://stackoverflow.com/a/51612402
class LoggerWriter(object):
    def __init__(self, writers):
        if not isinstance(writers, (list, tuple)):
            writers = (writers,)
        self._writers = writers
        self._msg: str = ""

    def write(self, message: str):
        self._msg = self._msg + message
        pos = self._msg.find("\n")
        while pos != -1:
            for writer in self._writers:
                writer(self._msg[:pos])
            self._msg = self._msg[pos + 1 :]
            pos = self._msg.find("\n")

    def flush(self):
        if self._msg != "":
            for writer in self._writers:
                writer(self._msg)
            self._msg = ""


# modified from DeepSpeed
# deepspeed/utils/logging.py
def create_logger(
    name=None,
    prefix=None,
    level=_default_logging_level,
    distributed_rank=None,
    log_file=None,
):
    """create a logger
    Args:
        name (str): name of the logger
        level: level of logger
    Raises:
        ValueError is name is None
    """

    if name is None:
        raise ValueError("name for logger cannot be None")

    if level > logging.DEBUG:
        # disable log to file
        log_file = None

    formatter = DynaPipeFormatter(
        prefix=prefix, distributed_rank=distributed_rank, colored=False
    )
    colored_formatter = DynaPipeFormatter(
        prefix=prefix, distributed_rank=distributed_rank, colored=True
    )

    logger_ = logging.getLogger(name)
    # if handler already present, remove it
    if logger_.hasHandlers():
        logger_.handlers.clear()
    logger_.setLevel(level)
    logger_.propagate = False
    handlers: List[Handler] = []
    if log_file is not None:
        full_log_path = os.path.join(_debug_log_dir, log_file)
        handler = logging.FileHandler(filename=full_log_path)
        handler.setFormatter(formatter)
        handler.setLevel(level)
        handlers.append(handler)
        # also log all warnings/errors to stderr
        warn_handler = logging.StreamHandler(stream=sys.__stderr__)
        warn_handler.setLevel(logging.WARNING)
        warn_handler.setFormatter(colored_formatter)
        handlers.append(warn_handler)
        # copy stderr to log file
        sys.stderr = LoggerWriter([logger_.warning])
    else:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(colored_formatter)
        handlers.append(ch)
    for handler in handlers:
        logger_.addHandler(handler)
    return logger_


logger = create_logger(name="DynaPipe", level=_default_logging_level)
