# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import sys

import pytest

LOGGING_TEST_DIR = "./logger_test"
CREATED_FILES = [
    "test_warning.log",
    "test_stderr_warning.log",
    "test_multiline_stderr_warning.log",
]

os.environ["DYNAPIPE_DEBUG"] = "DEBUG"
os.environ["DYNAPIPE_LOGGING_DEBUG_DIR"] = LOGGING_TEST_DIR

from dynapipe.utils.logger import create_logger  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def prepare_and_cleanup():
    if os.path.exists(LOGGING_TEST_DIR):
        for files_to_remove in CREATED_FILES:
            fn = os.path.join(LOGGING_TEST_DIR, files_to_remove)
            if os.path.exists(fn):
                os.remove(fn)
    yield
    if os.path.exists(LOGGING_TEST_DIR):
        shutil.rmtree(LOGGING_TEST_DIR)


def _get_number_of_lines(string: str):
    return len(string.strip().split("\n"))


def test_logger_warning(capfd):
    logger = create_logger(
        "test_logger", prefix="Warn Test", log_file="test_warning.log"
    )
    logger.warning("This is a warning.")

    _, stderr_contents = capfd.readouterr()
    # stderr should be colored
    assert "\x1b[0m" in stderr_contents
    # stderr should contain one line
    assert _get_number_of_lines(stderr_contents) == 1
    fn = os.path.join(LOGGING_TEST_DIR, "test_warning.log")
    with open(fn, "r") as f:
        log_contents = f.read()
    # log file should not be colored
    assert "\x1b[0m" not in log_contents
    # log file should contain one line
    assert _get_number_of_lines(log_contents) == 1


def test_stderr_warning(capfd):
    _ = create_logger(
        "test_logger", prefix="Warn Test", log_file="test_stderr_warning.log"
    )
    print("This is a warning from stderr.", file=sys.stderr)
    _, stderr_contents = capfd.readouterr()
    # stderr should be colored
    assert "\x1b[0m" in stderr_contents
    # stderr should contain one line
    assert _get_number_of_lines(stderr_contents) == 1
    fn = os.path.join(LOGGING_TEST_DIR, "test_stderr_warning.log")
    with open(fn, "r") as f:
        log_contents = f.read()
    # log file should not be colored
    assert "\x1b[0m" not in log_contents
    # log file should contain one line
    assert _get_number_of_lines(log_contents) == 1


def test_stderr_multiline(capfd):
    _ = create_logger(
        "test_logger",
        prefix="Warn Test",
        log_file="test_multiline_stderr_warning.log",
    )
    print(
        "This is a warning from stderr.\nThis is the second line.",
        file=sys.stderr,
    )
    _, stderr_contents = capfd.readouterr()
    # stderr should be colored
    assert "\x1b[0m" in stderr_contents
    # stderr should contain two lines
    assert _get_number_of_lines(stderr_contents) == 2
    fn = os.path.join(LOGGING_TEST_DIR, "test_multiline_stderr_warning.log")
    with open(fn, "r") as f:
        log_contents = f.read()
    # log file should not be colored
    assert "\x1b[0m" not in log_contents
    # log file should contain one line
    assert _get_number_of_lines(log_contents) == 2


if __name__ == "__main__":
    # test_stderr_multiline()
    pytest.main([__file__])
