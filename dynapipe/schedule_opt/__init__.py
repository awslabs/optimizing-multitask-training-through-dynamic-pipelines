# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .cyclic_schedule import CyclicScheduler
from .fifo_schedule import FIFOScheduler
from .ofob_schedule import OFOBSchedulerRegistry as reg
from .wait_free_cyclic_schedule import WaitFreeCyclicScheduler

AVAILABLE_SCHEDULERS = {
    "cyclic": CyclicScheduler,
    "fifo": FIFOScheduler,
    "wait-free-cyclic": WaitFreeCyclicScheduler,
    "1F1B": reg.get_scheduler_factory(placement_type="linear"),
    "relaxed-1F1B": reg.get_scheduler_factory(strictness="relaxed"),
    "interleaved-1F1B": reg.get_scheduler_factory(
        placement_type="interleaved"
    ),
    "interleaved-relaxed-1F1B": reg.get_scheduler_factory(
        strictness="interleaved-relaxed"
    ),
    "interleaved-cyclic-1F1B": reg.get_scheduler_factory(
        placement_type="interleaved", dependency_policy="cyclic"
    ),
}


def get_available_schedulers():
    return AVAILABLE_SCHEDULERS.keys()


def get_scheduler_class(scheduler_name):
    if scheduler_name not in AVAILABLE_SCHEDULERS:
        raise ValueError(
            f"Scheduler {scheduler_name} not available."
            f"Available schedulers: {get_available_schedulers()}"
        )
    return AVAILABLE_SCHEDULERS[scheduler_name]
