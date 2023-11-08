# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

from dynapipe.model import DynaPipeCluster, TransformerModelSpec

from .instructions import *  # noqa: F403


def validate_device_assignment(
    model_spec: TransformerModelSpec,
    cluster_spec: DynaPipeCluster,
    device_assignment: List[int],
):
    """
    Validate device assignment and detect device assignment type.
    Args:
        device_assignment: List of device ids for each layer.
    """
    appeared_devices = set()
    for device in device_assignment:
        if device not in appeared_devices:
            # new device
            assert device == len(appeared_devices), (
                "Devices must appear in indexed order. "
                "e.g. [0, 1, 2, 3] is valid, "
                "[0, 1, 3, 2] is not valid."
            )
            appeared_devices.add(device)
    n_devices = len(appeared_devices)
    assert n_devices == cluster_spec.n_devices, (
        "Number of devices used in device assignment "
        "must be equal to number of devices in cluster spec."
    )
    virtual_layer_to_actual_layers = [[]]
    virtual_layer_devices = [0]
    last_device = 0
    for device in device_assignment:
        if device == last_device:
            virtual_layer_to_actual_layers[-1].append(device)
        else:
            virtual_layer_to_actual_layers.append([device])
            virtual_layer_devices.append(device)
        last_device = device
    n_actual_layers_per_virtual_layer = len(virtual_layer_to_actual_layers[0])
    for virtual_layer in virtual_layer_to_actual_layers:
        n_encoder_layers_in_virtual_layer = len(
            [
                layer
                for layer in virtual_layer
                if layer < model_spec.n_encoder_layers
            ]
        )
        n_decoder_layers_in_virtual_layer = (
            len(virtual_layer) - n_encoder_layers_in_virtual_layer
        )
        if n_encoder_layers_in_virtual_layer > 0:
            assert (
                len(virtual_layer) == n_encoder_layers_in_virtual_layer
            ), "Number of layers on each virtual layer must be the same."
        if n_decoder_layers_in_virtual_layer > 0:
            assert (
                len(virtual_layer) == n_decoder_layers_in_virtual_layer
            ), "Number of layers on each virtual layer must be the same."
    if len(device_assignment) != n_actual_layers_per_virtual_layer:
        # only check if we are actually using pipeline parallelism
        assert (
            model_spec.n_encoder_layers % n_actual_layers_per_virtual_layer
            == 0
        ), (
            f"Number of encoder layers ({model_spec.n_encoder_layers}) "
            f"must be divisible by number of layers on each virtual layer "
            f"({n_actual_layers_per_virtual_layer})."
        )
        assert (
            model_spec.n_decoder_layers % n_actual_layers_per_virtual_layer
            == 0
        ), (
            f"Number of decoder layers ({model_spec.n_decoder_layers}) "
            f"must be divisible by number of layers on each virtual layer "
            f"({n_actual_layers_per_virtual_layer})."
        )
    # classify device assignment into linear, interleaved and other
    device_assignment_type = "other"
    if len(virtual_layer_devices) == n_devices:
        if virtual_layer_devices == list(range(n_devices)):
            device_assignment_type = "linear"
    else:
        n_chunks = len(virtual_layer_devices) // n_devices
        interleaved_assignment = list(range(n_devices)) * n_chunks
        if interleaved_assignment == virtual_layer_devices:
            device_assignment_type = "interleaved"
    if (
        device_assignment_type == "interleaved"
        and model_spec.n_decoder_layers == 0
    ):
        # interleaved device assignment is not supported for decoder only
        # models
        raise NotImplementedError(
            "Interleaved device assignment is not supported "
            "for decoder only models."
        )
    valid_schedule_methods = ["wait-free-cyclic"]
    if device_assignment_type == "linear" and n_devices > 1:
        valid_schedule_methods.append("1F1B")
    elif device_assignment_type == "interleaved":
        valid_schedule_methods.append("interleaved-1F1B")
    n_chunks_per_device = len(virtual_layer_devices) // n_devices
    return (
        device_assignment_type,
        valid_schedule_methods,
        n_actual_layers_per_virtual_layer,
        n_chunks_per_device,
    )


def check_deadlock(eps: List[ExecutionPlan]):
    # validate if the instruction sequence will result in a deadlock
    # filter out all communication instructions
    _INSTR_TYPE_MAP = {
        SendActivationStart: RecvActivationStart,
        SendGradStart: RecvGradStart,
        RecvActivationStart: SendActivationStart,
        RecvGradStart: SendGradStart,
    }
    comm_ops_per_exec = []
    for ep in eps:
        instrs = ep.instructions
        comm_ops = []
        for instr in instrs:
            if isinstance(
                instr,
                (
                    SendActivationStart,
                    SendGradStart,
                    RecvActivationStart,
                    RecvGradStart,
                ),
            ):
                comm_ops.append(instr)
        comm_ops_per_exec.append(comm_ops)

    def _alert_deadlock(exec_idx, peer_idx, current_instr_idx, peer_instr_idx):
        additonal_info = ""
        if peer_idx is None:
            additonal_info += (
                "Executor {exec_idx} "
                "has unfinished instruction "
                f"{comm_ops_per_exec[exec_idx][current_instr_idx]}. \n"
            )
        else:
            if current_instr_idx < len(
                comm_ops_per_exec[exec_idx]
            ) and peer_instr_idx < len(comm_ops_per_exec[peer_idx]):
                instr_order_str = "\n\t".join(
                    [str(x) for x in comm_ops_per_exec[exec_idx]]
                )
                peer_instr_order_str = "\n\t".join(
                    [str(x) for x in comm_ops_per_exec[peer_idx]]
                )
                additonal_info += (
                    "Mismatched instructions "
                    f"{comm_ops_per_exec[exec_idx][current_instr_idx]}\n"
                    "\t\t\tand "
                    f"{comm_ops_per_exec[peer_idx][peer_instr_idx]}.\n"
                )
            else:
                additonal_info += (
                    "No matching instruction for "
                    f"{comm_ops_per_exec[exec_idx][current_instr_idx]}. \n"
                )
            additonal_info += (
                f"Instruction order: \n\t{instr_order_str}.\n"
                f"Peer instruction order: \n\t{peer_instr_order_str}."
            )
        raise RuntimeError(
            "[INTERNAL ERROR] "
            f"Deadlock detected between exec {exec_idx} "
            f"(current) and {peer_idx} (peer).\n" + additonal_info
        )

    current_instrs_per_exec = [0] * len(eps)
    # eliminate matching instructions
    while True:
        progress = False
        for exec_idx, instr_idx in enumerate(current_instrs_per_exec):
            if instr_idx >= len(comm_ops_per_exec[exec_idx]):
                continue
            instr = comm_ops_per_exec[exec_idx][instr_idx]
            # check if there is a matching instruction on peer exec
            peer = instr.peer
            peer_instr_idx = current_instrs_per_exec[peer]
            if peer_instr_idx >= len(comm_ops_per_exec[peer]):
                # deadlock
                _alert_deadlock(exec_idx, peer, instr_idx, peer_instr_idx)
            peer_instr = comm_ops_per_exec[peer][peer_instr_idx]
            # check peer
            if peer_instr.peer != exec_idx:
                # not current receiving from/sending to us
                continue
            # if peer is receiving/sending to us, the instructions must match
            # check if the instruction type matches
            if not isinstance(instr, _INSTR_TYPE_MAP[type(peer_instr)]):
                # deadlock
                _alert_deadlock(exec_idx, peer, instr_idx, peer_instr_idx)
            # check shape
            if instr.buffer_shapes != peer_instr.buffer_shapes:
                # deadlock
                _alert_deadlock(exec_idx, peer, instr_idx, peer_instr_idx)
            # check passed, increment instruction index on both execs
            current_instrs_per_exec[exec_idx] += 1
            current_instrs_per_exec[peer] += 1
            progress = True
        if not progress:
            break
    # check if all instructions are consumed
    for exec_idx, instr_idx in enumerate(current_instrs_per_exec):
        if instr_idx < len(comm_ops_per_exec[exec_idx]):
            _alert_deadlock(exec_idx, None, instr_idx, None)
