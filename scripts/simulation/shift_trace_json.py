# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import networkx as nx


@dataclass
class ScheduleOp:
    start: float
    duration: float
    microbatch: int
    is_fw: bool
    is_comm: bool
    device: int

    def askey(self):
        return (
            self.start,
            self.duration,
            self.microbatch,
            self.is_fw,
            self.device,
        )


def get_next_stage(device, is_fw, is_comm, ndevices):
    should_wait = True
    if device == ndevices - 1:
        next_is_fw = False
        if is_fw:
            assert not is_comm
            next_device = device
            next_is_comm = False
            should_wait = False
        else:
            if is_comm:
                next_device = device - 1
                next_is_comm = False
            else:
                next_device = device
                next_is_comm = True
    elif device == 0:
        if not is_fw:
            assert not is_comm
            next_device = None
            next_is_fw = False
            should_wait = False
            next_is_comm = False
        else:
            next_is_fw = True
            if is_comm:
                next_device = device + 1
                next_is_comm = False
            else:
                next_device = device
                next_is_comm = True
    else:
        if is_comm:
            if is_fw:
                next_device = device + 1
            else:
                next_device = device - 1
        else:
            next_device = device
        next_is_fw = is_fw
        next_is_comm = not is_comm
    return next_device, next_is_fw, next_is_comm, should_wait


def load_events(trace):
    events = trace["traceEvents"]
    filtered_trace_events = []
    per_device_events: List[List[ScheduleOp]] = []
    per_device_comm_events: List[List[ScheduleOp]] = []
    for ev in events:
        if ev["ph"] == "X":
            device = ev["pid"]
            if ev["tid"] == 0:
                is_comm = False
                events_store = per_device_events
            else:
                is_comm = True
                events_store = per_device_comm_events
            while len(events_store) <= device:
                events_store.append([])
            if is_comm:
                name = ev["name"].split("_")[0]
            else:
                name = ev["name"]
            if "B" in name:
                is_fw = False
                microbatch = int(name.split("B")[0])
            else:
                is_fw = True
                microbatch = int(name)
            events_store[device].append(
                ScheduleOp(
                    ev["ts"], ev["dur"], microbatch, is_fw, is_comm, device
                )
            )
        elif ev["ph"] == "M":
            filtered_trace_events.append(ev)
    for device_events in per_device_events:
        device_events.sort(key=lambda ev: ev.start)
    for device_events in per_device_comm_events:
        device_events.sort(key=lambda ev: ev.start)
    return per_device_events, per_device_comm_events, filtered_trace_events


def delay_events(events, comm_events, event: ScheduleOp):
    shifted_events = False
    # delay next event on this device
    end = event.start + event.duration
    if event.is_comm:
        current_event_list = comm_events[event.device]
    else:
        current_event_list = events[event.device]
    current_idx = current_event_list.index(event)
    if current_idx < len(current_event_list) - 1:
        next_event = current_event_list[current_idx + 1]
        if next_event.start < end:
            if (
                next_event.microbatch == 0
                and next_event.is_fw
                and next_event.is_comm
                and next_event.device == 0
            ):
                print(
                    "Shifting comm 0 event start time "
                    "from {} to {}, due to {}".format(
                        next_event.start, end, event
                    )
                )
            shifted_events = True
            next_event.start = end
            delay_events(events, comm_events, next_event)
    # delay all successor events
    next_device, next_is_fw, next_is_comm, should_wait = get_next_stage(
        event.device,
        event.is_fw,
        event.is_comm,
        len(events),
    )
    if next_device is not None:
        # find the next event
        if next_is_comm:
            next_event_list = comm_events[next_device]
        else:
            next_event_list = events[next_device]
        for next_event_idx, next_event in enumerate(next_event_list):
            next_event: ScheduleOp
            if (
                event.microbatch == next_event.microbatch
                and next_event.is_fw == next_is_fw
                and next_event.is_comm == next_is_comm
            ):
                if next_event.start < event.start + event.duration:
                    next_event.start = event.start + event.duration
                    shifted_events = True
                    delay_events(events, comm_events, next_event)
                break
    return shifted_events


def reconstruct_json_trace(trace, ref_trace):
    reference_events, ref_comm_events, meta_events = load_events(ref_trace)
    actual_events, actual_comm_events, _ = load_events(trace)
    for ref_events in reference_events:
        for ref_event in ref_events:
            # find the corresponding event in the actual trace
            for actual_ev in actual_events[ref_event.device]:
                if (
                    actual_ev.microbatch == ref_event.microbatch
                    and actual_ev.is_fw == ref_event.is_fw
                ):
                    if ref_event.duration != actual_ev.duration:
                        ref_event.duration = actual_ev.duration
                        delay_events(
                            reference_events, ref_comm_events, ref_event
                        )
    for ref_events in ref_comm_events:
        for ref_event in ref_events:
            for actual_ev in actual_comm_events[ref_event.device]:
                if (
                    actual_ev.microbatch == ref_event.microbatch
                    and actual_ev.is_fw == ref_event.is_fw
                ):
                    if ref_event.duration != actual_ev.duration:
                        ref_event.duration = actual_ev.duration
                        delay_events(
                            reference_events, ref_comm_events, ref_event
                        )
    # add back to trace
    for device_events in reference_events:
        for event in device_events:
            meta_events.append(
                {
                    "ph": "X",
                    "name": str(event.microbatch)
                    if event.is_fw
                    else str(event.microbatch) + "B",
                    "ts": event.start,
                    "dur": event.duration,
                    "pid": event.device,
                    "tid": 0,
                }
            )
    for device_events in ref_comm_events:
        for event in device_events:
            meta_events.append(
                {
                    "ph": "X",
                    "name": (
                        str(event.microbatch)
                        if event.is_fw
                        else str(event.microbatch) + "B"
                    )
                    + "_Comm",
                    "ts": event.start,
                    "dur": event.duration,
                    "pid": event.device,
                    "tid": 1,
                }
            )
    trace["traceEvents"] = meta_events
    return trace


class FrozenScheduleOp(NamedTuple):
    start: float
    duration: float
    microbatch: int
    is_fw: bool
    is_comm: bool
    device: int
    comm_channel: Optional[Tuple[int, int]] = None


def single_source_longest_dag_path_length(graph, s):
    assert graph.in_degree(s) == 0
    dist = dict.fromkeys(graph.nodes, -float("inf"))
    dist[s] = 0
    topo_order = nx.topological_sort(graph)
    for n in topo_order:
        for s in graph.successors(n):
            if dist[s] < dist[n] + graph.edges[n, s]["weight"]:
                dist[s] = dist[n] + graph.edges[n, s]["weight"]
    return dist


def construct_exec_time_dict(trace):
    node_time_dict = {}
    events, comm_events, meta_events = load_events(trace)
    for device, device_events in enumerate(events):
        for event in device_events:
            event: ScheduleOp
            node_time_dict[
                (event.device, event.microbatch, event.is_fw, event.is_comm)
            ] = event.duration
    for device, device_events in enumerate(comm_events):
        for event in device_events:
            event: ScheduleOp
            node_time_dict[
                (event.device, event.microbatch, event.is_fw, event.is_comm)
            ] = event.duration
    return node_time_dict


def convert_to_multistream_comm(trace, node_time_dict=None):
    events, comm_events, meta_events = load_events(trace)
    n_comp_devices = len(events)
    node_dict = {}
    g = nx.DiGraph()
    for device_events in events:
        for event in device_events:
            if node_time_dict is None:
                duration = event.duration
            else:
                duration = node_time_dict[
                    (
                        event.device,
                        event.microbatch,
                        event.is_fw,
                        event.is_comm,
                    )
                ]
            node = FrozenScheduleOp(
                event.start,
                duration,
                event.microbatch,
                event.is_fw,
                event.is_comm,
                event.device,
            )
            g.add_node(node)
            node_dict[
                (event.device, event.microbatch, event.is_fw, event.is_comm)
            ] = node

    for device_events in comm_events:
        for event in device_events:
            (
                next_device,
                next_is_fw,
                next_is_comm,
                should_wait,
            ) = get_next_stage(
                event.device, event.is_fw, event.is_comm, n_comp_devices
            )
            assert next_device is not None
            if next_device > event.device:
                comm_channel = (event.device, next_device)
            else:
                comm_channel = (next_device, event.device)
            if node_time_dict is None:
                duration = event.duration
            else:
                duration = node_time_dict[
                    (
                        event.device,
                        event.microbatch,
                        event.is_fw,
                        event.is_comm,
                    )
                ]
            node = FrozenScheduleOp(
                event.start,
                duration,
                event.microbatch,
                event.is_fw,
                event.is_comm,
                event.device,
                comm_channel,
            )
            g.add_node(node)
            node_dict[
                (event.device, event.microbatch, event.is_fw, event.is_comm)
            ] = node
    # add dependency edges
    for node in g.nodes:
        node: FrozenScheduleOp
        next_device, next_is_fw, next_is_comm, should_wait = get_next_stage(
            node.device, node.is_fw, node.is_comm, n_comp_devices
        )
        if next_device is not None:
            next_node = node_dict[
                (next_device, node.microbatch, next_is_fw, next_is_comm)
            ]
            g.add_edge(node, next_node, weight=node.duration)
    # add control edges
    for device_events in events:
        for idx in range(len(device_events) - 1):
            node = node_dict[
                (
                    device_events[idx].device,
                    device_events[idx].microbatch,
                    device_events[idx].is_fw,
                    device_events[idx].is_comm,
                )
            ]
            next_node = node_dict[
                (
                    device_events[idx + 1].device,
                    device_events[idx + 1].microbatch,
                    device_events[idx + 1].is_fw,
                    device_events[idx + 1].is_comm,
                )
            ]
            g.add_edge(node, next_node, weight=node.duration)
    # collect nodes in comm channels
    per_channel_nodes = {}
    for node in g.nodes:
        node: FrozenScheduleOp
        if node.comm_channel is not None:
            if node.comm_channel not in per_channel_nodes:
                per_channel_nodes[node.comm_channel] = []
            per_channel_nodes[node.comm_channel].append(node)
    # sort nodes in comm channels
    for channel, nodes in per_channel_nodes.items():
        nodes.sort(key=lambda x: x.start)
    # add control edges in comm channels
    for channel, nodes in per_channel_nodes.items():
        for idx in range(len(nodes) - 1):
            g.add_edge(nodes[idx], nodes[idx + 1], weight=nodes[idx].duration)
    # add src node
    src_node = "src"
    g.add_node(src_node)
    for node in g.nodes:
        if node != "src":
            g.add_edge(src_node, node, weight=0)
    # add sink node
    sink_node = "sink"
    g.add_node(sink_node)
    for node in g.nodes:
        if node != "src" and node != "sink":
            g.add_edge(node, sink_node, weight=0)
    # start time of all nodes
    start_time_dict = single_source_longest_dag_path_length(g, "src")
    new_events = []
    for node in g.nodes:
        node: FrozenScheduleOp
        if node != "src" and node != "sink":
            if node.is_comm:
                pid = str(node.comm_channel)
            else:
                pid = str(node.device)
            new_events.append(
                {
                    "ph": "X",
                    "name": (
                        str(node.microbatch)
                        if node.is_fw
                        else str(node.microbatch) + "B"
                    )
                    + ("" if not node.is_comm else "_Comm"),
                    "ts": start_time_dict[node],
                    "dur": node.duration,
                    "pid": pid,
                    "tid": 0,
                }
            )
    trace["traceEvents"] = new_events + meta_events
    return trace, start_time_dict["sink"]


if __name__ == "__main__":
    with open("./test_wfcyclic_trace.json", "r") as f:
        wfcyclic_trace = json.load(f)
    with open("./test_reference.json", "r") as f:
        ref_trace = json.load(f)
    exec_time_dict = construct_exec_time_dict(wfcyclic_trace)
    multistream_ref_trace, ref_makespan = convert_to_multistream_comm(
        ref_trace, exec_time_dict
    )
    multistream_wfcyclic_trace, makespan = convert_to_multistream_comm(
        wfcyclic_trace
    )
    print("Ref Makespan: {}, Makespan: {}".format(ref_makespan, makespan))
    with open("./test_multistream_cyclic_trace.json", "w") as f:
        json.dump(multistream_wfcyclic_trace, f)
    with open("./test_multistream_ref_trace.json", "w") as f:
        json.dump(multistream_ref_trace, f)
