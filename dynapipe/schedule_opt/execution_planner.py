# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import logging
import math
from typing import List, Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans

from dynapipe.data_opt.cost_models import ProfileBasedCostModelWithRC
from dynapipe.model import (
    DynaPipeCluster,
    DynaPipeMicrobatch,
    DynaPipeMinibatch,
    TransformerModelSpec,
    get_simulator,
)
from dynapipe.pipe.instruction_optimizer import InstructionOptimizer
from dynapipe.pipe.instructions import (
    ExecutionPlan,
    PipeInstruction,
    name_to_recompute_method,
)
from dynapipe.pipe.utils import validate_device_assignment
from dynapipe.utils.memory_utils import get_transformer_output_memory


def optimize_schedule(
    sch_type: str,
    opt_minibatch: DynaPipeMinibatch,
    opt_cluster: DynaPipeCluster,
    device_assignment: List[int],
    try_permutations=True,
    perm_clusters=None,
    perm_cluster_algo="kmeans",
    include_memory_stats=False,
    progress_bar=False,
    memory_limit=float("inf"),
    disable_scheduler_memory_limit=False,
    max_otf_microbatches=None,
    raise_on_oom=True,
    rc_type: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
):
    if try_permutations:
        if perm_clusters is None:
            if len(opt_minibatch.microbatches) > 20:
                perm_clusters = 3
            else:
                perm_clusters = 4
        if len(opt_minibatch.microbatches) > perm_clusters:
            mb_vectors = []
            for mb in opt_minibatch.microbatches:
                # use fw and bw time as features
                mb_vectors.append(
                    [
                        mb.fw_exec_times[0],
                        mb.fw_exec_times[-1],
                        mb.bw_exec_times[0],
                        mb.bw_exec_times[-1],
                    ]
                )
            mb_vectors = np.array(mb_vectors)
            if perm_cluster_algo == "kmeans":
                cluster = KMeans(
                    perm_clusters,
                    random_state=0,
                    n_init="auto",
                ).fit(mb_vectors)
            elif perm_cluster_algo == "agglomerative":
                cluster = AgglomerativeClustering(
                    perm_clusters,
                    linkage="complete",
                ).fit(mb_vectors)
            mb_labels = list(cluster.labels_)
            n_clusters = max(mb_labels) + 1
            assert n_clusters <= perm_clusters
            mb_groups = [[] for _ in range(n_clusters)]
            mb_idx2group = {}
            for i, label in enumerate(mb_labels):
                mb_groups[label].append(i)
                mb_idx2group[i] = label
            result_premutations = []
            for perm in itertools.permutations(range(len(mb_groups))):
                # generate a random permutation for each group
                mb_random_perm_per_label = {}
                for label, mb_indices in enumerate(mb_groups):
                    shuffled_indices = np.random.permutation(mb_indices)
                    mb_random_perm_per_label[label] = list(shuffled_indices)
                reconstructed_perm = []
                for label in perm:
                    reconstructed_perm.extend(mb_random_perm_per_label[label])
                result_premutations.append(reconstructed_perm)
            permutations = result_premutations
        else:
            permutations = list(
                itertools.permutations(range(len(opt_minibatch.microbatches)))
            )
    else:
        permutations = []
    # always try the original order
    permutations.append(list(range(len(opt_minibatch.microbatches))))

    def _run_schedules(scheduler_memory_limit):
        max_makespan = 0.0
        max_stats = None
        max_instructions = []

        min_makespan = float("inf")
        min_stats = None
        min_instructions = []
        if progress_bar:
            from tqdm import tqdm

            iterator = tqdm(permutations)
        else:
            iterator = permutations
        debug_json = None
        mem_for_perms = []
        for perm in iterator:
            permuted_minibatch = opt_minibatch.permute_microbatches(perm)
            # get simulator
            simulator = get_simulator(
                sch_type,
                permuted_minibatch,
                opt_cluster,
                device_assignment,
                include_memory_stats=include_memory_stats,
                memory_limit=scheduler_memory_limit,
                max_otf_microbatches=max_otf_microbatches,
                logger=logger,
            )
            timeline_json = simulator.schedule()
            instructions = simulator.get_instructions()
            peak_memory = simulator.get_executor_peak_memory()
            max_memory_device = -1
            max_device_memory = -1
            for device, memory in peak_memory.items():
                if memory > max_device_memory:
                    max_memory_device = device
                    max_device_memory = memory
            makespan = simulator.get_makespan()
            if makespan is None:
                continue
            makespan = makespan / 1000.0
            debug_json = timeline_json
            mem_for_perms.append(max_device_memory)
            if max_device_memory > memory_limit:
                continue
            if makespan > max_makespan:
                max_makespan = makespan
                max_stats = (
                    perm,
                    max_device_memory,
                    max_memory_device,
                    timeline_json,
                )
                max_instructions = instructions
            if makespan < min_makespan:
                min_makespan = makespan
                min_stats = (
                    perm,
                    max_device_memory,
                    max_memory_device,
                    timeline_json,
                )
                min_instructions = instructions
        if logger is not None and max_makespan > 0.0:
            logger.debug(
                "Sched mem limit: {}, RC type: {}, Schedule type: {}, "
                "min peak memory: {} MB, makespan: {}.".format(
                    scheduler_memory_limit,
                    rc_type,
                    sch_type,
                    min(mem_for_perms),
                    min_makespan,
                )
            )
        return (
            max_makespan,
            max_stats,
            max_instructions,
            min_makespan,
            min_stats,
            min_instructions,
            debug_json,
            mem_for_perms,
        )

    # first try without setting memory limit on scheduler
    # (i.e. see if there exist a feasible permutation)
    (
        max_makespan,
        max_stats,
        max_instructions,
        min_makespan,
        min_stats,
        min_instructions,
        debug_json,
        mem_for_perms,
    ) = _run_schedules(float("inf"))
    if (
        max_makespan == 0.0
        and sch_type == "wait-free-cyclic"
        and not disable_scheduler_memory_limit
    ):
        # try with scheduler memory limit
        if logger is not None:
            logger.debug("Trying with scheduler memory limit.")
        (
            max_makespan,
            max_stats,
            max_instructions,
            min_makespan,
            min_stats,
            min_instructions,
            debug_json,
            mem_for_perms,
        ) = _run_schedules(memory_limit)
    if max_makespan == 0.0 and raise_on_oom:
        # with open("./test_memory.json", "w") as f:
        #     json.dump(debug_json, f)
        raise RuntimeError(
            "No feasible schedule within memory limit found. "
            "Memory consumption for different permutations: "
            "min: {}, max: {}.".format(
                [] if not mem_for_perms else min(mem_for_perms),
                [] if not mem_for_perms else max(mem_for_perms),
            )
        )

    return (
        max_makespan,
        max_stats,
        max_instructions,
        min_makespan,
        min_stats,
        min_instructions,
    )


def construct_minibatch_spec(
    model_spec: TransformerModelSpec,
    cost_model: ProfileBasedCostModelWithRC,
    minibatch: List[Tuple[int, int, int]],
    rc_type: str,
    dp_size: int = 1,
    tp_size: int = 1,
    zero_stage: int = 0,
    minibatch_idx: Optional[int] = None,
    name="microbatch",
):
    # use cost model to get the execution time and memory consumption
    # of each stage
    microbatches = []
    for microbatch_idx, (mbsize, input_seqlen, target_seqlen) in enumerate(
        minibatch
    ):
        # sanity check
        if model_spec.n_decoder_layers == 0 and target_seqlen != 0:
            raise ValueError(
                "Target sequence length must be 0 if there are "
                "no decoder layers."
            )
        if target_seqlen == 0 and model_spec.n_decoder_layers > 0:
            raise ValueError(
                "Target sequence length cannot be 0 if there are "
                "decoder layers."
            )
        mb = DynaPipeMicrobatch(str(microbatch_idx))

        def _get_cost(stage_name, seqlen):
            return (
                cost_model.get_cost(
                    tp_size,
                    rc_type,
                    stage_name,
                    seqlen,
                    mbsize,
                )
                * 1000
            )

        def _get_stored_activation(stage_name, seqlen):
            return cost_model.get_stored_activation(
                tp_size,
                rc_type,
                stage_name,
                seqlen,
                mbsize,
            )

        def _get_peak_activation(stage_name, seqlen):
            return cost_model.get_peak_activation(
                tp_size,
                rc_type,
                stage_name,
                seqlen,
                mbsize,
            )

        # every cost needs to time * 1000 since get_cost returns time
        # in milliseconds and we need to convert it to microseconds
        # the costs are for each single actual transformer layer in the model
        enc_fw_time = _get_cost("Encoder FW", input_seqlen)
        enc_bw_time = _get_cost("Encoder BW", input_seqlen)
        enc_postprocess_fw_time = 0
        enc_postprocess_bw_time = 0
        dec_postprocess_fw_time = 0
        dec_postprocess_bw_time = 0
        if target_seqlen > 0:
            dec_fw_time = _get_cost(
                "Decoder FW", (input_seqlen, target_seqlen)
            )
            dec_bw_time = _get_cost(
                "Decoder BW", (input_seqlen, target_seqlen)
            )
            dec_postprocess_fw_time = _get_cost(
                "Postprocess FW", target_seqlen
            )
            dec_postprocess_bw_time = _get_cost(
                "Postprocess BW", target_seqlen
            )
        else:
            dec_fw_time = 0
            dec_bw_time = 0
            enc_postprocess_fw_time = _get_cost("Postprocess FW", input_seqlen)
            enc_postprocess_bw_time = _get_cost("Postprocess BW", input_seqlen)
        enc_stored_activation_memory = _get_stored_activation(
            "Encoder", input_seqlen
        )
        if target_seqlen > 0:
            dec_stored_activation_memory = _get_stored_activation(
                "Decoder", (input_seqlen, target_seqlen)
            )
        else:
            dec_stored_activation_memory = 0
        enc_peak_activation_memory = _get_peak_activation(
            "Encoder", input_seqlen
        )
        if target_seqlen > 0:
            dec_peak_activation_memory = _get_peak_activation(
                "Decoder", (input_seqlen, target_seqlen)
            )
        else:
            dec_peak_activation_memory = 0
        emb_model_state_memory = cost_model.get_model_state(
            tp_size,
            rc_type,
            "Embedding",
            n_shards=dp_size,
            zero_stage=zero_stage,
        )
        enc_model_state_memory = cost_model.get_model_state(
            tp_size,
            rc_type,
            "Encoder",
            n_shards=dp_size,
            zero_stage=zero_stage,
        )
        if target_seqlen > 0:
            dec_model_state_memory = cost_model.get_model_state(
                tp_size,
                rc_type,
                "Decoder",
                n_shards=dp_size,
                zero_stage=zero_stage,
            )
        else:
            dec_model_state_memory = 0
        enc_model_output_memory = get_transformer_output_memory(
            input_seqlen, mbsize, model_spec.hidden_dim, bytes_per_element=2
        )
        if target_seqlen > 0:
            dec_model_output_memory = get_transformer_output_memory(
                target_seqlen,
                mbsize,
                model_spec.hidden_dim,
                bytes_per_element=2,
            )
        else:
            dec_model_output_memory = 0
        # sanity check
        stats = [
            enc_fw_time,
            enc_bw_time,
            dec_fw_time,
            dec_bw_time,
            emb_model_state_memory,
            enc_stored_activation_memory,
            dec_stored_activation_memory,
            enc_peak_activation_memory,
            dec_peak_activation_memory,
            enc_model_state_memory,
            dec_model_state_memory,
        ]
        stats_names = [
            "enc_fw_time",
            "enc_bw_time",
            "dec_fw_time",
            "dec_bw_time",
            "emb_model_state_memory",
            "enc_stored_activation_memory",
            "dec_stored_activation_memory",
            "enc_peak_activation_memory",
            "dec_peak_activation_memory",
            "enc_model_state_memory",
            "dec_model_state_memory",
        ]
        for s, s_name in zip(stats, stats_names):
            if s is None or math.isnan(s) or math.isinf(s):
                # for debug purpose:
                # raise ValueError(
                #     f"In minibatch {minibatch_idx}, "
                #     f"microbatch {microbatch_idx} "
                #     f"({(mbsize, input_seqlen, target_seqlen)}), "
                #     f"{s_name} is invalid ({s})."
                # )

                # invalid cost, return None
                return None

        # populate execution time statistics for microbatch
        mb.set_fw_exec_times(
            [enc_fw_time] * (model_spec.n_encoder_layers - 1)
            + [enc_fw_time + enc_postprocess_fw_time]
            + [dec_fw_time] * max(0, model_spec.n_decoder_layers - 1)
            + (
                [dec_fw_time + dec_postprocess_fw_time]
                if target_seqlen > 0
                else []
            )
        )
        mb.set_bw_exec_times(
            (
                [dec_bw_time + dec_postprocess_bw_time]
                if target_seqlen > 0
                else []
            )
            + [dec_bw_time] * max(0, model_spec.n_decoder_layers - 1)
            + [enc_bw_time + enc_postprocess_bw_time]
            + [enc_bw_time] * (model_spec.n_encoder_layers - 1)
        )
        mb.set_fw_comm_size(
            [enc_model_output_memory]
            * (model_spec.n_encoder_layers - (1 if target_seqlen == 0 else 0))
            + [enc_model_output_memory + dec_model_output_memory]
            * max(0, model_spec.n_decoder_layers - 1)
        )
        mb.set_bw_comm_size(
            [enc_model_output_memory + dec_model_output_memory]
            * model_spec.n_decoder_layers
            + [enc_model_output_memory] * (model_spec.n_encoder_layers - 1)
        )
        mb.set_model_stored_activation_memory(
            [enc_stored_activation_memory] * model_spec.n_encoder_layers
            + [dec_stored_activation_memory] * model_spec.n_decoder_layers
        )
        mb.set_model_peak_activation_memory(
            [enc_peak_activation_memory] * model_spec.n_encoder_layers
            + [dec_peak_activation_memory] * model_spec.n_decoder_layers
        )
        # first layer of encoder and decoder also have embedding model state
        mb.set_model_state_memory(
            [emb_model_state_memory + enc_model_state_memory]
            + [enc_model_state_memory] * (model_spec.n_encoder_layers - 1)
            + (
                [emb_model_state_memory + dec_model_state_memory]
                if target_seqlen > 0
                else []
            )
            + [dec_model_state_memory]
            * max(0, model_spec.n_decoder_layers - 1)
        )
        # shapes should be a tuple of tuples
        mb.set_activation_shapes(
            [[(mbsize, input_seqlen, model_spec.hidden_dim)]]
            * model_spec.n_encoder_layers
            + [
                [
                    (mbsize, input_seqlen, model_spec.hidden_dim),
                    (mbsize, target_seqlen, model_spec.hidden_dim),
                ]
            ]
            * model_spec.n_decoder_layers
        )
        mb.check_all_set()
        microbatches.append(mb)
    minibatch_spec = DynaPipeMinibatch(name, microbatches)
    return minibatch_spec


class ExecutionPlanner:
    def __init__(
        self,
        cluster_spec: DynaPipeCluster,
        model_spec: TransformerModelSpec,
        device_assignment: List[int],
        device_memory_limit: int,
        cost_model: ProfileBasedCostModelWithRC,
        dp_size: int = 1,
        tp_size: int = 1,
        zero_stage: int = 0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cluster_spec = cluster_spec
        self.model_spec = model_spec
        self.cost_model = cost_model
        self.device_assignment = device_assignment
        self.n_devices = max(device_assignment) + 1
        self.device_memory_limit = device_memory_limit
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.zero_stage = zero_stage
        self.logger = logger
        (
            self.device_assignment_type,
            self.valid_schedule_methods,
            self.n_layers_per_stage,
            self.n_chunks_per_device,
        ) = validate_device_assignment(
            model_spec, cluster_spec, self.device_assignment
        )

    def _create_candidates(
        self,
        batch: List[Tuple[int, int, int]],
        schedule_method="dynamic",
        rc_type=None,
    ):
        if rc_type is not None:
            if not isinstance(rc_type, list):
                available_rc_types = [rc_type]
            else:
                available_rc_types = rc_type
        else:
            available_rc_types = ["none", "selective", "full"]
        if schedule_method == "dynamic":
            sch_methods = self.valid_schedule_methods
            spec_args = []
            for rc_type in available_rc_types:
                for sch in sch_methods:
                    spec_args.append((sch, rc_type))
        else:
            if schedule_method not in self.valid_schedule_methods:
                raise ValueError(
                    "Invalid schedule scheme: "
                    "{} for device assignment: {}".format(
                        schedule_method, self.device_assignment
                    )
                )
            spec_args = [
                (schedule_method, rc_type) for rc_type in available_rc_types
            ]
        candidates = []
        for schedule_method, rc_type in spec_args:
            minibatch_spec = construct_minibatch_spec(
                self.model_spec,
                self.cost_model,
                batch,
                rc_type,
                dp_size=self.dp_size,
                tp_size=self.tp_size,
                zero_stage=self.zero_stage,
            )
            if minibatch_spec is not None:
                candidates.append((schedule_method, rc_type, minibatch_spec))
        return candidates

    def _optimize_instructions(
        self,
        instructions: List[List[PipeInstruction]],
        n_stages: int,
    ):
        # instructions: instructions for each executor
        # Necessary steps to ensure correctness:
        #   1. Add CommunicationFinishInsturctions at appropriate places
        #   2. Allocate buffer slots (not buffer themselves)
        # Potential optimizations:
        #   1. Merge consecutive communication instructions (trade-off)
        #   2. Reschedule communication instructions
        #   3. Pre-allocate buffers to reduce memory fragmentation
        instrs, n_buffers = InstructionOptimizer(
            instructions, n_stages
        ).optimize()
        return instrs, n_buffers

    def generate_execution_plan(
        self,
        batch: List[Tuple[int, int, int]],
        limit_rc_type=None,
        schedule_method="dynamic",
        disable_permute_microbatches=False,
        disable_scheduler_memory_limit=False,
        current_batch_idx=None,
    ):
        candidates = self._create_candidates(
            batch, schedule_method=schedule_method, rc_type=limit_rc_type
        )
        best_instrs = None
        best_sch = None
        best_rc = None
        best_cost = None
        best_stats = None
        for schedule_method, rc_type, minibatch_spec in candidates:
            (
                max_makespan,
                _,
                _,
                min_makespan,
                min_stats,
                min_instructions,
            ) = optimize_schedule(
                schedule_method,
                minibatch_spec,
                self.cluster_spec,
                self.device_assignment,
                try_permutations=not disable_permute_microbatches,
                include_memory_stats=True,
                progress_bar=False,
                memory_limit=self.device_memory_limit,
                disable_scheduler_memory_limit=disable_scheduler_memory_limit,
                raise_on_oom=False,
                rc_type=rc_type,
                logger=self.logger,
            )
            if max_makespan < 1e-5:
                # no feasible schedule
                if self.logger:
                    self.logger.debug(
                        "No feasible schedule for batch {} "
                        "using {} and recompute {}".format(
                            current_batch_idx, schedule_method, rc_type
                        )
                    )
                continue
            if best_cost is None or min_makespan < best_cost:
                best_cost = min_makespan
                best_sch = schedule_method
                best_rc = rc_type
                best_instrs = min_instructions
                best_stats = min_stats
        if best_instrs is None:
            raise RuntimeError(
                "No feasible schedule for batch {}.".format(current_batch_idx)
            )
        # get total number of stages
        best_instrs: List[List[PipeInstruction]]
        n_stages = (
            max([instr.stage for instrs in best_instrs for instr in instrs])
            + 1
        )
        assigned_stages_per_executor = []
        for instrs in best_instrs:
            assigned_stages = set()
            for instr in instrs:
                assigned_stages.add(instr.stage)
            assigned_stages = sorted(list(assigned_stages))
            assigned_stages_per_executor.append(assigned_stages)
        # construct execution plan
        if best_cost is None:
            # no feasible schedule
            return None, None, None, None, None
        assert len(best_instrs) == self.n_devices
        # run necessary optimization pass on instructions
        optimized_instrs, n_buffers = self._optimize_instructions(
            best_instrs, n_stages
        )
        execution_plans = [
            ExecutionPlan(
                instr,
                len(batch),
                self.n_devices,
                n_stages,
                i,
                assigned_stages_per_executor[i],
                name_to_recompute_method(best_rc),
                n_buffer,
            )
            for i, (instr, n_buffer) in enumerate(
                zip(optimized_instrs, n_buffers)
            )
        ]
        return execution_plans, best_cost, best_stats, best_rc, best_sch
