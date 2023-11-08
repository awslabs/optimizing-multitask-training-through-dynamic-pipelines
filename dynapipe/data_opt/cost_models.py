# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pickle
import re
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class LMStage:
    EMBEDDING = "embedding"
    ENCODER = "encoder"
    DECODER = "decoder"
    POSTPROCESS = "postprocess"


class TrainStage:
    FORWARD = "forward"
    BACKWARD = "backward"


class ProfileBasedCostModel(object):
    """
    Cost model for a single LM layer's computation time and memory usage based
    on profiled data. Profiles are generated through microbenchmarks, see the
    following files for examples:
    https://github.com/chenyu-jiang/Megatron-LM/blob/dynapipe/microbenchmark_gpt.py
    https://github.com/chenyu-jiang/Megatron-LM/blob/dynapipe/microbenchmark_t5.py
    """

    def __init__(
        self,
        profile_paths=[],
        _timing_data=None,
        _stored_activation_data=None,
        _peak_activation_data=None,
        _model_state_data=None,
        _max_throughput_achieved=None,
    ) -> None:
        self.profile_paths = profile_paths
        # metadata
        self._metadata = {}
        if _timing_data is None:
            # read profile data from file
            (
                self.timing_data,
                self.stored_activation_data,
                self.peak_activation_data,
                self.model_state_data,
                self.max_throughput_achieved,
            ) = self._read_profile_data()
        else:
            (
                self.timing_data,
                self.stored_activation_data,
                self.peak_activation_data,
                self.model_state_data,
                self.max_throughput_achieved,
            ) = (
                _timing_data,
                _stored_activation_data,
                _peak_activation_data,
                _model_state_data,
                _max_throughput_achieved,
            )
        # create interpolators
        self.exec_time_interpolators = {}
        self.throughput_interpolators = {}
        # memory interpolators
        self.stored_activation_interpolators = {}
        self.peak_activation_interpolators = {}

        for stage_key, seqlen_dict in self.timing_data.items():
            interps = self._create_interpolator(seqlen_dict)
            assert len(interps) == 2
            self.exec_time_interpolators[stage_key] = interps[0]
            self.throughput_interpolators[stage_key] = interps[1]
        for stage_key, seqlen_dict in self.stored_activation_data.items():
            interps = self._create_interpolator(seqlen_dict)
            assert len(interps) == 1
            self.stored_activation_interpolators[stage_key] = interps[0]
        for stage_key, seqlen_dict in self.peak_activation_data.items():
            interps = self._create_interpolator(seqlen_dict)
            assert len(interps) == 1
            self.peak_activation_interpolators[stage_key] = interps[0]
        self._supported_sequence_lengths = {}
        for stage_key in self.timing_data.keys():
            self._supported_sequence_lengths[stage_key] = sorted(
                self.timing_data[stage_key].keys()
            )
        self._supported_sequence_lengths_activation = {}
        for stage_key in self.stored_activation_data.keys():
            self._supported_sequence_lengths_activation[stage_key] = sorted(
                self.stored_activation_data[stage_key].keys()
            )
        # cache
        self._interpolate_cost_cache = {}
        self._interpolate_stored_activation_cache = {}
        self._interpolate_peak_activation_cache = {}

    def _create_interpolator(self, seqlen_dict):
        # organize data into nd grids, [mbs, seqlens, data]
        # where seqlens and data can have multiple dimensions
        # we create a different interpolator for each data dimension
        mbs = set()
        sample_seqlen_key = list(seqlen_dict.keys())[0]
        n_seqlen_dims = (
            len(sample_seqlen_key)
            if isinstance(sample_seqlen_key, tuple)
            else 1
        )
        sample_value = list(seqlen_dict.values())[0]
        n_data_dims = len(sample_value[0]) - 1
        seqlens = [set() for _ in range(n_seqlen_dims)]
        values = [set() for _ in range(n_data_dims)]
        for seqlen_key, mbs_and_data in seqlen_dict.items():
            if not isinstance(seqlen_key, tuple):
                seqlen_key = (seqlen_key,)
            for i, seqlen in enumerate(seqlen_key):
                seqlens[i].add(seqlen)
            for t in mbs_and_data:
                # t is an tuple of (mbs, data_dim0, data_dim1, ...)
                mbs.add(t[0])
                for i, datum in enumerate(t[1:]):
                    values[i].add(datum)
        # we extend the mbs and seqlens by one element to disallow
        # extrapolation beyond maximum values profiled
        mbs = sorted(mbs)
        mbs = mbs + [mbs[-1] * 2]
        seqlens = [sorted(s) for s in seqlens]
        seqlens = [s + [s[-1] * 2] for s in seqlens]
        values = [sorted(s) for s in values]
        values = [s + [s[-1] * 2] for s in values]
        # create interpolators
        interpolators = []
        x_shapes = [len(mbs)] + [len(s) for s in seqlens]
        # don't use infinity as fill value here, as during interpolation
        # it may get multiplied by a zero weight, resulting in nan
        # use a large number instead
        value_arrays = [np.full(x_shapes, fill_value=1e100) for _ in values]
        for seqlen_key, mbs_and_data in seqlen_dict.items():
            if not isinstance(seqlen_key, tuple):
                seqlen_key = (seqlen_key,)
            for t in mbs_and_data:
                x_coords = [mbs.index(t[0])] + [
                    seqlens[i].index(seqlen_key[i])
                    for i in range(len(seqlens))
                ]
                for i, datum in enumerate(t[1:]):
                    value_arrays[i][tuple(x_coords)] = datum
        for i in range(len(values)):
            # allow extrapolation
            interpolators.append(
                RegularGridInterpolator(
                    [mbs] + seqlens,
                    value_arrays[i],
                    method="linear",
                    bounds_error=False,
                    fill_value=None,
                )
            )
        return interpolators

    def _read_profile_data(self):
        """Read profile data from file."""
        timing_data_dict = defaultdict(lambda: defaultdict(list))
        stored_activation_data_dict = defaultdict(lambda: defaultdict(list))
        peak_activation_data_dict = defaultdict(lambda: defaultdict(list))
        # model state have fixed keys
        model_state_data_dict = {
            LMStage.EMBEDDING: [],
            LMStage.ENCODER: [],
            LMStage.DECODER: [],
        }
        max_throughput_achieved = defaultdict(lambda: defaultdict(float))
        # auxiliary functions

        def _check_or_add_to_metadata(key, value):
            if key in self._metadata:
                assert self._metadata[key] == value, (
                    "Inconsistent metadata."
                    "Profile paths have different values for key {}.".format(
                        key
                    )
                )
            else:
                self._metadata[key] = value

        def _get_data_from_line(line):
            return float(line.split(" ")[1])

        _KEY_MAP = {
            "forward_encoder": (LMStage.ENCODER, TrainStage.FORWARD),
            "backward_encoder": (LMStage.ENCODER, TrainStage.BACKWARD),
            "forward_decoder": (LMStage.DECODER, TrainStage.FORWARD),
            "backward_decoder": (LMStage.DECODER, TrainStage.BACKWARD),
            "forward_postprocess": (LMStage.POSTPROCESS, TrainStage.FORWARD),
            "backward_postprocess": (LMStage.POSTPROCESS, TrainStage.BACKWARD),
        }
        for profile_path in self.profile_paths:
            metadata = re.findall(r"\d+", os.path.basename(profile_path))
            if len(metadata) == 8:
                # Encoder decoder case
                (
                    tp_size,
                    hs,
                    ah,
                    kv,
                    ffhs,
                    enc_seqlen,
                    dec_seqlen,
                    mbs,
                ) = metadata
            elif len(metadata) == 7:
                # Decoder only case
                tp_size, hs, ah, kv, ffhs, enc_seqlen, mbs = metadata
                dec_seqlen = 0
            else:
                raise Exception(
                    "Invalid profile file name: {}".format(profile_path)
                )
            tp_size, hs, ah, kv, ffhs, enc_seqlen, dec_seqlen, mbs = list(
                map(
                    int,
                    [tp_size, hs, ah, kv, ffhs, enc_seqlen, dec_seqlen, mbs],
                )
            )
            _check_or_add_to_metadata("tp_size", tp_size)
            if "rc_full_uniform" in profile_path:
                _check_or_add_to_metadata("rc", "full")
            elif "rc_selective" in profile_path:
                _check_or_add_to_metadata("rc", "selective")
            _check_or_add_to_metadata("hs", hs)
            _check_or_add_to_metadata("ah", ah)
            _check_or_add_to_metadata("kv", kv)
            _check_or_add_to_metadata("ffhs", ffhs)
            try:
                current_line = 0
                with open(profile_path, "r") as f:
                    for line in f:
                        current_line += 1
                        if line.startswith("encoder_activation"):
                            enc_activation = _get_data_from_line(line)
                            stored_activation_data_dict[LMStage.ENCODER][
                                enc_seqlen
                            ].append((mbs, enc_activation))
                        elif line.startswith("peak_encoder_activation"):
                            peak_enc_activation = _get_data_from_line(line)
                            peak_activation_data_dict[LMStage.ENCODER][
                                enc_seqlen
                            ].append((mbs, peak_enc_activation))
                        elif line.startswith("decoder_activation"):
                            dec_activation = _get_data_from_line(line)
                            stored_activation_data_dict[LMStage.DECODER][
                                (enc_seqlen, dec_seqlen)
                            ].append((mbs, dec_activation))
                        elif line.startswith("peak_decoder_activation"):
                            peak_dec_activation = _get_data_from_line(line)
                            peak_activation_data_dict[LMStage.DECODER][
                                (enc_seqlen, dec_seqlen)
                            ].append((mbs, peak_dec_activation))
                        elif line.startswith("model"):
                            if line.startswith("model_embedding"):
                                emb_params = _get_data_from_line(line)
                                model_state_data_dict[
                                    LMStage.EMBEDDING
                                ].append(emb_params)
                            elif line.startswith("model_encoder"):
                                enc_params = _get_data_from_line(line)
                                model_state_data_dict[LMStage.ENCODER].append(
                                    enc_params
                                )
                            elif line.startswith("model_decoder"):
                                dec_params = _get_data_from_line(line)
                                model_state_data_dict[LMStage.DECODER].append(
                                    dec_params
                                )
                            else:
                                raise ValueError(
                                    "Unknown model state: {}".format(line)
                                )
                        else:
                            for key, (
                                lm_stage,
                                train_stage,
                            ) in _KEY_MAP.items():
                                if line.startswith(key):
                                    exec_time = _get_data_from_line(line)
                                    seqlen = (
                                        enc_seqlen
                                        if (
                                            lm_stage == LMStage.ENCODER
                                            or dec_seqlen == 0
                                        )
                                        else dec_seqlen
                                    )
                                    throughput = mbs * seqlen * 1.0 / exec_time
                                    stage_key = (lm_stage, train_stage)
                                    if lm_stage == LMStage.DECODER:
                                        seqlen_key = (enc_seqlen, dec_seqlen)
                                    elif lm_stage == LMStage.POSTPROCESS:
                                        if dec_seqlen == 0:
                                            seqlen_key = enc_seqlen
                                        else:
                                            seqlen_key = dec_seqlen
                                    else:
                                        seqlen_key = enc_seqlen
                                    timing_data_dict[stage_key][
                                        seqlen_key
                                    ].append((mbs, exec_time, throughput))
                                    max_throughput_achieved[stage_key][
                                        seqlen_key
                                    ] = max(
                                        max_throughput_achieved[stage_key][
                                            seqlen_key
                                        ],
                                        throughput,
                                    )
            except IOError:
                raise ValueError(
                    "Profile data file {} does not exist.".format(profile_path)
                )
        model_state_data_dict[LMStage.EMBEDDING] = sum(
            model_state_data_dict[LMStage.EMBEDDING]
        ) / len(model_state_data_dict[LMStage.EMBEDDING])
        model_state_data_dict[LMStage.ENCODER] = sum(
            model_state_data_dict[LMStage.ENCODER]
        ) / len(model_state_data_dict[LMStage.ENCODER])
        if len(model_state_data_dict[LMStage.DECODER]) > 0:
            model_state_data_dict[LMStage.DECODER] = sum(
                model_state_data_dict[LMStage.DECODER]
            ) / len(model_state_data_dict[LMStage.DECODER])
        else:
            del model_state_data_dict[LMStage.DECODER]

        for stage, seqlen_dict in timing_data_dict.items():
            for seqlen, data in seqlen_dict.items():
                timing_data_dict[stage][seqlen] = sorted(data)
        for stage, seqlen_dict in stored_activation_data_dict.items():
            for seqlen, data in seqlen_dict.items():
                stored_activation_data_dict[stage][seqlen] = sorted(data)
        for stage, seqlen_dict in peak_activation_data_dict.items():
            for seqlen, data in seqlen_dict.items():
                peak_activation_data_dict[stage][seqlen] = sorted(data)
        return (
            timing_data_dict,
            stored_activation_data_dict,
            peak_activation_data_dict,
            model_state_data_dict,
            max_throughput_achieved,
        )

    def _map_stage(self, str_stage, lm_stage_only=False):
        if not isinstance(str_stage, str):
            return str_stage
        lm_stage = None
        train_stage = None
        if "fw" in str_stage.lower() or "forward" in str_stage.lower():
            train_stage = TrainStage.FORWARD
        elif "bw" in str_stage.lower() or "backward" in str_stage.lower():
            train_stage = TrainStage.BACKWARD
        if "encoder" in str_stage.lower():
            lm_stage = LMStage.ENCODER
        elif "decoder" in str_stage.lower():
            lm_stage = LMStage.DECODER
        elif "embedding" in str_stage.lower() or "emb" in str_stage.lower():
            lm_stage = LMStage.EMBEDDING
        elif "postprocess" in str_stage.lower():
            lm_stage = LMStage.POSTPROCESS
        if lm_stage is None:
            raise ValueError("Unknown stage: {}".format(str_stage))
        if lm_stage_only:
            return lm_stage
        else:
            if train_stage is None:
                raise ValueError("Unknown stage: {}".format(str_stage))
            return lm_stage, train_stage

    def is_valid_stage(self, stage):
        stage = self._map_stage(stage)
        return stage in self.timing_data

    def valid_stages(self):
        """Return a list of valid stage names."""
        return list(self.timing_data.keys())

    def supported_sequence_lengths(self, stage, lm_stage_only=False):
        stage = self._map_stage(stage, lm_stage_only)
        if not lm_stage_only:
            return self._supported_sequence_lengths[stage]
        else:
            return self._supported_sequence_lengths_activation[stage]

    def _validate_args(self, stage, seqlen, lm_stage_only=False):
        """Validate the arguments."""
        stage = self._map_stage(stage, lm_stage_only)
        if lm_stage_only:
            if stage not in self.stored_activation_data:
                raise ValueError(
                    "Stage {} is not supported. Supported stages: {}".format(
                        stage, list(self.stored_activation_data.keys())
                    )
                )
        else:
            if not self.is_valid_stage(stage):
                raise ValueError(
                    "Stage {} is not supported. Supported stages: {}".format(
                        stage, self.valid_stages()
                    )
                )
        if (
            (isinstance(stage, tuple) and stage[0] == LMStage.DECODER)
            or stage == LMStage.DECODER
        ) and (not isinstance(seqlen, tuple) or len(seqlen) != 2):
            raise ValueError(
                "For decoder stage, "
                "seqlen must be a tuple of "
                "(enc_seqlen, dec_seqlen)."
            )

    def get_cost(
        self,
        stage,
        seq_len,
        mbs,
    ):
        """Get the computation cost of the stage in milliseconds (ms),
        under given sequence length and micro-batch size.
        """
        stage = self._map_stage(stage)
        self._validate_args(stage, seq_len)
        cache_key = (stage, seq_len, mbs)
        if cache_key not in self._interpolate_cost_cache:
            if isinstance(seq_len, (tuple, list)):
                x_coords = [mbs] + list(seq_len)
            else:
                x_coords = [mbs, seq_len]
            self._interpolate_cost_cache[cache_key] = float(
                self.exec_time_interpolators[stage](x_coords)
            )
        return self._interpolate_cost_cache[cache_key]

    def get_stored_activation(
        self,
        stage,
        seq_len,
        mbs,
    ):
        """Get the stored activation of the stage in megabytes (MB),
        under given sequence length and micro-batch size. Stored activation
        is the activation that needs to be saved in memory for backward
        pass.
        """
        stage = self._map_stage(stage, lm_stage_only=True)
        self._validate_args(stage, seq_len, lm_stage_only=True)
        cache_key = (stage, seq_len, mbs)
        if cache_key not in self._interpolate_stored_activation_cache:
            if isinstance(seq_len, (tuple, list)):
                x_coords = [mbs] + list(seq_len)
            else:
                x_coords = [mbs, seq_len]
            self._interpolate_stored_activation_cache[cache_key] = float(
                self.stored_activation_interpolators[stage](x_coords)
            )
        return self._interpolate_stored_activation_cache[cache_key]

    def get_peak_activation(
        self,
        stage,
        seq_len,
        mbs,
    ):
        """Get the peak activation of the stage in megabytes (MB),
        under given sequence length and micro-batch size.
        Peak activation is the maximum activation that is needed
        through the stage, including some intermediate data which
        is not necessarily stored for backward.
        """
        stage = self._map_stage(stage, lm_stage_only=True)
        self._validate_args(stage, seq_len, lm_stage_only=True)
        cache_key = (stage, seq_len, mbs)
        if cache_key not in self._interpolate_peak_activation_cache:
            if isinstance(seq_len, (tuple, list)):
                x_coords = [mbs] + list(seq_len)
            else:
                x_coords = [mbs, seq_len]
            self._interpolate_peak_activation_cache[cache_key] = float(
                self.peak_activation_interpolators[stage](x_coords)
            )
        return self._interpolate_peak_activation_cache[cache_key]

    def _get_param_factor(self, n_shards=1, zero_stage=0):
        """Get the parameter factor for the optimizer, which is the ratio
        between total model state size and model parameter size.
        We assume the optimizer is 16-bit Adam, see ZeRO paper for details.
        """
        params = 1
        grads = 1
        fp32_param = 2
        fp32_ema = 2
        fp32_ema_sq = 2
        if zero_stage >= 1:
            # optimizer states
            fp32_param /= n_shards
            fp32_ema /= n_shards
            fp32_ema_sq /= n_shards
        if zero_stage >= 2:
            # grads
            grads /= n_shards
        if zero_stage >= 3:
            # params
            params /= n_shards
            fp32_param /= n_shards
        return params + grads + fp32_param + fp32_ema + fp32_ema_sq

    def get_model_state(
        self, stage, n_shards=1, zero_stage=0, param_factor=None
    ):
        """Get the model state of the stage in megabytes (MB), including
        parameters, gradient buffers and optimizer states.

        param_factor is the ratio between total model state size and
        model parameter size. if param_factor is not provided, we assume
        the optimizer is 16-bit Adam, whose param_factor is computed
        based on n_shards (data parallel degree) and zero_stage.
        Otherwise we use the provided param_factor.
        """
        stage = self._map_stage(stage, lm_stage_only=True)
        if stage not in self.model_state_data:
            raise ValueError(
                "Stage {} is not supported. Supported stages: {}".format(
                    stage, list(self.model_state_data.keys())
                )
            )
        if param_factor is None:
            param_factor = self._get_param_factor(
                n_shards=n_shards, zero_stage=zero_stage
            )
        return self.model_state_data[stage] * param_factor

    def serialize(self):
        return pickle.dumps(
            (
                dict(self.timing_data),
                dict(self.stored_activation_data),
                dict(self.peak_activation_data),
                dict(self.model_state_data),
                dict(self.max_throughput_achieved),
            )
        )

    @classmethod
    def deserialize(cls, serialized):
        (
            timing_data,
            stored_activation_data,
            peak_activation_data,
            model_state_data,
            max_throughput_achieved,
        ) = pickle.loads(serialized)
        return cls(
            _timing_data=timing_data,
            _stored_activation_data=stored_activation_data,
            _peak_activation_data=peak_activation_data,
            _model_state_data=model_state_data,
            _max_throughput_achieved=max_throughput_achieved,
        )


class ProfileBasedCostModelWithRC(object):
    """
    Wrapper class for multiple ProfileBasedCostModel objects, one for each
    tensor parallel degree and recomputation method.
    """

    def __init__(
        self,
        profile_paths=None,
        _serialized_cms: Optional[Dict[Tuple[int, str], bytes]] = None,
    ) -> None:
        self.cost_models: dict[str, ProfileBasedCostModel] = {}
        if _serialized_cms is not None:
            for cm_key, serialized_cm in _serialized_cms.items():
                self.cost_models[cm_key] = ProfileBasedCostModel.deserialize(
                    serialized_cm
                )
            return
        if not isinstance(profile_paths, list):
            # profile_paths is a dir
            assert os.path.isdir(profile_paths), (
                f"Profile path {profile_paths} is not a directory "
                "or list of paths"
            )
            profile_paths = [
                os.path.join(profile_paths, x)
                for x in os.listdir(profile_paths)
                if x.startswith("microbench") and x.endswith("txt")
            ]
        # separate paths by cost model key (tp_size, rc_type)
        self.per_key_profile_paths = defaultdict(list)
        for path in profile_paths:
            cm_key = self._parse_cm_key(path)
            self.per_key_profile_paths[cm_key].append(path)
        for cm_key, paths in self.per_key_profile_paths.items():
            self.cost_models[cm_key] = ProfileBasedCostModel(paths)

    def _parse_cm_key(self, filename):
        basename = os.path.basename(filename)
        if "rc_full_uniform" in basename:
            rc_type = "full"
        elif "rc_selective" in basename:
            rc_type = "selective"
        else:
            rc_type = "none"
        tp_size = int(basename.split("_")[1][2:])
        return tp_size, rc_type

    def _check_valid_cm_key(self, cm_key):
        assert (
            cm_key in self.cost_models
        ), f"Key {cm_key} not recorded in profile."

    def is_valid_stage(self, tp_size, rc_type, stage):
        self._check_valid_cm_key((tp_size, rc_type))
        return self.cost_models[(tp_size, rc_type)].is_valid_stage(stage)

    def valid_stages(self, tp_size, rc_type):
        self._check_valid_cm_key((tp_size, rc_type))
        return self.cost_models[(tp_size, rc_type)].valid_stages()

    def supported_sequence_lengths(self, tp_size, rc_type, stage):
        self._check_valid_cm_key((tp_size, rc_type))
        return self.cost_models[(tp_size, rc_type)].supported_sequence_lengths(
            stage
        )

    def get_cost(
        self,
        tp_size,
        rc_type,
        stage,
        seq_len,
        mbs,
    ):
        """Select the corresponding cost model based on TP degree and
        recomputation type and get the computation cost.
        """
        self._check_valid_cm_key((tp_size, rc_type))
        return self.cost_models[(tp_size, rc_type)].get_cost(
            stage, seq_len, mbs
        )

    def get_stored_activation(self, tp_size, rc_type, stage, seq_len, mbs):
        """Select the corresponding cost model based on TP degree and
        recomputation type and get the stored activation.
        """
        self._check_valid_cm_key((tp_size, rc_type))
        return self.cost_models[(tp_size, rc_type)].get_stored_activation(
            stage, seq_len, mbs
        )

    def get_peak_activation(self, tp_size, rc_type, stage, seq_len, mbs):
        """Select the corresponding cost model based on TP degree and
        recomputation type and get the peak activation.
        """
        self._check_valid_cm_key((tp_size, rc_type))
        return self.cost_models[(tp_size, rc_type)].get_peak_activation(
            stage, seq_len, mbs
        )

    def get_model_state(
        self,
        tp_size,
        rc_type,
        stage,
        n_shards=1,
        zero_stage=0,
        param_factor=None,
    ):
        """Select the corresponding cost model based on TP degree and
        recomputation type and get the model state.
        """
        self._check_valid_cm_key((tp_size, rc_type))
        return self.cost_models[(tp_size, rc_type)].get_model_state(
            stage,
            n_shards=n_shards,
            zero_stage=zero_stage,
            param_factor=param_factor,
        )

    def get_raw_cost_model(self, tp_size, rc_type):
        """Get the raw cost model for the given TP degree and recomputation
        type.
        """
        self._check_valid_cm_key((tp_size, rc_type))
        return self.cost_models[(tp_size, rc_type)]

    def save(self, path):
        serialized_dict = {}
        for cm_key, cost_model in self.cost_models.items():
            serialized_dict[cm_key] = cost_model.serialize()
        with open(path, "wb") as f:
            pickle.dump(serialized_dict, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            serialized_dict = pickle.load(f)
        return cls(_serialized_cms=serialized_dict)
