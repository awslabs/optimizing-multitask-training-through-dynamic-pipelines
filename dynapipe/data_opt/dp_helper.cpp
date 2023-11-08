/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <limits>
#include <unordered_map>

#ifdef HELPER_DEBUG
#include <chrono>
#include <iostream>
#endif

namespace py = pybind11;

using Sequence = std::vector<int>;
using Microbatch = std::vector<Sequence>;
using Minibatch = std::vector<Microbatch>;
using SubseqInfo = std::tuple<float, float, float, bool>;
using SampleWithId = std::tuple<int, int, int>;
using ObjInfo =
    std::tuple<float, float, float, std::vector<float>, std::vector<float>>;
using MicrobatchTimeCosts = std::vector<float>;

static const int MIN_TMAX_INTERVAL = 5;

std::tuple<ObjInfo, Minibatch, MicrobatchTimeCosts>
cpp_consecutive_partition_dp(
    py::object cost_model, const int32_t num_stages,
    const int32_t n_chunks_per_device, const int32_t n_layers_per_stage,
    const int32_t dp_size, const float per_mb_stored_activation_limit,
    const float peak_activation_limit,
    const std::vector<SampleWithId> samples_with_ids, bool enable_packing,
    int round_seqlen_multiple, int len_pack_sep_tokens,
    int len_decoder_additional_tokens, int seqlen_offset) {
  std::vector<std::vector<SubseqInfo>> subseq_cache;
  std::vector<std::vector<bool>> subseq_cache_valid;

  subseq_cache.resize(samples_with_ids.size());
  subseq_cache_valid.resize(samples_with_ids.size());

  for (int i = 0; i < static_cast<int>(samples_with_ids.size()); i++) {
    subseq_cache[i].resize(samples_with_ids.size() + 1);
    subseq_cache_valid[i].resize(samples_with_ids.size() + 1);
    for (int j = 0; j <= static_cast<int>(samples_with_ids.size()); j++) {
      subseq_cache_valid[i][j] = false;
    }
  }

  auto _round_seqlen = [&](int seqlen, bool is_decoder = false) {
    if (seqlen == 0) {
      return 0;
    }
    if (is_decoder) {
      seqlen += len_decoder_additional_tokens;
    }
    assert(seqlen > seqlen_offset);
    seqlen -= seqlen_offset;
    return (seqlen + round_seqlen_multiple - 1) / round_seqlen_multiple *
           round_seqlen_multiple;
  };

  auto _calculate_packed_sample_ids = [&](int i, int j, int input_seq_len,
                                          int decoder_seq_len, int mbs,
                                          float time_cost) {
    int enc_target_seqlen = input_seq_len;
    int dec_target_seqlen =
        std::max(0, decoder_seq_len - len_decoder_additional_tokens);
    bool no_dec_tokens = (decoder_seq_len == 0);
    if (no_dec_tokens) {
      // no dec tokens, set to max so it will not affect calculation
      dec_target_seqlen = std::numeric_limits<int>::max();
    }
    Microbatch packed_sample_ids;
    packed_sample_ids.reserve(mbs);
    int packed_enc_seqlen = -1, packed_dec_seqlen = -1;

    std::vector<int> min_enc_seqlen_after_i;
    std::vector<int> min_dec_seqlen_after_i;
    min_enc_seqlen_after_i.resize(mbs);
    min_dec_seqlen_after_i.resize(mbs);
    min_enc_seqlen_after_i[mbs - 1] = std::get<0>(samples_with_ids[j - 1]);
    min_dec_seqlen_after_i[mbs - 1] = std::get<1>(samples_with_ids[j - 1]);
    for (int k = mbs - 2; k >= 0; k--) {
      min_enc_seqlen_after_i[k] = std::min(
          min_enc_seqlen_after_i[k + 1], std::get<0>(samples_with_ids[i + k]));
      min_dec_seqlen_after_i[k] = std::min(
          min_dec_seqlen_after_i[k + 1], std::get<1>(samples_with_ids[i + k]));
    }
    std::vector<bool> packed(mbs, false);
    for (int k = 0; k < mbs; k++) {
      if (packed[k]) {
        continue;
      }
      Sequence current_sequence;
      current_sequence.reserve(mbs - k);
      current_sequence.push_back(std::get<2>(samples_with_ids[i + k]));
      int current_enc_seqlen = std::get<0>(samples_with_ids[i + k]);
      int current_dec_seqlen = std::get<1>(samples_with_ids[i + k]);
      for (int l = k + 1; l < mbs; l++) {
        if (packed[l]) {
          continue;
        }
        if (current_enc_seqlen + min_enc_seqlen_after_i[l] +
                    len_pack_sep_tokens >
                enc_target_seqlen ||
            current_dec_seqlen + min_dec_seqlen_after_i[l] +
                    len_pack_sep_tokens >
                dec_target_seqlen) {
          // skip packing if the remaining samples all have longer sequence
          // length than required
          break;
        }
        int current_sample_enc_seqlen = std::get<0>(samples_with_ids[i + l]);
        int current_sample_dec_seqlen = std::get<1>(samples_with_ids[i + l]);
        if (current_sample_enc_seqlen + current_enc_seqlen +
                    len_pack_sep_tokens <=
                enc_target_seqlen &&
            current_sample_dec_seqlen + current_dec_seqlen +
                    len_pack_sep_tokens <=
                dec_target_seqlen) {
          // pack this sample
          current_sequence.push_back(std::get<2>(samples_with_ids[i + l]));
          current_enc_seqlen += len_pack_sep_tokens + current_sample_enc_seqlen;
          current_dec_seqlen += len_pack_sep_tokens + current_sample_dec_seqlen;
          // we do not recalculate the min_enc_seqlen_after_i and
          // min_dec_seqlen_after_i since that's too slow
          packed[l] = true;
        }
      }
      packed_sample_ids.emplace_back(std::move(current_sequence));
      packed_enc_seqlen = std::max(packed_enc_seqlen, current_enc_seqlen);
      packed_dec_seqlen = std::max(packed_dec_seqlen, current_dec_seqlen);
    }
    packed_enc_seqlen = _round_seqlen(packed_enc_seqlen);
    packed_dec_seqlen = _round_seqlen(packed_dec_seqlen, true);
    // calculate packed cost
    float packed_enc_cost =
        cost_model
            .attr("get_cost")("Encoder FW", packed_enc_seqlen,
                              packed_sample_ids.size())
            .cast<float>();
    packed_enc_cost += cost_model
                           .attr("get_cost")("Encoder BW", packed_enc_seqlen,
                                             packed_sample_ids.size())
                           .cast<float>();
    packed_enc_cost *= n_layers_per_stage;

    float packed_dec_cost = 0.0f;
    if (!no_dec_tokens) {
      // only calculate decoder cost if there are decoder tokens
      packed_dec_cost +=
          cost_model
              .attr("get_cost")(
                  "Decoder FW",
                  std::make_tuple(packed_enc_seqlen, packed_dec_seqlen),
                  packed_sample_ids.size())
              .cast<float>();
      packed_dec_cost +=
          cost_model
              .attr("get_cost")(
                  "Decoder BW",
                  std::make_tuple(packed_enc_seqlen, packed_dec_seqlen),
                  packed_sample_ids.size())
              .cast<float>();
    }
    packed_dec_cost *= n_layers_per_stage;

    // add postprocess cost
    if (no_dec_tokens) {
      packed_enc_cost +=
          cost_model
              .attr("get_cost")("Postprocess FW", packed_enc_seqlen,
                                packed_sample_ids.size())
              .cast<float>();
      packed_enc_cost +=
          cost_model
              .attr("get_cost")("Postprocess BW", packed_enc_seqlen,
                                packed_sample_ids.size())
              .cast<float>();
    } else {
      // postprocess cost for encoder-decoder model uses decoder key
      packed_dec_cost +=
          cost_model
              .attr("get_cost")("Postprocess FW", packed_dec_seqlen,
                                packed_sample_ids.size())
              .cast<float>();
      packed_dec_cost +=
          cost_model
              .attr("get_cost")("Postprocess BW", packed_dec_seqlen,
                                packed_sample_ids.size())
              .cast<float>();
    }
    float packed_cost = 0.0f;
    if (n_chunks_per_device == 1) {
      // no interleaving
      packed_cost = std::max(packed_enc_cost, packed_dec_cost);
    } else {
      packed_cost =
          (packed_enc_cost + packed_dec_cost) * (n_chunks_per_device / 2);
    }
    if (packed_cost < time_cost) {
      return std::make_tuple(packed_cost, std::move(packed_sample_ids));
    }
    Microbatch sample_ids;
    // no samples can be packed
    sample_ids.reserve(mbs);
    for (int k = i; k < j; k++) {
      sample_ids.push_back({std::get<2>(samples_with_ids[k])});
    }
    return std::make_tuple(time_cost, std::move(sample_ids));
  };

  auto _get_time_and_memory_for_microbatch = [&](int i, int j) {
    if (i == j) {
      return std::make_tuple(0.0f, 0.0f, 0.0f, false);
    }
    int mbs = j - i;
    int input_seq_len = 0, decoder_seq_len = 0;
    for (int k = i; k < j; k++) {
      auto& sample = samples_with_ids[k];
      input_seq_len = std::max(input_seq_len, std::get<0>(sample));
      decoder_seq_len = std::max(decoder_seq_len, std::get<1>(sample));
    }
    input_seq_len = _round_seqlen(input_seq_len);
    decoder_seq_len = _round_seqlen(decoder_seq_len, true);
    // encoder cost
    float enc_time_cost =
        cost_model.attr("get_cost")("Encoder FW", input_seq_len, mbs)
            .cast<float>();
    enc_time_cost +=
        cost_model.attr("get_cost")("Encoder BW", input_seq_len, mbs)
            .cast<float>();
    enc_time_cost *= n_layers_per_stage;
    // decoder cost
    float dec_time_cost = 0.0f;
    if (decoder_seq_len > 0) {
      dec_time_cost +=
          cost_model
              .attr("get_cost")("Decoder FW",
                                std::make_tuple(input_seq_len, decoder_seq_len),
                                mbs)
              .cast<float>();
      dec_time_cost +=
          cost_model
              .attr("get_cost")("Decoder BW",
                                std::make_tuple(input_seq_len, decoder_seq_len),
                                mbs)
              .cast<float>();
    }
    dec_time_cost *= n_layers_per_stage;
    // postprocessing cost
    if (decoder_seq_len > 0) {
      // postprocess cost for encoder-decoder model uses decoder key
      dec_time_cost +=
          cost_model.attr("get_cost")("Postprocess FW", decoder_seq_len, mbs)
              .cast<float>();
      dec_time_cost +=
          cost_model.attr("get_cost")("Postprocess BW", decoder_seq_len, mbs)
              .cast<float>();
    } else {
      enc_time_cost +=
          cost_model.attr("get_cost")("Postprocess FW", input_seq_len, mbs)
              .cast<float>();
      enc_time_cost +=
          cost_model.attr("get_cost")("Postprocess BW", input_seq_len, mbs)
              .cast<float>();
    }
    float time_cost = 0.0f;
    if (n_chunks_per_device == 1) {
      // no interleaving
      time_cost = std::max(enc_time_cost, dec_time_cost);
    } else {
      time_cost = (enc_time_cost + dec_time_cost) * (n_chunks_per_device / 2);
    }
    bool packed = false;
    if (enable_packing) {
      auto packed_result = _calculate_packed_sample_ids(
          i, j, input_seq_len, decoder_seq_len, mbs, time_cost);
      float packed_time_cost = std::get<0>(packed_result);
      if (packed_time_cost < time_cost) {
        time_cost = packed_time_cost;
        auto sample_ids = std::move(std::get<1>(packed_result));
        mbs = sample_ids.size();
        packed = true;
      }
    }
    float stored_activation;
    float peak_activation;
    // peak activation always use max of encoder and decoder
    float encoder_peak_activation =
        cost_model.attr("get_peak_activation")("Encoder", input_seq_len, mbs)
            .cast<float>();
    float decoder_peak_activation = 0.0f;
    if (decoder_seq_len > 0) {
      decoder_peak_activation =
          cost_model
              .attr("get_peak_activation")(
                  "Decoder", std::make_tuple(input_seq_len, decoder_seq_len),
                  mbs)
              .cast<float>();
    }
    peak_activation =
        std::max(encoder_peak_activation, decoder_peak_activation);
    // stored activation
    float encoder_stored_activation =
        cost_model.attr("get_stored_activation")("Encoder", input_seq_len, mbs)
            .cast<float>();
    float decoder_stored_activation = 0.0f;
    if (decoder_seq_len > 0) {
      decoder_stored_activation =
          cost_model
              .attr("get_stored_activation")(
                  "Decoder", std::make_tuple(input_seq_len, decoder_seq_len),
                  mbs)
              .cast<float>();
    }
    if (n_chunks_per_device == 1) {
      // use max of encoder and decoder
      stored_activation =
          std::max(encoder_stored_activation, decoder_stored_activation);
    } else {
      // use average of encoder and decoder
      stored_activation = encoder_stored_activation + decoder_stored_activation;
      stored_activation = stored_activation * n_chunks_per_device / 2;
    }
    // add peak activation on top of previous stored layers
    peak_activation += stored_activation * (n_layers_per_stage - 1);
    // scale stored activation by number of layers
    stored_activation = stored_activation * n_layers_per_stage;
    return std::make_tuple(std::move(time_cost), std::move(stored_activation),
                           std::move(peak_activation), std::move(packed));
  };

  auto _get_mb_info_cached = [&](int i, int j) -> SubseqInfo& {
    assert(i >= 0 && i < static_cast<int>(samples_with_ids.size()));
    assert(j > i && j <= static_cast<int>(samples_with_ids.size()));
    if (!subseq_cache_valid[i][j]) {
      subseq_cache[i][j] = _get_time_and_memory_for_microbatch(i, j);
      subseq_cache_valid[i][j] = true;
    }
    return subseq_cache[i][j];
  };

#ifdef HELPER_DEBUG
  std::chrono::steady_clock::time_point begin_tmax =
      std::chrono::steady_clock::now();
#endif
  std::vector<float> all_possible_tmax;
  std::vector<int> valid_range_per_i;
  for (int i = 0; i < static_cast<int>(samples_with_ids.size()); i++) {
    valid_range_per_i.push_back(samples_with_ids.size() + 1);
  }
  all_possible_tmax.reserve(samples_with_ids.size() *
                            (samples_with_ids.size() + 1) / 2);
  for (int i = 0; i < static_cast<int>(samples_with_ids.size()); i++) {
    for (int j = i + 1; j <= static_cast<int>(samples_with_ids.size()); j++) {
      auto& subseq_info = _get_mb_info_cached(i, j);
      if (std::get<1>(subseq_info) <= per_mb_stored_activation_limit &&
          std::get<2>(subseq_info) <= peak_activation_limit) {
        all_possible_tmax.push_back(std::get<0>(subseq_info));
      } else {
        valid_range_per_i[i] = j;
        break;
      }
    }
  }
  std::sort(all_possible_tmax.begin(), all_possible_tmax.end());
  std::vector<float> filtered_tmax;
  filtered_tmax.reserve(all_possible_tmax.size());
  float last_tmax_preserved = -1;
  for (int i = 0; i < static_cast<int>(all_possible_tmax.size()); i++) {
    float tmax = all_possible_tmax[i];
    if (std::isinf(tmax) || std::isnan(tmax)) {
      break;
    }
    if (i != 0 && tmax - last_tmax_preserved <= MIN_TMAX_INTERVAL) {
      continue;
    }
    filtered_tmax.push_back(tmax);
    last_tmax_preserved = tmax;
  }
#ifdef HELPER_DEBUG
  std::chrono::steady_clock::time_point end_tmax =
      std::chrono::steady_clock::now();
  std::cout << "Computing possible tmax used "
            << std::chrono::duration_cast<std::chrono::microseconds>(end_tmax -
                                                                     begin_tmax)
                   .count()
            << " [µs]" << std::endl;

  std::chrono::steady_clock::time_point begin_dp =
      std::chrono::steady_clock::now();
#endif
  std::vector<std::tuple<float, std::vector<int>>> subseq_infos_for_tmax;
  for (int tmax_idx = 0; tmax_idx < static_cast<int>(filtered_tmax.size());
       tmax_idx++) {
    float tmax = filtered_tmax[tmax_idx];
    std::vector<std::tuple<float, int>> f_values = {{0.0f, -1}};
    for (int n = 1; n < static_cast<int>(samples_with_ids.size()) + 1; n++) {
      float min_time_if_partition_at_i = std::numeric_limits<float>::infinity();
      int min_i = -1;
      for (int i = 0; i < n; i++) {
        if (valid_range_per_i[i] <= n) {
          continue;
        }
        auto& subseq_info = _get_mb_info_cached(i, n);
        float mb_time = std::get<0>(subseq_info);
        float mb_stored_memory = std::get<1>(subseq_info);
        float mb_peak_memory = std::get<2>(subseq_info);
        if (mb_stored_memory > per_mb_stored_activation_limit ||
            mb_peak_memory > peak_activation_limit || mb_time > tmax) {
          continue;
        }
        auto f_at_i = f_values[i];
        if (i != 0 && std::get<1>(f_at_i) < 0) {
          // this partition is not valid
          continue;
        }
        auto time_if_partition_at_i = mb_time + std::get<0>(f_at_i);
        if (time_if_partition_at_i < min_time_if_partition_at_i) {
          min_time_if_partition_at_i = time_if_partition_at_i;
          min_i = i;
        }
      }
      if (std::isinf(min_time_if_partition_at_i) ||
          std::isnan(min_time_if_partition_at_i)) {
        f_values.push_back({0.0f, -1});
      } else {
        f_values.push_back({min_time_if_partition_at_i, min_i});
      }
    }
    if (std::get<1>(f_values.back()) >= 0) {
      // valid partition found for this tmax, reconstruct slicing points
      std::vector<int> slicing_points;
      int current_idx = f_values.size() - 1;
      while (std::get<1>(f_values[current_idx]) >= 0) {
        slicing_points.push_back(std::get<1>(f_values[current_idx]));
        current_idx = std::get<1>(f_values[current_idx]);
      }
      std::reverse(slicing_points.begin(), slicing_points.end());
      subseq_infos_for_tmax.push_back(
          {std::get<0>(f_values.back()), slicing_points});
    } else {
      // no valid partition found for this tmax
      subseq_infos_for_tmax.push_back(
          {std::numeric_limits<float>::infinity(), {}});
    }
  }
#ifdef HELPER_DEBUG
  std::chrono::steady_clock::time_point end_dp =
      std::chrono::steady_clock::now();
  std::cout << "Computing dp used "
            << std::chrono::duration_cast<std::chrono::microseconds>(end_dp -
                                                                     begin_dp)
                   .count()
            << " [µs]" << std::endl;

  std::chrono::steady_clock::time_point begin_recover =
      std::chrono::steady_clock::now();
#endif
  float objective_value = std::numeric_limits<float>::infinity();
  int objective_value_idx = -1;
  for (int i = 0; i < static_cast<int>(filtered_tmax.size()); i++) {
    float obj = (num_stages - 1) * (filtered_tmax[i] / n_chunks_per_device) +
                std::get<0>(subseq_infos_for_tmax[i]) / dp_size;
    if (obj < objective_value) {
      objective_value = obj;
      objective_value_idx = i;
    }
  }
  if (objective_value_idx < 0) {
    // no valid partition found
    ObjInfo obj_result = std::make_tuple(
        objective_value, -1, -1, std::vector<float>(), std::vector<float>());
    return std::make_tuple(obj_result, Minibatch(), MicrobatchTimeCosts());
  }
  auto slicing_points = std::get<1>(subseq_infos_for_tmax[objective_value_idx]);
  std::vector<float> stored_memory_per_mb;
  std::vector<float> peak_memory_per_mb;

  auto _recover_sample_ids = [&](int i, int j, bool packed) -> Microbatch {
    int mbs = j - i;
    int input_seq_len = 0, decoder_seq_len = 0;
    for (int k = i; k < j; k++) {
      auto& sample = samples_with_ids[k];
      input_seq_len = std::max(input_seq_len, std::get<0>(sample));
      decoder_seq_len = std::max(decoder_seq_len, std::get<1>(sample));
    }
    input_seq_len = _round_seqlen(input_seq_len);
    decoder_seq_len = _round_seqlen(decoder_seq_len, true);
    if (packed) {
      auto packed_result = _calculate_packed_sample_ids(
          i, j, input_seq_len, decoder_seq_len, mbs,
          std::numeric_limits<float>::infinity());
      return std::get<1>(packed_result);
    }
    // no packing
    Microbatch sample_ids;
    sample_ids.reserve(mbs);
    for (int k = i; k < j; k++) {
      sample_ids.push_back({std::get<2>(samples_with_ids[k])});
    }
    return sample_ids;
  };

  Minibatch result;
  MicrobatchTimeCosts microbatch_costs;
  for (int i = 0; i < static_cast<int>(slicing_points.size()) - 1; i++) {
    auto& mb_info =
        _get_mb_info_cached(slicing_points[i], slicing_points[i + 1]);
    auto stored_memory = std::get<1>(mb_info);
    auto peak_memory = std::get<2>(mb_info);
    stored_memory_per_mb.push_back(stored_memory);
    peak_memory_per_mb.push_back(peak_memory);
    microbatch_costs.push_back(std::get<0>(mb_info));
    bool packed = std::get<3>(mb_info);
    result.emplace_back(
        _recover_sample_ids(slicing_points[i], slicing_points[i + 1], packed));
  }
  if (!slicing_points.empty()) {
    auto& mb_info =
        _get_mb_info_cached(slicing_points.back(), samples_with_ids.size());
    stored_memory_per_mb.push_back(std::get<1>(mb_info));
    peak_memory_per_mb.push_back(std::get<2>(mb_info));
    microbatch_costs.push_back(std::get<0>(mb_info));
    bool packed = std::get<3>(mb_info);
    result.emplace_back(_recover_sample_ids(slicing_points.back(),
                                            samples_with_ids.size(), packed));
  }
  ObjInfo obj_result =
      std::make_tuple(objective_value, filtered_tmax[objective_value_idx],
                      std::get<0>(subseq_infos_for_tmax[objective_value_idx]),
                      stored_memory_per_mb, peak_memory_per_mb);
#ifdef HELPER_DEBUG
  std::chrono::steady_clock::time_point end_recover =
      std::chrono::steady_clock::now();
  std::cout << "Recoder indices used "
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   end_recover - begin_recover)
                   .count()
            << " [µs]" << std::endl;
#endif
  return std::make_tuple(obj_result, result, microbatch_costs);
}

PYBIND11_MODULE(dp_helper, m) {
  m.doc() = "C++ implementation of the DP algorithm";
  m.def("cpp_consecutive_partition_dp", &cpp_consecutive_partition_dp,
        "DP algorithm to generate microbatch partitions", py::arg("cost_model"),
        py::arg("num_stages"), py::arg("n_chunks_per_device"),
        py::arg("n_layers_per_stage"), py::arg("dp_size"),
        py::arg("per_mb_stored_activation_limit"),
        py::arg("peak_activation_limit"), py::arg("samples_with_ids"),
        py::arg("enable_packing") = true, py::arg("round_seqlen_multiple") = 8,
        py::arg("len_pack_sep_tokens") = 1,
        py::arg("len_decoder_additional_tokens") = 2,
        py::arg("seqlen_offset") = 0);
}