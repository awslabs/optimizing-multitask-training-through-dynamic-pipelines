# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynapipe.model import TransformerModelSpec, get_uniform_cluster
from dynapipe.pipe.utils import validate_device_assignment


def get_uniform_model(n_encoder_layers, n_decoder_layers):
    return TransformerModelSpec(
        n_encoder_layers, n_decoder_layers, 1024, 128, 65536, 128
    )


@pytest.mark.parametrize(
    "model_spec, cluster_spec, device_assignment, expected",
    [
        # linear
        (
            get_uniform_model(4, 4),
            get_uniform_cluster(8),
            [0, 1, 2, 3, 4, 5, 6, 7],
            (
                "linear",
                set(["wait-free-cyclic", "1F1B"]),
                1,
                1,
            ),
        ),
        # interleaved 1
        (
            get_uniform_model(4, 4),
            get_uniform_cluster(4),
            [0, 1, 2, 3, 0, 1, 2, 3],
            (
                "interleaved",
                set(["wait-free-cyclic", "interleaved-1F1B"]),
                1,
                2,
            ),
        ),
        # interleaved 2
        (
            get_uniform_model(8, 4),
            get_uniform_cluster(4),
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            (
                "interleaved",
                set(["wait-free-cyclic", "interleaved-1F1B"]),
                1,
                3,
            ),
        ),
        # multiple layer per virtual layer
        (
            get_uniform_model(4, 4),
            get_uniform_cluster(4),
            [0, 0, 1, 1, 2, 2, 3, 3],
            (
                "linear",
                set(["wait-free-cyclic", "1F1B"]),
                2,
                1,
            ),
        ),
        # multiple layer per virtual layer, interleaved
        (
            get_uniform_model(8, 8),
            get_uniform_cluster(4),
            [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3],
            (
                "interleaved",
                set(["wait-free-cyclic", "interleaved-1F1B"]),
                2,
                2,
            ),
        ),
        # decoder only models like GPT (note we specify all layers as
        # encoder layers since we assume decoder layers are those containing
        # encoder-decoder attention)
        (
            get_uniform_model(4, 0),
            get_uniform_cluster(4),
            [0, 1, 2, 3],
            (
                "linear",
                set(["wait-free-cyclic", "1F1B"]),
                1,
                1,
            ),
        ),
        (
            get_uniform_model(8, 0),
            get_uniform_cluster(4),
            [0, 0, 1, 1, 2, 2, 3, 3],
            (
                "linear",
                set(["wait-free-cyclic", "1F1B"]),
                2,
                1,
            ),
        ),
        # single gpu
        (
            get_uniform_model(4, 0),
            get_uniform_cluster(1),
            [0, 0, 0, 0],
            (
                "linear",
                set(["wait-free-cyclic"]),
                4,
                1,
            ),
        ),
        # other
        (
            get_uniform_model(4, 4),
            get_uniform_cluster(4),
            [0, 1, 2, 3, 3, 2, 1, 0],
            (
                "other",
                set(["wait-free-cyclic"]),
                1,
                1,
            ),
        ),
    ],
)
def test_valid_device_assignments(
    model_spec, cluster_spec, device_assignment, expected
):
    (
        device_assignment_type,
        valid_schedule_methods,
        n_actual_layers_per_virtual_layer,
        n_chunks_per_device,
    ) = validate_device_assignment(model_spec, cluster_spec, device_assignment)
    valid_schedule_methods = set(valid_schedule_methods)
    assert device_assignment_type == expected[0]
    assert valid_schedule_methods == expected[1]
    assert n_actual_layers_per_virtual_layer == expected[2]
    assert n_chunks_per_device == expected[3]


def test_incorrect_appear_order():
    with pytest.raises(AssertionError):
        validate_device_assignment(
            get_uniform_model(4, 4),
            get_uniform_cluster(4),
            [0, 2, 1, 3, 0, 2, 1, 3],
        )


def test_incorrect_n_devices():
    with pytest.raises(AssertionError):
        validate_device_assignment(
            get_uniform_model(4, 4),
            get_uniform_cluster(8),
            [0, 1, 2, 3, 0, 1, 2, 3],
        )


def test_incorrect_interleaving_no_decoder():
    with pytest.raises(NotImplementedError):
        validate_device_assignment(
            get_uniform_model(0, 0),
            get_uniform_cluster(4),
            [0, 1, 2, 3, 0, 1, 2, 3],
        )


if __name__ == "__main__":
    pytest.main([__file__])
