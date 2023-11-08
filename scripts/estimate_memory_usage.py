# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from dynapipe.utils.memory_utils import TransformerMemoryModel

# Scripts for plotting memory usage of transformer models with different
# sequence lengths abd microbatch sizes.


def main():
    hidden_dim = 8192
    num_attn_heads = 64
    kv_channels = 128
    mlp_hidden_dim = 6560
    bytes_per_element = 2
    tp_size = 8

    prod_ = 1024 * 12
    js = []
    for enc_sec_len in [512, 768, 1024, 1536, 2048]:
        dec_sec_len = enc_sec_len / 2
        mbs = prod_ / enc_sec_len
        enc_mem_model = TransformerMemoryModel(
            batch_size=mbs,
            sequence_length=enc_sec_len,
            hidden_dim=hidden_dim,
            num_attn_heads=num_attn_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            kv_channels=kv_channels,
            bytes_per_element=bytes_per_element,
            optimizer_state_multiplier=12,
            tp_size=tp_size,
        )
        dec_mem_model = TransformerMemoryModel(
            batch_size=mbs,
            sequence_length=dec_sec_len,
            hidden_dim=hidden_dim,
            num_attn_heads=num_attn_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            kv_channels=kv_channels,
            bytes_per_element=bytes_per_element,
            optimizer_state_multiplier=12,
            tp_size=tp_size,
            is_decoder=True,
        )
        j = {
            "enc_seq_len": enc_sec_len,
            "dec_seq_len": dec_sec_len,
            "mbs": mbs,
            "enc_per_layer_output_memory": enc_mem_model.get_output_memory(),
            "enc_per_layer_activation_memory": enc_mem_model.get_activation_memory(),  # noqa
            "enc_per_layer_model_state_memory": enc_mem_model.get_model_state_memory(),  # noqa
            "dec_per_layer_output_memory": dec_mem_model.get_output_memory(),  # noqa
            "dec_per_layer_activation_memory": dec_mem_model.get_activation_memory(),  # noqa
            "dec_per_layer_model_state_memory": dec_mem_model.get_model_state_memory(),  # noqa
        }
        js.append(j)

    df = pd.DataFrame(js)
    df.to_csv("out.csv", index=False)
    print(df)
    return df


if __name__ == "__main__":
    main()
