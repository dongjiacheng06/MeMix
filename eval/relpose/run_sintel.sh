#!/bin/bash

set -euo pipefail

workdir='.'
default_model_names="cut ttt ttsa cut_memix ttt_memix ttsa_memix"
read -r -a model_names <<< "${MODEL_NAMES:-$default_model_names}"
data="${EVAL_DATASET:-sintel}"
ckpt_name="${CKPT_NAME:-cut3r_512_dpt_4_64}"
model_weights="${WEIGHTS:-${workdir}/src/${ckpt_name}.pth}"
num_processes="${NUM_PROCESSES:-1}"
port="${PORT:-29552}"
eval_protocol="${EVAL_PROTOCOL:-short}"
seq_lengths="${SEQ_LENGTHS:-}"

views_label() {
    if [[ "$1" == "0" ]]; then
        echo "all"
    else
        echo "$1"
    fi
}

case "${eval_protocol}" in
    short)
        seq_lengths="50"
        preset_start=0
        preset_stride=1
        ;;
    long)
        if [[ -z "${seq_lengths}" ]]; then
            echo "EVAL_PROTOCOL=long for sintel requires SEQ_LENGTHS, e.g. SEQ_LENGTHS=\"100 200\"" >&2
            exit 1
        fi
        preset_start=0
        preset_stride=1
        ;;
    *)
        echo "Unsupported EVAL_PROTOCOL: ${eval_protocol}" >&2
        exit 1
        ;;
esac

for model_name in "${model_names[@]}"; do
    for seq_len in ${seq_lengths}; do
        if [[ "${eval_protocol}" == "short" ]]; then
            preset_views=50
            dataset_dir="${data}"
        else
            preset_views="${seq_len}"
            dataset_dir="${data}_long_${seq_len}"
        fi

        start_frame="${START_FRAME:-$preset_start}"
        pose_eval_stride="${POSE_EVAL_STRIDE:-$preset_stride}"
        max_views="${MAX_VIEWS:-$preset_views}"

        if [[ "${start_frame}" != "${preset_start}" || "${pose_eval_stride}" != "${preset_stride}" || "${max_views}" != "${preset_views}" ]]; then
            dataset_dir="${dataset_dir}_start${start_frame}_stride${pose_eval_stride}_views$(views_label "${max_views}")"
        fi

        output_dir="${workdir}/eval_results/relpose/${dataset_dir}/${model_name}"
        echo "${output_dir}"

        accelerate launch --num_processes "${num_processes}" --main_process_port "${port}" eval/relpose/launch.py \
            --weights "${model_weights}" \
            --output_dir "${output_dir}" \
            --eval_dataset "${data}" \
            --eval_protocol "${eval_protocol}" \
            --seq_len "${seq_len}" \
            --start_frame "${start_frame}" \
            --pose_eval_stride "${pose_eval_stride}" \
            --max_views "${max_views}" \
            --size 512 \
            --model_variant "${model_name}"
    done
done
