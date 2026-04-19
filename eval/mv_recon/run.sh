#!/bin/bash

set -euo pipefail

workdir='.'
default_model_names="cut ttt ttsa cut_memix ttt_memix ttsa_memix"
read -r -a model_names <<< "${MODEL_NAMES:-$default_model_names}"
read -r -a datasets <<< "${DATASETS:-7scenes NRGBD}"
read -r -a max_views_list <<< "${MAX_VIEWS:-300}"
ckpt_name="${CKPT_NAME:-cut3r_512_dpt_4_64}"
model_weights="${WEIGHTS:-${workdir}/src/${ckpt_name}.pth}"
port="${PORT:-29502}"
kf_every="${KF_EVERY:-2}"
dist_timeout="${DIST_TIMEOUT:-${MOM3R_DIST_TIMEOUT:-${TORCH_DISTRIBUTED_DEFAULT_TIMEOUT:-${NCCL_TIMEOUT:-7200}}}}"

if [[ -n "${NUM_PROCESSES:-}" ]]; then
    num_processes="${NUM_PROCESSES}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
    num_processes="${#visible_devices[@]}"
else
    num_processes=1
fi

for model_name in "${model_names[@]}"; do
for data in "${datasets[@]}"; do
for max_views in "${max_views_list[@]}"; do
    output_dir="${workdir}/eval_results/mv_recon/${data}_${max_views}/${model_name}"
    echo "$output_dir"
    MOM3R_DIST_TIMEOUT="${dist_timeout}" TORCH_DISTRIBUTED_DEFAULT_TIMEOUT="${dist_timeout}" NCCL_TIMEOUT="${dist_timeout}" \
    accelerate launch --num_processes "${num_processes}" --main_process_port "${port}" eval/mv_recon/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --model_name "cut3r" \
        --model_variant "$model_name" \
        --dataset "$data" \
        --max_views "$max_views" \
        --kf_every "$kf_every"
done
done
done
