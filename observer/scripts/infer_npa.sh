#!/bin/bash

model_path="/root/data/ckpt_4"
simulation_dir="/root/simulation/"
model_training_dir="/root/ms_recommender_original"
original_news_file_path="/root/data/news.tsv"
original_user_file_path="/root/data/user_groups.tsv"
nid2bias_path="/root/data/nid2bias_prob.pkl"

rm -rf "${1}"
mkdir -p "${1}"
cp -r "/root/data/train_data" "${1}/"

while read p; do
  cd "${simulation_dir}"
  python simulate_behaviors.py \
    --original_behaviors_path "${original_user_file_path}" \
    --original_news_path "${original_news_file_path}" \
    --nid2bias_path "${nid2bias_path}" \
    --write_dir "${1}" \
    --pseudo_step "${p}" \
    --candidate_news_mode "${2}" \
    --news_candidate_strategy "${3}"

  cd "${model_training_dir}"
  python -u inference_npa.py \
    --data_dir "${1}/hp_dataset" \
    --epochs 5 \
    --device_no "${4}" \
    --save_dir "${1}" \
    --infer_model_path "${model_path}" \
    --news_file "${1}/news_pseudo${p}.tsv" \
    --behaviors_file "${1}/behaviors_pseudo${p}.tsv"

  cd "${simulation_dir}"
  if [[ ${3} == 'fixed' ]]; then
    echo "fixed prediction"
    python simulate_prediction_single_fixed.py \
      --top_k "${5}" \
      --pseudo_step "${p}" \
      --write_dir "${1}"
  else
    echo "dynamic prediction"
    python simulate_prediction_single_dynamic.py \
      --top_k "${5}" \
      --pseudo_step "${p}" \
      --write_dir "${1}"
  fi

  cd "${simulation_dir}"
  python simulate_feedback.py \
    --original_behaviors_path "${original_user_file_path}" \
    --pseudo_step "${p}" \
    --write_dir "${1}" \
    --nid2bias_path "${nid2bias_path}" \
    --original_behaviors_path "${original_user_file_path}" \
    --selected_n "${6}" \
    --strategy "${7}"

done <"${8}"
