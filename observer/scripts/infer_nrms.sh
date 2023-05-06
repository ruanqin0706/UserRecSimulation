#!/bin/bash

simulation_dir="/root/simulation/"
model_training_dir="/root/ms_recommender_original"
original_news_file_path="/root/data/news.tsv"
original_user_file_path="/root/data/user_groups.tsv"
nid2bias_path="/root/data/nid2bias_prob.pkl"
model_path="/root/autodl-tmp/nrms_res/ckpt_3"

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
  python inference_nrms.py \
    --data_dir "${1}/train_data" \
    --epochs 30 \
    --device_no "${4}" \
    --infer_model_path "${model_path}" \
    --save_dir "${1}" \
    --news_file "${1}/news_pseudo${p}.tsv" \
    --gen_news \
    --behaviors_file "${1}/behaviors_pseudo${p}.tsv"

  cd "${simulation_dir}"
  if [[ ${3} == 'fixed' ]]; then
    echo "fixed prediction"
    python simulate_prediction_fixed.py \
      --news_file "${1}/news_pseudo${p}.tsv" \
      --infer_news_file "${1}/behaviors_pseudo${p}_news.pkl" \
      --infer_user_file "${1}/behaviors_pseudo${p}_user.pkl" \
      --top_k "${5}" \
      --pseudo_step "${p}" \
      --write_dir "${1}"
  else
    echo "dynamic prediction"
    python simulate_prediction_dynamic.py \
      --news_file "${1}/news_pseudo${p}.tsv" \
      --infer_news_file "${1}/behaviors_pseudo${p}_news.pkl" \
      --infer_user_file "${1}/behaviors_pseudo${p}_user.pkl" \
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
