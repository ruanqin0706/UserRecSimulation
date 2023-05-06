#!/bin/bash
# Environments:
#PyTorch / 1.9.0 / 3.8(ubuntu18.04) / 11.1 ｜ 3090，24gb

original_news_file_path="/root/data/news.tsv"
original_user_file_path="/root/data/user_groups.tsv"
nid2bias_path="/root/data/nid2bias_prob.pkl"
simulation_dir="/root/simulation/"
model_training_dir="/root/News-Recommendation/src"
data_root="/root"
model_path="/root/fim/379848.model"
rm -rf "${1}"
mkdir -p "${1}"

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
  rm -rf "${model_training_dir}/data"
  rm -rf "${data_root}/MIND/MINDlarge_test/behaviors.tsv"
  rm -rf "${data_root}/MIND/MINDlarge_test/news.tsv"
  cp "${1}/behaviors_pseudo${p}.tsv" "${data_root}/MIND/MINDlarge_test/behaviors.tsv"
  cp "${1}/news_pseudo${p}.tsv" "${data_root}/MIND/MINDlarge_test/news.tsv"

  python -m main.fim \
    --scale 'large' \
    --data-root "${data_root}" \
    --infer-dir "${1}" \
    --suffix "behaviors_pseudo${p}" \
    --checkpoint "${model_path}" \
    --batch-size 16 \
    --world-size 1 \
    --device "${4}" \
    --mode "test" \
    --batch-size-eval 16

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
