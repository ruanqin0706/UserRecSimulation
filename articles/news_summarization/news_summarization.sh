#!/bin/bash

pip install transformers
pip install sentencepiece

mkdir -p ${1}
while read p; do

  nohup python news_summarization_generator.py /root/hp_bypublisher_article_filtering_folder/${p} ${1}/${p} "/root/google_pegasus-cnn_dailymail" &

done <"${2}"
