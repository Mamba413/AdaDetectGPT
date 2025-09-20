#!/usr/bin/env bash
# Copyright (c) Jin Zhu.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
exp_path=exp_main
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

datasets="xsum squad writing"
source_models="gpt2-xl opt-2.7b gpt-neo-2.7B gpt-j-6B gpt-neox-20b"

# Simple black-box Setting 
echo `date`, Evaluate models in the black-box setting:
for D in $datasets; do
  # build train_dataset as the other two datasets joined by '&'
  train_parts=()
  for d in $datasets; do
    if [[ ${d} != ${D} ]]; then
      train_parts+=("$d")
    fi
  done
  for M in $source_models; do
    M1=gpt-j-6B  # sampling model
    M2=gpt-neo-2.7B  # scoring model
    
    echo `date`, Evaluating FastDetectGPT on ${D}_${M}.${M1}_${M2} ...
    python scripts/detect_gpt_fast.py --sampling_model_name ${M1} --scoring_model_name ${M2} --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}_${M2} --discrepancy_analytic

    train_dataset="${data_path}/${train_parts[0]}_${M}&${data_path}/${train_parts[1]}_${M}"
    echo `date`, Evaluating AdaDetectGPT on ${D}_${M}.${M1}_${M2} ...
    python scripts/detect_gpt_ada.py --sampling_model_name ${M1} --scoring_model_name ${M2} --dataset $D --train_dataset "$train_dataset" --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}_${M2}
  done
done