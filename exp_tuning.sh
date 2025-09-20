#!/usr/bin/env bash
# Copyright (c) Jin Zhu.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
exp_path=exp_tuning
data_path=exp_main/data
res_path=$exp_path/results
mkdir -p $exp_path $res_path

datasets="xsum squad writing"
source_models="gpt2-xl opt-2.7b gpt-neo-2.7B"

echo `date`, Evaluate tuning parameter n_base:
n_bases=( 4 8 16 32 64 )
for N in "${n_bases[@]}"; do
  for D in $datasets; do
    # build train_dataset as the other two datasets joined by '&'
    train_parts=()
    for d in $datasets; do
      if [[ ${d} != ${D} ]]; then
        train_parts+=("$d")
      fi
    done

    for M in $source_models; do
      train_dataset="${data_path}/${train_parts[0]}_${M}&${data_path}/${train_parts[1]}_${M}"
      echo `date`, Evaluating StatsDetectGPT on ${D}_${M} with "$N" bases ...

      python scripts/detect_gpt_ada.py --sampling_model_name $M --scoring_model_name $M --dataset $D --train_dataset "$train_dataset" --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}_base"$N" --config "{\"start\": -32, \"end\": 0, \"n_bases\": $N, \"spline_order\": 2}"
    done
  done
done

echo `date`, Evaluate tuning parameter spline_order:
spline_orders=( 1 2 3 4 )
for N in "${spline_orders[@]}"; do
  for D in $datasets; do
    # build train_dataset as the other two datasets joined by '&'
    train_parts=()
    for d in $datasets; do
      if [[ ${d} != ${D} ]]; then
        train_parts+=("$d")
      fi
    done

    for M in $source_models; do
      train_dataset="${data_path}/${train_parts[0]}_${M}&${data_path}/${train_parts[1]}_${M}"
      echo `date`, Evaluating StatsDetectGPT on ${D}_${M} with "$N" bases ...

      python scripts/detect_gpt_ada.py --sampling_model_name $M --scoring_model_name $M --dataset $D --train_dataset "$train_dataset" --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}_order"$N" --config "{\"start\": -32, \"end\": 0, \"n_bases\": 16, \"spline_order\": $N}"
    done
  done
done
