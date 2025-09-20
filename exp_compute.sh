#!/usr/bin/env bash
# Copyright (c) Jin Zhu.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
exp_path=exp_compute
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path
datasets="squad"
source_models="gpt2-xl"

## preparing dataset
for D in $datasets; do
  for M in $source_models; do
    echo `date`, Preparing dataset ${D}_${M} ...
    python scripts/data_builder.py --dataset $D --n_samples 500 --base_model_name $M --output_file $data_path/${D}_${M}
  done
done

echo `date`, Evaluate the computational performance of ImBD and AdaDetectGPT:

# AdaDetectGPT and ImBD: scaling as n
n_train=( 350 300 250 200 150 100 )
for N in "${n_train[@]}"; do
  for D in $datasets; do
    for M in $source_models; do
      train_dataset="${data_path}/xsum_${M}"
      echo `date`, Evaluating StatsDetectGPT on ${D}_${M} with "$N" bases ...

      python scripts/detect_ImBD.py --base_model $M --train_dataset "$train_dataset" --eval_dataset $data_path/${D}_${M} --output_file $res_path/${D}_${M}_sample"$N" --datanum="$N" --eval_after_train

      python scripts/detect_gpt_ada.py --sampling_model_name $M --scoring_model_name $M --dataset $D --train_dataset "$train_dataset" --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}_sample"$N" --num_subsample="$N"
    done
  done
done

# AdaDetectGPT: scaling as d
n_bases=( 19 15 11 7 3 )
for N in "${n_bases[@]}"; do
  for D in $datasets; do
    for M in $source_models; do
      train_dataset="${data_path}/xsum_${M}"
      echo `date`, Evaluating StatsDetectGPT on ${D}_${M} with "$N" bases ...

      python scripts/detect_gpt_ada.py --sampling_model_name $M --scoring_model_name $M --dataset $D --train_dataset "$train_dataset" --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}_base"$N" --config "{\"start\": -32, \"end\": 0, \"n_bases\": $N, \"spline_order\": 2}"
    done
  done
done

