#!/usr/bin/env bash
# Copyright (c) Jin Zhu.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
exp_path=exp_sample
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

datasets="xsum squad writing"
source_models="gpt2-xl opt-2.7b gpt-neo-2.7B"
sample_sizes=( 100 200 300 400 500 600 )

# preparing dataset
for D in $datasets; do
  for M in $source_models; do
    for N in "${sample_sizes[@]}"; do
      echo `date`, Preparing dataset ${D}_${M}_"$N" ...
      python scripts/data_builder.py --dataset $D --n_samples "$N" --base_model_name $M --output_file $data_path/${D}_${M}_"$N" 
    done
  done
done

# evaluate Fast-DetectGPT and fast baselines
for N in "${sample_sizes[@]}"; do
  for D in $datasets; do
    # build train_dataset as the other two datasets joined by '&'
    train_parts=()
    for d in $datasets; do
      if [[ ${d} != ${D} ]]; then
        train_parts+=("$d")
      fi
    done

    for M in $source_models; do
      train_dataset="${data_path}/${train_parts[0]}_${M}_${N}&${data_path}/${train_parts[1]}_${M}_${N}"
      echo `date`, Evaluating AdaDetectGPT on ${D}_${M}_"$N" ...

      python scripts/detect_gpt_ada.py --sampling_model_name $M --scoring_model_name $M --dataset $D --train_dataset "$train_dataset" --dataset_file $data_path/${D}_${M}_"$N" --output_file $res_path/${D}_${M}_"$N"

      echo `date`, Evaluating FastDetectGPT on ${D}_${M}_"$N" ...
      python scripts/detect_gpt_fast.py --sampling_model_name $M --scoring_model_name $M --dataset $D --dataset_file $data_path/${D}_${M}_"$N" --output_file $res_path/${D}_${M}_"$N"

      echo `date`, Evaluating baseline methods on ${D}_${M}_"$N" ...
      python scripts/detect_gltr.py --scoring_model_name $M --dataset $D --dataset_file $data_path/${D}_${M}_"$N" --output_file $res_path/${D}_${M}_"$N"

      echo `date`, Evaluating DNA-GPT on ${D}_${M}_"$N" ...
      python scripts/detect_gpt_dna.py --base_model_name $M --dataset $D --dataset_file $data_path/${D}_${M}_"$N" --output_file $res_path/${D}_${M}_"$N"
    done
  done
done
