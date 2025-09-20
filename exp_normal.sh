#!/usr/bin/env bash
# Copyright (c) Jin Zhu.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
exp_path=exp_normal
datasets="squad writing xsum"
source_models="gpt2-xl gpt-neo-2.7B"

data_path=$exp_path/data_exact
res_path=$exp_path/results_exact
mkdir -p $exp_path $data_path $res_path
# preparing dataset
for D in $datasets; do
    for M in $source_models; do
        echo `date`, Preparing dataset ${D}_${M} ...
        python scripts/data_builder.py --dataset $D --n_samples 200 --base_model_name $M --output_file $data_path/${D}_${M} --max_length=500 --batch_size=5 --n_prompts=1 --do_exact_cond_prob
    done
done

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
        python scripts/detect_gpt_ada.py --sampling_model_name $M --scoring_model_name $M --dataset $D --dataset_file $data_path/${D}_${M} --train_dataset "$train_dataset" --output_file $res_path/${D}_${M}
    done
done

data_path=$exp_path/data_inexact
res_path=$exp_path/results_inexact
mkdir -p $exp_path $data_path $res_path
# preparing dataset
for D in $datasets; do
    for M in $source_models; do
        echo `date`, Preparing dataset ${D}_${M} ...
        python scripts/data_builder.py --dataset $D --n_samples 200 --base_model_name $M --output_file $data_path/${D}_${M} --max_length=200 --batch_size=5 --n_prompts=50
    done
done

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
        python scripts/detect_gpt_ada.py --sampling_model_name $M --scoring_model_name $M --dataset $D --dataset_file $data_path/${D}_${M} --train_dataset "$train_dataset" --output_file $res_path/${D}_${M}
    done
done
