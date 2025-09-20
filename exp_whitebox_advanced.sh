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

datasets="xsum squad writing yelp essay"
source_models="qwen-7b mistralai-7b llama3-8b"

## preparing dataset
for D in $datasets; do
  for M in $source_models; do
    echo `date`, Preparing dataset ${D}_${M} ...
    python scripts/data_builder.py --dataset $D --n_samples 500 --base_model_name $M --output_file $data_path/${D}_${M}
  done
done

# White-box Setting
echo `date`, Evaluate models in the white-box setting:

# evaluate AdaDetectGPT, Fast-DetectGPT, Fast baselines
for D in $datasets; do
  # build train_dataset as the other two datasets joined by '&'
  train_parts=()
  for d in $datasets; do
    if [[ ${d} != ${D} ]]; then
      train_parts+=("$d")
    fi
  done

  for M in $source_models; do
    echo `date`, Evaluating StatsDetectGPT/FastDetectGPT on ${D}_${M} ...

    python scripts/detect_gpt_fast.py --sampling_model_name $M --scoring_model_name $M --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M} --discrepancy_analytic

    if [[ "$D" == "yale" || "$D" == "essay" ]]; then
      train_dataset="${data_path}/squad_${M}&${data_path}/xsum_${M}"
    else
      train_dataset="${data_path}/${train_parts[0]}_${M}&${data_path}/${train_parts[1]}_${M}"
    fi
    python scripts/detect_gpt_ada.py --sampling_model_name $M --scoring_model_name $M --dataset $D --train_dataset "$train_dataset" --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}

    echo `date`, Evaluating baseline methods on ${D}_${M} ...
    python scripts/detect_gltr.py --scoring_model_name $M --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
  done
done

# evaluate DetectGPT and its improvement DetectLLM
for D in $datasets; do
  for M in $source_models; do
    echo `date`, Evaluating DetectGPT on ${D}_${M} ...
    python scripts/detect_gpt.py --scoring_model_name $M --mask_filling_model_name t5-3b --n_perturbations 100 --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
    echo `date`, Evaluating DetectLLM methods on ${D}_${M} ...
    python scripts/detect_llm.py --scoring_model_name $M --dataset $D --dataset_file $data_path/${D}_${M}.t5-3b.perturbation_100 --output_file $res_path/${D}_${M}
  done
done


# evaluate Binoculars
for D in $datasets; do
  for M in $source_models; do
    echo `date`, Evaluating DNA-GPT on ${D}_${M} ...
    python scripts/detect_bino.py --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
  done
done

# evaluate DNA-GPT
for D in $datasets; do
  for M in $source_models; do
    echo `date`, Evaluating DNA-GPT on ${D}_${M} ...
    python scripts/detect_gpt_dna.py --base_model_name $M --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
  done
done

# evaluate Text-Fluoroscopy
for D in $datasets; do
  train_parts=()
  for d in $datasets; do
    if [[ ${d} != ${D} ]]; then
      train_parts+=("$d")
    fi
  done
  for M in $source_models; do
    echo `date`, Evaluating Text-Fluoroscopy on ${D}_${M} ...
    train_dataset="${data_path}/${train_parts[0]}_${M}"
    valid_dataset="${data_path}/${train_parts[1]}_${M}"
    python scripts/detect_fluoroscopy.py \
      --train_dataset $train_dataset --valid_dataset $valid_dataset \
      --test_dataset $data_path/${D}_${M} \
      --output_file=$res_path/${D}_${M}
  done
done

# evaluate RADAR
for D in $datasets; do
  for M in $source_models; do
    echo `date`, Evaluating RADAR on ${D}_${M} ...
    python scripts/detect_radar.py --dataset $data_path/${D}_${M} --output_file $res_path/${D}_${M}
  done
done

# evaluate ImBD
for D in $datasets; do
  train_parts=()
  for d in $datasets; do
    if [[ ${d} != ${D} ]]; then
      train_parts+=("$d")
    fi
  done
  for M in $source_models; do
    echo `date`, Evaluating ImBD on ${D}_${M} ...
    train_dataset="${data_path}/${train_parts[0]}_${M}&${data_path}/${train_parts[1]}_${M}"
    python scripts/detect_ImBD.py \
      --train_dataset $train_dataset --base_model $M \
      --eval_dataset=$data_path/${D}_${M} \
      --eval_after_train \
      --output_file=$res_path/${D}_${M}
  done
done

# evaluate BiScope
for D in $datasets; do
  train_parts=()
  for d in $datasets; do
    if [[ ${d} != ${D} ]]; then
      train_parts+=("$d")
    fi
  done
  for M in $source_models; do
    echo `date`, Evaluating BiScope on ${D}_${M} ...
    train_dataset="${train_parts[0]}_${M}.raw_data.json&${train_parts[1]}_${M}.raw_data.json"
    python scripts/detect_biscope.py --train_dataset $train_dataset --test_dataset ${D}_${M}.raw_data.json --output_file $res_path/${D}_${M}
  done
done