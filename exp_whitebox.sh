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

# preparing dataset
for D in $datasets; do
  for M in $source_models; do
    echo `date`, Preparing dataset ${D}_${M} ...
    python scripts/data_builder.py --dataset $D --n_samples 500 --base_model_name $M --output_file $data_path/${D}_${M}
  done
done

# White-box Setting
echo `date`, Evaluate models in the white-box setting:

# evaluate AdaDetectGPT
for D in $datasets; do
  # build train_dataset as the other two datasets joined by '&'
  train_parts=()
  for d in $datasets; do
    if [[ ${d} != ${D} ]]; then
      train_parts+=("$d")
    fi
  done

  for M in $source_models; do
    echo `date`, Evaluating AdaDetectGPT on ${D}_${M} ...
    train_dataset="${data_path}/${train_parts[0]}_${M}&${data_path}/${train_parts[1]}_${M}"
    python scripts/detect_gpt_ada.py --sampling_model_name $M --scoring_model_name $M --dataset $D --train_dataset "$train_dataset" --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
  done
done


# evaluate Fast-DetectGPT and fast baselines
for D in $datasets; do
  for M in $source_models; do
    echo `date`, Evaluating Fast-DetectGPT on ${D}_${M} ...
    python scripts/detect_gpt_fast.py --sampling_model_name $M --scoring_model_name $M --dataset $D \
                          --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}

    echo `date`, Evaluating baseline methods on ${D}_${M} ...
    python scripts/detect_gltr.py --scoring_model_name $M --dataset $D \
                          --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
  done
done

# evaluate DNA-GPT
for D in $datasets; do
  for M in $source_models; do
    echo `date`, Evaluating DNA-GPT on ${D}_${M} ...
    python scripts/detect_gpt_dna.py --base_model_name $M --dataset $D \
                          --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
  done
done

# evaluate DetectGPT and its improvement DetectLLM
for D in $datasets; do
  for M in $source_models; do
    echo `date`, Evaluating DetectGPT on ${D}_${M} ...
    python scripts/detect_gpt.py --scoring_model_name $M --mask_filling_model_name t5-3b --n_perturbations 100 --dataset $D \
                          --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
     # we leverage DetectGPT to generate the perturbations
    echo `date`, Evaluating DetectLLM methods on ${D}_${M} ...
    python scripts/detect_llm.py --scoring_model_name $M --dataset $D \
                          --dataset_file $data_path/${D}_${M}.t5-3b.perturbation_100 --output_file $res_path/${D}_${M}
  done
done
