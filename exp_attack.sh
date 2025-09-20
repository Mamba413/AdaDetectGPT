#!/usr/bin/env bash
# Copyright (c) Jin Zhu.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
para=random  # "t5" for paraphrasing attack, or "random" for decoherence attack
exp_path=exp_attack
data_path=$exp_path/data/$para
res_path=$exp_path/results/$para
mkdir -p $exp_path $exp_path/data/ $exp_path/results/ $data_path $res_path

src_path=exp_gpt3to4
src_data_path=$src_path/data

datasets="xsum writing pubmed"
source_models="gpt-3.5-turbo"

# preparing dataset
for D in $datasets; do
  for M in $source_models; do
    echo `date`, Preparing dataset ${D}_${M} by paraphrasing  ${src_data_path}/${D}_${M} ...
    python scripts/paraphrasing.py --dataset $D --dataset_file $src_data_path/${D}_${M} --paraphraser $para --output_file $data_path/${D}_${M}
  done
done

# evaluate FastDetectGPT and AdaDetectGPT in the black-box setting
settings="gpt-j-6B:gpt2-xl gpt-j-6B:gpt-neo-2.7B gpt-j-6B:gpt-j-6B"
for M in $source_models; do
  for D in $datasets; do
    # build train_dataset as the other two datasets joined by '&'
    train_lists=()
    for d in $datasets; do
      if [[ ${d} != ${D} ]]; then
        train_lists+=("$d")
      fi
    done
    train_dataset="${data_path}/${train_lists[0]}_gpt-3.5-turbo&${data_path}/${train_lists[1]}_gpt-3.5-turbo"
    for S in $settings; do
      IFS=':' read -r -a S <<< $S && M1=${S[0]} && M2=${S[1]}
      echo `date`, Evaluating Fast-DetectGPT on ${D}_${M}.${M1}_${M2} ...
      python scripts/detect_gpt_fast.py --sampling_model_name $M1 --scoring_model_name $M2 --discrepancy_analytic --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}_${M2}
      
      echo `date`, Evaluating StatDetectGPT on ${D}_${M}.${M1}_${M2} ...
      python scripts/detect_gpt_ada.py --sampling_model_name $M1 --scoring_model_name $M2 --dataset $D --train_dataset "$train_dataset" --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}_${M2} --w_func='bspline'
    done
  done
done
