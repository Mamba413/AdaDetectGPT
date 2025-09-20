#!/usr/bin/env bash
# Copyright (c) Jin Zhu.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
exp_path=exp_gpt3to4
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

datasets="xsum writing yelp essay"
source_models="gemini-2.5-flash claude-3-5-haiku gpt-4o"

# preparing dataset
openai_base="https://api.openai.com/v1"
openai_key=""  # replace with your own key for generating your own test set

####################### Data generation #######################
# # We use a temperature of 0.8 for creativity writing
# for M in gpt-4o; do
#   for D in $datasets; do
#     echo `date`, Preparing dataset ${D} by sampling from openai/${M} ...
#     python scripts/data_builder.py --openai_model $M --openai_key $openai_key --openai_base $openai_base \
#                 --dataset $D --n_samples 150 --do_temperature --temperature 0.8 --batch_size 1 \
#                 --output_file $data_path/${D}_${M}
#   done
# done

# for M in gemini-2.5-flash; do
#   for D in $datasets; do
#     echo `date`, Preparing dataset ${D} by sampling from openai/${M} ...
#     python scripts/data_builder.py --gemini_model $M --dataset $D --n_samples 150 --do_temperature --temperature 0.8 --batch_size 1 --output_file $data_path/${D}_${M}
#   done
# done

# for M in claude-3-5-haiku; do
#   for D in $datasets; do
#     echo `date`, Preparing dataset ${D} by sampling from openai/${M} ...
#     python scripts/data_builder.py --claude_model $M --dataset $D --n_samples 150 --do_temperature --temperature 0.8 --batch_size 1 --output_file $data_path/${D}_${M}
#   done
# done
#################################################################

# Evaluate AdaDetectGPT
settings='gemma-9b:gemma-9b-instruct'
model_lists=('gpt-4o' 'gemini-2.5-flash' 'claude-3-5-haiku')
for M in $source_models; do
  model_lists=()
  for m in $source_models; do
    if [[ ${m} != ${M} ]]; then
      model_lists+=("$m")
    fi
  done
  for D in $datasets; do
    train_lists=()
    for d in $datasets; do
      if [[ ${d} != ${D} ]]; then
        train_lists+=("$d")
      fi
    done
    train_dataset="${data_path}/${train_lists[0]}_${M}&${data_path}/${train_lists[1]}_${M}&${data_path}/${train_lists[2]}_${M}"
    for S in $settings; do
      IFS=':' read -r -a S <<< $S && M1=${S[0]} && M2=${S[1]}
      echo `date`, Evaluating AdaDetectGPT on ${D}_${M}.${M1}_${M2} ...
      python scripts/detect_gpt_ada.py --sampling_model_name $M1 --scoring_model_name $M2 --dataset $D --train_dataset "$train_dataset" --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}_${M2} --w_func='bspline'
    done
  
    echo `date`, Evaluating Fast-DetectGPT on ${D}_${M}.gemma-9b:gemma-9b-instruct ...
    python scripts/detect_gpt_fast.py --sampling_model_name gemma-9b --scoring_model_name gemma-9b-instruct --discrepancy_analytic --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.gemma-9b_gemma-9b-instruct

    echo `date`, Evaluating Binoculars on ${D}_${M}.gemma-9b:gemma-9b-instruct ...
    python scripts/detect_bino.py --model1_name gemma-9b --model2_name gemma-9b-instruct --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.gemma-9b_gemma-9b-instruct
  done
done

# evaluate supervised detectors
supervised_models="roberta-base-openai-detector roberta-large-openai-detector"
for M in $source_models; do
  for D in $datasets; do
    for SM in $supervised_models; do
      echo `date`, Evaluating ${SM} on ${D}_${M} ...
      python scripts/detect_roberta.py --model_name $SM --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
    done
  done
done

# evaluate baselines
scoring_models="gemma-9b"
for M in $source_models; do
  for D in $datasets; do
    for M2 in $scoring_models; do
      echo `date`, Evaluating baseline methods on ${D}_${M}.${M2} ...
      python scripts/detect_gltr.py --scoring_model_name ${M2} --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M2}
    done
  done
done

# evaluate RADAR
for D in $datasets; do
  for M in $source_models; do
    echo `date`, Evaluating RADAR on ${D}_${M} ...
    python scripts/detect_radar.py --dataset $data_path/${D}_${M} --output_file $res_path/${D}_${M}
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
    train_dataset="${train_parts[0]}_${M}.raw_data.json&${train_parts[1]}_${M}.raw_data.json&${train_parts[2]}_${M}.raw_data.json"
    python scripts/detect_biscope.py --train_dataset $train_dataset --test_dataset ${D}_${M}.raw_data.json --output_file $res_path/${D}_${M} --base_dir="./exp_gpt3to4/data"
  done
done


# # Evaluate FastDetectGPT
# echo `date`, Evaluating Fast-DetectGPT on ${D}_${M}.falcon-7b_falcon-7b-instruct ...
# settings="falcon-7b:falcon-7b-instruct"
# for M in $source_models; do
#   for D in $datasets; do
#     for S in $settings; do
#       IFS=':' read -r -a S <<< $S && M1=${S[0]} && M2=${S[1]}
#       echo `date`, Evaluating Fast-DetectGPT on ${D}_${M}.${M1}_${M2} ...
#       python scripts/detect_gpt_fast.py --sampling_model_name $M1 --scoring_model_name $M2 --discrepancy_analytic --dataset $D --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}_${M2}
#     done
#   done
# done
