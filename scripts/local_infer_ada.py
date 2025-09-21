# Copyright (c) Jin Zhu.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import torch
from torch import nn
import argparse
from model import load_tokenizer, load_model
from nuisance_func import BSplineTwoSample
from nuisance_func_human import BSplineTheory
from utils import load_training_data, separated_string
import json
from detect_gpt_ada import get_classification_stat

def load_model_and_tokenizer(args):
    DEVICE = 'cuda'

    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, DEVICE, args.cache_dir)
    scoring_model.eval()
    if args.sampling_model_name != args.scoring_model_name:
        sampling_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)
        sampling_model = load_model(args.sampling_model_name, DEVICE, args.cache_dir)
        sampling_model.eval()
    else:
        sampling_tokenizer = scoring_tokenizer
        sampling_model = scoring_model

    return scoring_tokenizer, scoring_model, sampling_tokenizer, sampling_model

def run(args, scoring_tokenizer, scoring_model, sampling_tokenizer, sampling_model):
    SEED = 2025
    DEVICE = 'cuda'
    # evaluate criterion
    criterion_fn = get_classification_stat
    # w function
    if args.w_func == 'identity':
        w_func = nn.Identity()
        beta = None
    else:
        bspline_args = args.config
        ## load training data
        print(f"Datasets for learning BSpline: {args.train_dataset}")

        if args.w_func == 'pretrained':
            w_func = BSplineTwoSample(bspline_args)
            w_func.beta_hat = torch.tensor([0.0, -0.011333, -0.037667, -0.056667, -0.281667, -0.592, 0.157833, 0.727333,]).to(DEVICE)
        else:
            if args.w_func == 'bspline':
                w_func = BSplineTwoSample(bspline_args)
            elif args.w_func == 'bspline_theory':
                w_func = BSplineTheory(bspline_args, machine_text=False)

            train_data = load_training_data(args.train_dataset)
            w_func.fit(train_data, scoring_tokenizer, scoring_model, args)
        beta = w_func.beta_hat.detach().cpu().tolist()

    shift_value = torch.zeros(1).to(DEVICE)
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    text = args.text
    tokenized = scoring_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(DEVICE)
    labels = tokenized.input_ids[:, 1:]
    if args.burn_in < 1.0:
        burn_in_num = int(args.burn_in * labels.size(-1))
    else:
        burn_in_num = int(args.burn_in)
    with torch.no_grad():
        logits_score = scoring_model(**tokenized).logits[:, :-1]
        if args.sampling_model_name == args.scoring_model_name:
            logits_ref = logits_score
        else:
            tokenized = sampling_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(DEVICE)
            assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
            logits_ref = sampling_model(**tokenized).logits[:, :-1]
        original_crit = criterion_fn(logits_ref, logits_score, labels, burn_in_num, w_func, shift_value)

    results = {
        "text": text,
        "crit": original_crit, 
        'beta': beta, 
    }
    return results['crit']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, help="Your text goes here. It can be a long text, and the longer the text, the more reliable the result will be.")
    parser.add_argument('--train_dataset', type=separated_string, default="./exp_main/data/xsum_gpt2-xl&./exp_main/data/writing_gpt2-xl")
    parser.add_argument('--sampling_mode_name', type=str, default="gemma-9b-instruct")
    parser.add_argument('--scoring_model_name', type=str, default="gemma-9b")
    parser.add_argument('--burn_in', type=float, default=0.0)
    parser.add_argument('--w_func', type=str, default='bspline', choices=['identity', 'pretrained', 'bspline', 'bspline_theory'])
    parser.add_argument("--config", type=json.loads, default='{"start": -32, "end": 0, "n_bases": 7, "spline_order": 2, "intercept": 1}', help='A JSON dict')
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    scoring_tokenizer, scoring_model, sampling_tokenizer, sampling_model = load_model_and_tokenizer(args)
    run(args, scoring_tokenizer, scoring_model, sampling_tokenizer, sampling_model)
