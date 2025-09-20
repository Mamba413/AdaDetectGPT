# Copyright (c) Jin Zhu.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random

import numpy as np
import torch
from torch import nn
import tqdm
import argparse
import json
from data_builder import load_data
from model import load_tokenizer, load_model
from metrics import get_roc_metrics, get_precision_recall_metrics
from nuisance_func import BSplineTwoSample
from nuisance_func_human import BSplineTheory
from utils import load_training_data, separated_string
import json
import time
from utils import GpuMem

def get_classification_stat(logits_ref, logits_score, labels, w_func, shift_value=None):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = w_func(torch.log_softmax(logits_score, dim=-1))
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)

    L = torch.tensor(var_ref.shape[0])
    stat = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1) - shift_value * L) / var_ref.sum(dim=-1).sqrt()
    stat = stat.mean()
    return stat.item()


def experiment(args):
    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()
    if args.sampling_model_name != args.scoring_model_name:
        sampling_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)
        sampling_model = load_model(args.sampling_model_name, args.device, args.cache_dir)
        sampling_model.eval()
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])
    # evaluate criterion
    name = "AdaDetectGPT"
    criterion_fn = get_classification_stat

    # w function
    start = time.perf_counter()
    tracker = GpuMem()
    if args.w_func == 'identity':
        w_func = nn.Identity()
        beta = None
    else:
        bspline_args = args.config
        ## load training data
        print(f"Datasets for learning BSpline: {args.train_dataset}")
        with tracker:
            train_data = load_training_data(args.train_dataset)
            if args.num_subsample > 0:
                args.num_subsample = min(args.num_subsample, len(train_data['original']))
                train_data['original'] = train_data['original'][:args.num_subsample]
                train_data['sampled'] = train_data['sampled'][:args.num_subsample]
            human_token_list = [scoring_tokenizer(x, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device) for x in train_data['original']]
            if args.w_func == 'bspline':
                machine_token_list = [scoring_tokenizer(x, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device) for x in train_data['sampled']]
            
            if args.w_func == 'bspline':
                w_func = BSplineTwoSample(bspline_args, args.device)
                w_func.fit(human_token_list, machine_token_list, scoring_model, args)
            elif args.w_func == 'bspline_theory':
                w_func = BSplineTheory(bspline_args, machine_text=False)
                w_func.fit(human_token_list, None, scoring_model, args)
        beta = w_func.beta_hat.detach().cpu().tolist()
    pre_time = time.perf_counter() - start
    pre_memory = tracker.memory_usage()
    

    shift_value = torch.zeros(1).to(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    results = []
    eval_time_list = []
    eval_memory_list = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        # original text
        start = time.perf_counter()
        with tracker:
            tokenized = scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
            labels = tokenized.input_ids[:, 1:]
            with torch.no_grad():
                logits_score = scoring_model(**tokenized).logits[:, :-1]
                if args.sampling_model_name == args.scoring_model_name:
                    logits_ref = logits_score
                else:
                    tokenized = sampling_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                    assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                    logits_ref = sampling_model(**tokenized).logits[:, :-1]
                original_crit = criterion_fn(logits_ref, logits_score, labels, w_func, shift_value)
        eval_time_list.append(time.perf_counter() - start)
        eval_memory_list.append(tracker.memory_usage())
        # sampled text
        tokenized = scoring_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            if args.sampling_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = sampling_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = sampling_model(**tokenized).logits[:, :-1]
            sampled_crit = criterion_fn(logits_ref, logits_score, labels, w_func, shift_value)
        # result
        results.append({"original": original_text,
                        "original_crit": original_crit,
                        "sampled": sampled_text,
                        "sampled_crit": sampled_crit})
    eval_time = np.mean(np.array([eval_time_list]))
    eval_memory = np.mean(np.array([eval_memory_list]))
    
    # results
    results_file = f'{args.output_file}.{name}.{args.w_func}.json'
    # compute prediction scores for real/sampled passages
    predictions = {'real': [x["original_crit"] for x in results],
                'samples': [x["sampled_crit"] for x in results]}
    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    results = { 'name': f'{name}_threshold',
                'info': {'n_samples': n_samples},
                'predictions': predictions,
                'raw_results': results,
                'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
                'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
                'loss': 1 - pr_auc, 
                'beta': beta, 
                'bias': shift_value.detach().cpu().tolist(),
                'compute_info': {'pre_time': pre_time, 'eval_time': eval_time, 
                                    'pre_memory': pre_memory, 'eval_memory': eval_memory,}
                    }
        
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_main/results/squad_gemma-9b")  # output file prefix
    parser.add_argument('--dataset', type=str, default="squad")
    parser.add_argument('--dataset_file', type=str, default="./exp_main/data/squad_gemma-9b")
    parser.add_argument('--train_dataset', type=separated_string, default="./exp_main/data/xsum_gemma-9b&./exp_main/data/writing_gemma-9b")
    parser.add_argument('--num_subsample', type=int, default=-1, help="number of samples to use for training w function, -1 means all samples")
    parser.add_argument('--sampling_model_name', type=str, default="gemma-9b")
    parser.add_argument('--scoring_model_name', type=str, default="gemma-9b-instruct")
    parser.add_argument('--w_func', type=str, default='bspline', choices=['identity', 'bspline', 'bspline_theory'])
    parser.add_argument("--config", type=json.loads, default='{"start": -32, "end": 0, "n_bases": 7, "spline_order": 2, "intercept": 1}', help='A JSON dict')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    experiment(args)
