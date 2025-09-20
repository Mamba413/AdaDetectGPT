import random

import numpy as np
import torch
from torch import nn
import tqdm
import argparse
import json
from data_builder import load_data
from model import load_tokenizer, load_model
from nuisance_func import BSplineTwoSample
from utils import load_training_data, separated_string

def sample_stat_value(logits_ref, logits_score, labels, w_func):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    lprobs_score = w_func(torch.log_softmax(logits_score, dim=-1))
    probs_ref = torch.softmax(logits_ref, dim=-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    stat = var_ref.flatten()
    stat = torch.cumsum(stat, dim=0) / torch.arange(1, len(stat)+1, dtype=torch.float32, device=stat.device)
    return stat

def human_stat_value(logits_ref, logits_score, labels, w_func):
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

    mean_ref = mean_ref.flatten()
    log_likelihood = log_likelihood.flatten()
    stat = torch.zeros(log_likelihood.shape[0], device=log_likelihood.device)
    for j in range(1, log_likelihood.shape[0] + 1):
        term1 = torch.var(log_likelihood[:j], unbiased=False)
        term2 = torch.var(mean_ref[:j], unbiased=False)
        stat[j-1] = term1 - term2
    stat[0] = torch.zeros(1, device=log_likelihood.device)

    stat = torch.cumsum(stat, dim=0) / torch.arange(1, len(stat)+1, dtype=torch.float32, device=stat.device)
    return stat

def compute_sample_variance(cumsum_stat_list, L):
    sample_var_list = torch.zeros(L-1)
    for l in range(L-1):
        sample_var = torch.var(torch.tensor([cumsum_stat[l] for cumsum_stat in cumsum_stat_list if len(cumsum_stat) >= l+1]), unbiased=False)
        sample_var_list[l] = sample_var
    sample_var_list = torch.cumsum(sample_var_list, dim=0) / torch.arange(1, L, dtype=torch.float32)
    return sample_var_list

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

    sample_criterion_fn = sample_stat_value
    human_criterion_fn = human_stat_value
    
    if args.w_func == 'identity':
        w_func = nn.Identity()
    else:
        bspline_args = args.config
        ## load training data
        print(f"Datasets for learning BSpline: {args.train_dataset}")
        train_data = load_training_data(args.train_dataset)
        human_token_list = [scoring_tokenizer(x, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device) for x in train_data['original']]
        if args.w_func == 'bspline' or args.w_func == 'bspline_theory_constrained':
            machine_token_list = [scoring_tokenizer(x, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device) for x in train_data['sampled']]
        
        if args.w_func == 'bspline':
            w_func = BSplineTwoSample(bspline_args, args.device)
            w_func.fit(human_token_list, machine_token_list, scoring_model, args)
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    results = []
    original_crit_list = []
    sampled_crit_list = []

    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing sample variance"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        if args.compute_text == 'human':
            computed_text = original_text
        else:
            computed_text = sampled_text
        # when next token is drawn from human
        tokenized = scoring_tokenizer(computed_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            if args.sampling_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = sampling_tokenizer(computed_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = sampling_model(**tokenized).logits[:, :-1]
            original_crit = human_criterion_fn(logits_ref, logits_score, labels, w_func)
        original_crit_list.append(original_crit)
        
        # when next token is drawn from machine (because the ratio involve the same random tokens as conditions, we set the text as the human one)
        tokenized = scoring_tokenizer(computed_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            if args.sampling_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = sampling_tokenizer(computed_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = sampling_model(**tokenized).logits[:, :-1]
            sampled_crit = sample_criterion_fn(logits_ref, logits_score, labels, w_func)
        sampled_crit_list.append(sampled_crit)

    results_file = f'{args.output_file}.{args.w_func}.json'
    ratio_list = []
    for idx in range(n_samples):
        L = min(len(original_crit_list[idx]), len(sampled_crit_list[idx]))
        ratio = original_crit_list[idx][:L] / sampled_crit_list[idx][:L]
        ratio_list.append(ratio)

    L = int(torch.quantile(torch.tensor([len(ratio)*1.0 for ratio in ratio_list]), q=0.81).item())

    ratio_var_list = torch.zeros(L-1)
    ratio_mean_list = torch.zeros(L-1)
    for l in range(L-1):
        ratio_l = torch.tensor([ratio[l] for ratio in ratio_list if len(ratio) >= l+1])
        ratio_var_list[l] = torch.var(ratio_l, unbiased=False)
        ratio_mean_list[l] = torch.mean(ratio_l)

    ratio_mean_list = ratio_mean_list[1:]
    ratio_var_list = ratio_var_list[1:]
    results = {
        'ratio_var': ratio_var_list.tolist(),
        'ratio_mean': ratio_mean_list.tolist(),
        'ratio_var/mean': (ratio_var_list / ratio_mean_list).tolist(),
    }

    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_variance/results/xsum_gpt2-xl")  # output file prefix
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="./exp_variance/data/xsum_gpt2-xl")
    parser.add_argument('--train_dataset', type=separated_string, default=[])
    parser.add_argument('--sampling_model_name', type=str, default="gpt2-xl")
    parser.add_argument('--scoring_model_name', type=str, default="gpt2-xl")
    parser.add_argument('--w_func', type=str, default='identity', choices=['identity', 'absoluate', 'bspline'])
    parser.add_argument("--config", type=json.loads, default='{"start": -32, "end": 0, "n_bases": 7, "spline_order": 2, "intercept": 1}', help='A JSON dict')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--compute_text', type=str, default='llm', choices=['human', 'llm'])
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    experiment(args)
