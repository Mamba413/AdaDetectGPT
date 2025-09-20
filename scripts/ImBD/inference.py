import numpy as np
import torch
import os
import glob
import argparse
import json
from .spo import get_sampling_discrepancy_analytic
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import time

# estimate the probability according to the distribution of our test results on GPT-3.5 and GPT-4
class ProbEstimator:
    def __init__(self, args):
        self.tasks = ["polish", "generate", "rewrite", "expand"] if args.task == "all" else [args.task]
        self.real_crits = {"polish":[], "generate":[], "rewrite":[], "expand":[]}
        self.fake_crits = {"polish":[], "generate":[], "rewrite":[], "expand":[]}
        for task in self.tasks:
            for result_file in glob.glob(os.path.join(args.ref_path, task, '*.json')):
                with open(result_file, 'r') as fin:
                    res = json.load(fin)
                    self.real_crits[task].extend(res['crit_real'])
                    self.fake_crits[task].extend(res['crit_sampled'])
        print(f'ProbEstimator: total {sum([len(self.real_crits[task]) for task in self.tasks]) * 2} samples.')

    def crit_to_prob(self, crit):
        real_crits = []
        fake_crits = []
        for task in self.tasks:
            real_crits.extend(self.real_crits[task])
            fake_crits.extend(self.fake_crits[task])
        offset = np.sort(np.abs(np.array(real_crits + fake_crits) - crit))[100]
        cnt_real = np.sum((np.array(real_crits) > crit - offset) & (np.array(real_crits) < crit + offset))
        cnt_fake = np.sum((np.array(fake_crits) > crit - offset) & (np.array(fake_crits) < crit + offset))
        return cnt_fake / (cnt_real + cnt_fake)

    def crit_to_prob_detail(self, crit):
        probs = []
        for task in self.tasks:
            real_crits = self.real_crits[task]
            fake_crits = self.fake_crits[task]
            offset = np.sort(np.abs(np.array(real_crits + fake_crits) - crit))[100]
            cnt_real = np.sum((np.array(real_crits) > crit - offset) & (np.array(real_crits) < crit + offset))
            cnt_fake = np.sum((np.array(fake_crits) > crit - offset) & (np.array(fake_crits) < crit + offset))
            probs.append(cnt_fake / (cnt_real + cnt_fake))
        return probs

# run interactive local inference
def run(args):
    
    # Generate the inference checkpoint from trained model
    # from spo import ComputeScore
    # model = ComputeScore("gpt-neo-2.7B", "gpt-neo-2.7B", cache_dir=args.cache_dir)
    # model.from_pretrained("ckpt/ai_detection_polish_500_spo_lr_0.0001_beta_0.05_a_1")
    # scoring_model = model.scoring_model
    # scoring_tokenizer = model.scoring_tokenizer
    # scoring_model.save_pretrained('models/ImBD-inference')
    # scoring_tokenizer.save_pretrained('models/ImBD-inference')
    
    print('Loading model')
    start_time = time.time()
    scoring_model = AutoPeftModelForCausalLM.from_pretrained('models/ImBD-inference') # Make sure you have downloaded the gpt-neo-2.7b and place it at `models` folder.
    scoring_tokenizer = AutoTokenizer.from_pretrained('models/ImBD-inference')
    scoring_tokenizer.pad_token = scoring_tokenizer.eos_token
    scoring_model.to(args.device)
    scoring_model.eval()
    print(f'Done. ({time.time()-start_time:.2f}s)')
    
    criterion_fn = get_sampling_discrepancy_analytic
    prob_estimator = ProbEstimator(args)
    
    # input text
    print('Local demo for ImBD, where the longer text has more reliable result.')
    print('To view detail results for all tasks, set `--detail` to True.')
    print('To view all-in-one results, set `--task` to `all`. (Not accurate enough)')
    print('')
    while True:
        print("Please enter your text: (Press Enter twice to start processing)")
        lines = []
        while True:
            line = input()
            if len(line) == 0:
                break
            lines.append(line)
        text = "\n".join(lines)
        if len(text) == 0:
            break
        # evaluate text
        tokenized = scoring_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            logits_ref = logits_score
            crit, _ = criterion_fn(logits_ref, logits_score, labels)
        # estimate the probability of machine generated text
        crit = crit.cpu().numpy().item()
        if args.detail:
            probs = prob_estimator.crit_to_prob_detail(crit)
            print(f'ImBD criterion is {crit:.4f}, suggesting that the text has a probability of',sep=" ")
            for task, prob in zip(prob_estimator.tasks, probs):
                print(f'{prob * 100:.0f}% to be machine-{task},',sep=" ")
        else:
            prob = prob_estimator.crit_to_prob(crit)
            print(f'ImBD criterion is {crit:.4f}, suggesting that the text has a probability of {prob * 100:.0f}% to be machine-{args.task}.')
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_path', type=str, default="./local_infer_ref")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    parser.add_argument('--task', type=str, default="generate", choices=["polish", "generate", "rewrite", "expand", "all"])
    parser.add_argument('--detail', type=bool, default=False)
    parser.add_argument('--from_pretrained', type=str, default='./ckpt/ai_detection_polish_500_spo_lr_0.0001_beta_0.05_a_1')
    args = parser.parse_args()

    run(args)
