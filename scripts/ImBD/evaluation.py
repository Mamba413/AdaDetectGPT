import numpy as np
import torch
import argparse
from .spo import get_sampling_discrepancy_analytic
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import time
from metrics import get_roc_metrics, get_precision_recall_metrics
import tqdm
from torch.utils.data import DataLoader
from .dataset import CustomDataset_rewrite
import os
import json

def run(args):
    
    # Generate the inference checkpoint from trained model
    # from spo import ComputeScore
    # model = ComputeScore("gpt-neo-2.7B", "gpt-neo-2.7B", cache_dir=args.cache_dir)
    # model.from_pretrained(args.from_pretrained)
    # scoring_model = model.scoring_model
    # scoring_tokenizer = model.scoring_tokenizer
    # scoring_model.save_pretrained('models/ImBD-inference')
    # scoring_tokenizer.save_pretrained('models/ImBD-inference')
    
    model_name = "models/ImBD-inference"
    print('Loading model')
    start_time = time.time()
    scoring_model = AutoPeftModelForCausalLM.from_pretrained(model_name) # Make sure you have downloaded the gpt-neo-2.7b and place it at `models` folder.
    scoring_tokenizer = AutoTokenizer.from_pretrained(model_name)
    scoring_tokenizer.pad_token = scoring_tokenizer.eos_token
    scoring_model.to(args.device)
    scoring_model.eval()
    print(f'Done. ({time.time()-start_time:.2f}s)')
    
    criterion_fn = get_sampling_discrepancy_analytic
    print('Loading dataset')
    start_time = time.time()
    val_data = CustomDataset_rewrite(data_json_dir=args.eval_dataset)
    print(f'Done. ({time.time()-start_time:.2f}s)')
    print(f"Evaluating on {args.eval_dataset.split('/')[-1]}")
    evaluate_model(scoring_model, scoring_tokenizer, criterion_fn, val_data, args)
    
        
def evaluate_model(model, tokenizer, criterion_fn, data, args):
    eval_loader = DataLoader(data, batch_size=1, shuffle=False)
    epoch_crit_train_original, epoch_crit_train_sampled = [],[]
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm.tqdm(eval_loader, desc=f"Evaluating {args.eval_dataset.split('/')[-1]}"):
            output = []
            for text in batch:
                tokenized = tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                labels = tokenized.input_ids[:, 1:]
                with torch.no_grad():
                    logits_score = model(**tokenized).logits[:, :-1]
                    logits_ref = logits_score
                    crit, _ = criterion_fn(logits_ref, logits_score, labels)
                crit = crit.cpu().numpy().item()
                output.append(crit)
                # output.append(inv_crit)
            crit_original, crit_sampled = output
            epoch_crit_train_original.append(crit_original)
            epoch_crit_train_sampled.append(crit_sampled)
            
        print(f"Total time: {time.time() - start_time:.4f}s")
        fpr, tpr, roc_auc = get_roc_metrics(epoch_crit_train_original, epoch_crit_train_sampled)
        p, r, pr_auc = get_precision_recall_metrics(epoch_crit_train_original, epoch_crit_train_sampled)
        os.makedirs("./evaluation", exist_ok=True)
        with open(os.path.join("./evaluation", args.eval_dataset.split("/")[-1].replace("raw_data", str(roc_auc)[:7])),'w') as f:
            json.dump({"fpr":fpr, "tpr":tpr, "roc_auc":roc_auc, "crit_real":epoch_crit_train_original, "crit_sampled":epoch_crit_train_sampled, "pr_auc":pr_auc}, f)
        print(f"ROC AUC:{roc_auc}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    parser.add_argument('--eval_dataset', type=str, default="./data/polish/gpt-4o/xsum_polish_gpt-4o.raw_data.json")
    # parser.add_argument('--from_pretrained', type=str, default='./ckpt/ai_detection_polish_500_spo_lr_0.0001_beta_0.05_a_1')
    args = parser.parse_args()

    run(args)
