from ImBD.dataset import CustomDataset_rewrite
from ImBD.spo import ComputeScore
from ImBD.engine import run
import torch
from torch.utils.data import Subset
import argparse
import numpy as np
import random
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('-a', type=int, default=1, help="accumulation steps")
    parser.add_argument('--task_name', type=str, default="ai_detection_500")
    parser.add_argument('--epochs', type=int, default=2, help="finetuning epochs")
    parser.add_argument('-ebt', action="store_true", help="Evaluate model before tuning")
    parser.add_argument('--datanum', type=int, default=500, help="num of training data")
    parser.add_argument('--eval_only', action="store_true")
    parser.add_argument('--eval_after_train', action="store_true")
    parser.add_argument('--SPOtrained', type=str, default="True", choices=["True", "False"], help="If false, means finetuned base model (ablation)")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--from_pretrained', type=str)
    parser.add_argument('--eval_dataset', type=str, default="./exp_main/data/xsum_gpt-neo-2.7B")
    parser.add_argument('--output_file', type=str, default="./exp_main/results/")
    parser.add_argument('--base_model', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    parser.add_argument('--train_dataset', type=str, default='./exp_main/data/squad_gpt-neo-2.7B')
    args = parser.parse_args()

    train_data = CustomDataset_rewrite(data_json_dir=args.train_dataset) 
    val_data = CustomDataset_rewrite(data_json_dir=args.eval_dataset)

    set_seed(args.seed)
    SPOtrained = True if args.SPOtrained == "True" else False
    print(f"Running with args: {args}")
    model = ComputeScore(args.base_model, args.base_model, SPOtrained=SPOtrained, SPO_beta=args.beta, cache_dir=args.cache_dir)
    
    if args.from_pretrained:
        print(f"Loading ckpt from {args.from_pretrained}...")
        model.from_pretrained(args.from_pretrained)
    
    # if not os.path.exists(args.output_file):
    #     os.makedirs(args.output_file)
    subset_indices = torch.randperm(len(train_data))[:args.datanum]
    
    # with open("indice.txt","w+") as f:
    #     f.write(str(subset_indices.tolist()))
        
    train_subset = Subset(train_data, subset_indices)
    print(len(train_subset))
    print(len(val_data))
    
    run(
        model, 
        [train_subset, val_data], 
        DEVICE='cuda', 
        ckpt_dir=f"./scripts/ImBD/ckpt/{args.task_name}_spo_lr_{args.lr}_beta_{args.beta}_a_{args.a}",
        args=args
    )