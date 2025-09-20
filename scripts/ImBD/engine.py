# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
import tqdm
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import json
from metrics import get_roc_metrics, get_precision_recall_metrics

from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from utils import GpuMem
try:
    from transformers import AdamW
except:
    from torch.optim import AdamW

def evaluate_model_SPO(model, data, DEVICE):
    model.to(DEVICE)
    model.eval()
    loss = 0
    eval_loader = DataLoader(data, batch_size=1, shuffle=False)
    epoch_crit_train_original, epoch_crit_train_sampled = [],[]
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm.tqdm(eval_loader, desc="Evaluating"):
            text = batch
            output = model(text)
            loss += output['loss'].item()
            epoch_crit_train_original.extend(output['crit'][1].tolist())
            epoch_crit_train_sampled.extend(output['crit'][3].tolist())
            
        print(f"Total time: {time.time() - start_time:.4f}s")
        avg_loss = loss / len(eval_loader)
        fpr, tpr, roc_auc = get_roc_metrics(epoch_crit_train_original, epoch_crit_train_sampled)
        p, r, pr_auc = get_precision_recall_metrics(epoch_crit_train_original, epoch_crit_train_sampled)
    
    # print(f"val_loss: {avg_loss:.6f}")
    print(f"val_ROC_AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    print(f"val_Real_mean/std: {np.mean(epoch_crit_train_original):.2f}/{np.std(epoch_crit_train_original):.2f}, val_Samples_mean/std: {np.mean(epoch_crit_train_sampled):.2f}/{np.std(epoch_crit_train_sampled):.2f}")
    print("="*10)
    
    results_dict = {
        "name": "imbd",
        'info': {'n_samples': len(epoch_crit_train_original)},
        'predictions': {'real': epoch_crit_train_original, 
                        'samples': epoch_crit_train_sampled},
        'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
        'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
    }
    return results_dict


def fine_tune_ours(model, data, DEVICE, ckpt_dir='./ckpt', args=None):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=f"./scripts/ImBD/logs/{args.task_name}_spo_lr_{args.lr}_beta_{args.beta}_a_{args.a}_{current_time}/train_ai_detection")

    train_loader = DataLoader(data[0], batch_size=1, shuffle=True)
    epochs = args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs, eta_min=0,
                                  last_epoch=-1)

    scaler = GradScaler()
    model.to(DEVICE)

    # Number of iterations for gradient accumulation
    accumulation_steps = args.a
    epoch_losses, i, loss = [], 0, torch.tensor(0.0).to(DEVICE)
    epoch_crit_train_original, epoch_crit_train_sampled = [],[]
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        start_time = time.time()
        for batch in tqdm.tqdm(train_loader, desc=f"Fine-tuning: {epoch} epoch"):
            text = batch
            scheduler.step()
            with autocast():
                outputs_1 = model(text)
                epoch_crit_train_original.extend(outputs_1['crit'][1].tolist())
                epoch_crit_train_sampled.extend(outputs_1['crit'][3].tolist())
                loss += (outputs_1['loss'].to(torch.float32)) / accumulation_steps
            
            
            if ((i + 1) % accumulation_steps) == 0:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                optimizer.zero_grad()
                scaler.update()
                writer.add_scalar('Loss/train', loss.item(), i)  
                epoch_losses.append(loss.item())
                loss = torch.tensor(0.0).to(DEVICE)
            epoch_losses.append(loss.item())
            i += 1
        print(f"Total time: {time.time() - start_time:.4f}s") 
        fpr, tpr, roc_auc = get_roc_metrics(epoch_crit_train_original, epoch_crit_train_sampled)
        p, r, pr_auc = get_precision_recall_metrics(epoch_crit_train_original, epoch_crit_train_sampled)
        
        print(f"ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
        print(f"Real mean/std: {np.mean(epoch_crit_train_original):.2f}/{np.std(epoch_crit_train_original):.2f}, Samples mean/std: {np.mean(epoch_crit_train_sampled):.2f}/{np.std(epoch_crit_train_sampled):.2f}")
        epoch_avg_loss = np.mean(epoch_losses)
        
        writer.add_scalar('Loss/epoch', epoch_avg_loss, epoch)
        writer.add_scalar('ROC_AUC/epoch', roc_auc, epoch)
        writer.add_scalar('PR_AUC/epoch', pr_auc, epoch)
        writer.add_scalar('Real_mean/epoch',np.mean(epoch_crit_train_original),epoch)
        writer.add_scalar('Real_std/epoch',np.std(epoch_crit_train_original),epoch)
        writer.add_scalar('Sampled_mean/epoch',np.mean(epoch_crit_train_sampled),epoch)
        writer.add_scalar('Sampled_std/epoch',np.std(epoch_crit_train_sampled),epoch)
        epoch_crit_train_original, epoch_crit_train_sampled = [],[] # reset crit
        print(f"\nAverage Loss for Epoch {epoch}: {epoch_avg_loss}")
            
    # if not os.path.exists(ckpt_dir):
    #     os.makedirs(ckpt_dir)
    # model.save_pretrained(ckpt_dir)
    # print(f"Saved finetuned model to {os.path.join(ckpt_dir, 'ours-finetuned.pth')}")
    
    writer.close()
    return model


def run(
    model, 
    data, 
    DEVICE,
    args,
    ckpt_dir='./ckpt',
    ):
    
    if args.ebt or args.eval_only:
        print("Evaluating model before tuning...")
        d = evaluate_model_SPO(model, data[1], DEVICE)
        if args.SPOtrained:
            output_path = f"{args.output_file}.imbd.json"
        else:
            method_name=args.base_model.split("_")[-1]
            output_path = f"{args.output_file}.{method_name}.json"
        with open(output_path, "w") as j:
            json.dump(d,j)
        print(f"Results saved to {output_path}.")
    if args.eval_only:
        return

    tracker = GpuMem()
    print('Fine-tuning model...')
    start = time.perf_counter()
    with tracker:
        model = fine_tune_ours(
            model, 
            data,
            DEVICE=DEVICE, 
            ckpt_dir=ckpt_dir,
            args=args
        )
    pre_time = time.perf_counter() - start
    pre_memory = tracker.memory_usage()
    
    if args.eval_after_train:
        print("Evaluating model after tuning...")
        start = time.perf_counter()
        with tracker:
            d = evaluate_model_SPO(model, data[1], DEVICE)
        eval_time = time.perf_counter() - start
        eval_time = eval_time / (len(data[1]) << 1)
        eval_memory = tracker.memory_usage()
        d['compute_info'] = {'pre_time': pre_time, 'eval_time': eval_time, 
                             'pre_memory': pre_memory, 'eval_memory': eval_memory,}
        if args.SPOtrained:
            output_path = f"{args.output_file}.imbd.json"
        else:
            method_name=args.base_model.split("_")[-1]
            output_path = f"{args.output_file}.{method_name}.json"
        with open(output_path, "w") as j:
            json.dump(d, j)
        print(f"Results saved to {output_path}.")
    



