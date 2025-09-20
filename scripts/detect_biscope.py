import os
import argparse
import random
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from metrics import get_roc_metrics, get_precision_recall_metrics
from sklearn.model_selection import cross_val_score
from BiScope.biscope_utils import data_generation
import torch
import json
from utils import load_training_data2

def parse_dataset_arg(ds):
    """
    Parse dataset string with expected format:
      {task}_{generative_model}
    For example: "Arxiv_gpt-3.5-turbo.raw_data.json"
    Returns a tuple: task, generative_model
    """
    parts = ds.split('_')
    generative_model = parts[1].split('.raw')[0]

    return parts[0], generative_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_main/results/yelp_llama3-8b")  # output file prefix
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--sample_clip', type=int, default=2000, help="Max token length for samples")
    parser.add_argument('--summary_model', type=str, default='none', help="Summary model key or 'none'")
    parser.add_argument('--base_dir', type=str, default="./exp_main/data", help="directory of data")
    parser.add_argument('--detect_model', type=str, default='llama2-7b', help="Detection model key")
    parser.add_argument('--train_dataset', type=str, default='essay_llama3-8b.raw_data.json',
                        help='Format: {task}_{generative_model}.raw_data.json')
    parser.add_argument('--test_dataset', type=str, default='yelp_llama3-8b.raw_data.json',
                        help='Format: {task}_{generative_model}.raw_data.json')
    parser.add_argument('--use_hf_dataset', type=bool, default=False, help="Load dataset from Hugging Face")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()
    
    name = 'biscope'

    if args.use_hf_dataset:
        print("Using Hugging Face datasets...")
    else:
        print("Using local datasets...")

    # Set seeds.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    if '&' in args.train_dataset:
        data_name_list = args.train_dataset.split('&')
        train_data = load_training_data2(data_name_list, args.base_dir)
        mix_data_name = "".join([x.split("_")[0] for x in data_name_list])
        model_name = data_name_list[0].split("_")[1]
        if model_name.endswith(".raw"):
            model_name = model_name[:-4]
        mix_data_name = f'{mix_data_name}_{model_name}.raw_data.json'
        print(f'Create Mixed data: {mix_data_name}')
        with open(f'./{args.base_dir}/{mix_data_name}', 'w') as fout:
            json.dump(train_data, fout)
        args.train_dataset = mix_data_name

    # Parse dataset arguments.
    train_task, train_gen = parse_dataset_arg(args.train_dataset)
    test_task, test_gen = parse_dataset_arg(args.test_dataset)
    
    # Create a base output directory that includes both train and test dataset strings.
    base_out_dir = os.path.join('./results', f"{args.test_dataset}.biscope")
    os.makedirs(base_out_dir, exist_ok=True)
    
    # Create separate subdirectories for train and test features.
    # If train and test datasets are identical, use the same directory.
    if args.train_dataset == args.test_dataset:
        train_dir = test_dir = base_out_dir
    else:
        train_dir = os.path.join(base_out_dir, "train")
        test_dir  = os.path.join(base_out_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
    
    # Generate features for the training dataset.
    print("Generating train features...")
    data_generation(args, train_dir, train_task, train_gen, args.base_dir)
    
    # Load train features.
    with open(os.path.join(train_dir, f"{train_task}_human_features.pkl"), 'rb') as f:
        train_human = np.array(pickle.load(f))
    with open(os.path.join(train_dir, f"{train_task}_GPT_features.pkl"), 'rb') as f:
        train_gpt = np.array(pickle.load(f))
    
    # If training and testing datasets are identical, use 5-fold CV.
    if args.train_dataset == args.test_dataset:
        feats = np.concatenate([train_human, train_gpt], axis=0)
        labels = np.concatenate([np.zeros(len(train_human)), np.ones(len(train_gpt))], axis=0)
        clf = RandomForestClassifier(n_estimators=100, random_state=args.seed)
        scores = cross_val_score(clf, feats, labels, cv=5, scoring='f1')
        print("5-fold CV F1 scores:", scores, "Average:", scores.mean())
        with open(os.path.join(base_out_dir, 'cv_scores.txt'), 'w') as f:
            f.write(" ".join(map(str, scores)))
    else:
        # For different train and test datasets, two cases are handled:
        # Case 1: Cross-model OOD setting: same task but different generative model/paraphrase status.
        if train_task == test_task:
            print("Evaluating cross-model OOD setting (same task):")
            # Train on human and GPT training features.
            train_feats = np.concatenate([train_human, train_gpt], axis=0)
            train_labels = np.concatenate([np.zeros(len(train_human)), np.ones(len(train_gpt))], axis=0)
            clf = RandomForestClassifier(n_estimators=100, random_state=args.seed)
            clf.fit(train_feats, train_labels)
            # Generate test features for GPT only.
            print("Generating test GPT features...")
            data_generation(args, test_dir, test_task, test_gen, args.base_dir)
            with open(os.path.join(test_dir, f"{test_task}_GPT_features.pkl"), 'rb') as f:
                test_gpt = np.array(pickle.load(f))
            with open(os.path.join(test_dir, f"{test_task}_human_features.pkl"), 'rb') as f:
                test_human = np.array(pickle.load(f))

        # Case 2: Cross-domain OOD setting: task changes.
        else:
            print("Evaluating cross-domain OOD setting (different task):")
            data_generation(args, test_dir, test_task, test_gen, args.base_dir)
            with open(os.path.join(test_dir, f"{test_task}_human_features.pkl"), 'rb') as f:
                test_human = np.array(pickle.load(f))
            with open(os.path.join(test_dir, f"{test_task}_GPT_features.pkl"), 'rb') as f:
                test_gpt = np.array(pickle.load(f))
            train_feats = np.concatenate([train_human, train_gpt], axis=0)
            train_labels = np.concatenate([np.zeros(len(train_human)), np.ones(len(train_gpt))], axis=0)
            # test_feats = np.concatenate([test_human, test_gpt], axis=0)
            # test_labels = np.concatenate([np.zeros(len(test_human)), np.ones(len(test_gpt))], axis=0)
            clf = RandomForestClassifier(n_estimators=100, random_state=args.seed)
            clf.fit(train_feats, train_labels)

        predictions = {'real': clf.predict(test_human).tolist(), 'samples': clf.predict(test_gpt).tolist()}
        print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
        fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
        p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
        print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
        # results
        results = { 'name': f'{name}_threshold',
                    'predictions': predictions,
                    'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
                    'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
                    'loss': 1 - pr_auc}

        results_file = f'{args.output_file}.{name}.json'
        with open(results_file, 'w') as fout:
            json.dump(results, fout)
            print(f'Results written into {results_file}')


if __name__ == '__main__':
    main()
