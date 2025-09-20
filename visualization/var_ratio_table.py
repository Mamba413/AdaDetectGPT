import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import json
from scipy import stats
from scipy.stats import norm

plot_path = 'visualization/normal_results'
include_legend = False

def get_results():
    datasets=["xsum", "squad", "writing"]
    source_models=["gpt2-xl", "gpt-neo-2.7B"]
    
    methods2 = {
        # 'sampling_discrepancy_analytic': 'FastDetectGPT',
        'bspline': 'AdaDetectGPT',
        # 'identity': 'AdaDetectGPT',
    }
    
    result_dir_template = 'exp_variance/results'
            
    def get_stats(result_file):
        with open(result_file, 'r') as fin:
            res = json.load(fin)
            return res['ratio_var/mean']

    def get_var_stats(result_file):
        with open(result_file, 'r') as fin:
            res = json.load(fin)
            return res['ratio_var']

    def get_mean_stats(result_file):
        with open(result_file, 'r') as fin:
            res = json.load(fin)
            return res['ratio_mean']

    def _get_method_stats(dataset, model, method, case):
        result_file = f'{result_dir_template}/{dataset}_{model}.{method}.json'
        if os.path.exists(result_file):
            if case == 'ratio':
                stats = np.array(get_stats(result_file))
            elif case == 'mean':
                stats = np.array(get_mean_stats(result_file))
            elif case == 'var':
                stats = np.array(get_var_stats(result_file))
        else:
            stats = np.array([0.0])
        return stats

    ratio_list_len = []
    for dataset in datasets:
        for model in source_models:
            for method in methods2:
                cols = _get_method_stats(dataset, model, method, 'ratio')
                ratio_list_len.append(len(cols))
    
    L = min(ratio_list_len)
    print("Minimum length of ratio list across datasets: ", L)
    n_l = np.array([100, 150, 200, 250, 300, 350])
    print('L: ' + ' '.join(n_l.astype(str).tolist()))
    print('------- Table of Mean -------')
    for dataset in datasets:
        for model in source_models:
            cols = _get_method_stats(dataset, model, method, 'mean')
            res_list = []
            for i, num in enumerate(n_l):
                res_list.append(cols[num])
            cols = [f'{col:.4f}' for col in res_list]
            cols = ' '.join(cols)
            print(f'{dataset}_{model}: {cols}')
    print('------- Table of Variance -------')
    for dataset in datasets:
        for model in source_models:
            cols = _get_method_stats(dataset, model, method, 'var')
            res_list = []
            for i, num in enumerate(n_l):
                res_list.append(cols[num])
            cols = [f'{col:.4f}' for col in res_list]
            cols = ' '.join(cols)
            print(f'{dataset}_{model}: {cols}')


if __name__ == '__main__':
    get_results()