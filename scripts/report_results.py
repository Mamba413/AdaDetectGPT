# Copyright (c) Jin Zhu.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os.path
import argparse
import json
import numpy as np
from scipy.stats import norm

def save_lines(lines, file):
    with open(file, 'w') as fout:
        fout.write('\n'.join(lines))

def get_auroc(result_file):
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        return res['metrics']['roc_auc']

def get_sampled_mean(result_file):
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        return np.mean(res['predictions']['samples'])

def get_fpr_tpr(result_file):
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        return res['metrics']['fpr'], res['metrics']['tpr']

def report_main_results(args):
    datasets = {
        'xsum': 'XSum',
        'squad': 'SQuAD',
        'writing': 'WritingPrompts', 
        'yelp': 'Yelp', 
        'essay': 'Essay',
    }
    source_models = {
        'gpt2-xl': 'GPT-2',
        'opt-2.7b': 'OPT-2.7',
        'gpt-neo-2.7B': 'Neo-2.7',
        'gpt-j-6B': 'GPT-J',
        'gpt-neox-20b': 'NeoX', 
    }
    methods1 = {
        'likelihood': 'Likelihood',
        'entropy': 'Entropy',
        'logrank': 'LogRank',
        'lrr': 'LRR',
        'npr': 'NPR', 
        'dna_gpt': 'DNAGPT',
    }
    methods2 = {
        'perturbation_100': 'DetectGPT',
        'sampling_discrepancy_analytic': 'FastDetectGPT',
        'AdaDetectGPT.bspline': 'AdaDetectGPT',
        }

    def _get_method_aurocs(dataset, method, filter=''):
        cols = []
        for model in source_models:
            result_file = f'{args.result_path}/{dataset}_{model}{filter}.{method}.json'
            if os.path.exists(result_file):
                auroc = get_auroc(result_file)
            else:
                auroc = 0.0
            cols.append(auroc)
        cols.append(np.mean(cols))
        return cols

    headers = ['Method'] + [source_models[model] for model in source_models] + ['Avg.']
    for dataset in datasets:
        print('----')
        print(datasets[dataset])
        print('----')
        print(' '.join(headers))
        # basic methods
        for method in methods1:
            method_name = methods1[method]
            cols = _get_method_aurocs(dataset, method)
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
        # white-box comparison
        results = {}
        for method in methods2:
            method_name = methods2[method]
            cols = _get_method_aurocs(dataset, method)
            results[method_name] = cols
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
        cols = np.array(results['AdaDetectGPT']) - np.array(results['FastDetectGPT'])
        cols = [f'{col:.4f}' for col in cols]
        relatives = np.array(results['AdaDetectGPT']) - np.array(results['FastDetectGPT'])
        relatives = 100 * relatives / (1.0 - np.array(results['FastDetectGPT']))
        relatives = [f'{relative:.4f}' for relative in relatives]
        print('Relative', ' '.join(relatives))
        ## black-box comparison
        print('>>>>>>>>>>>> Black-box comparison >>>>>>>>>>>>')
        methods3 = {
            'sampling_discrepancy_analytic': 'FastDetectGPT', 
            'AdaDetectGPT.bspline': 'AdaDetectGPT',
            }
        filters = {
            'sampling_discrepancy_analytic': '.gpt-j-6B_gpt-neo-2.7B', 
            'AdaDetectGPT.bspline': '.gpt-j-6B_gpt-neo-2.7B',
        } 
        results = {}
        for method in methods3:
            method_name = methods3[method]
            cols = _get_method_aurocs(dataset, method, filters[method])
            results[method_name] = cols
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
        relatives = np.array(results['AdaDetectGPT']) - np.array(results['FastDetectGPT'])
        relatives = 100 * relatives / (1.0 - np.array(results['FastDetectGPT']))
        relatives = [f'{relative:.4f}' for relative in relatives]
        print('Relative', ' '.join(relatives))

def report_main_ext_results(args):
    datasets = {
        'xsum': 'XSum',
        'squad': 'SQuAD',
        'writing': 'WritingPrompts', 
        'yelp': 'Yelp', 
        'essay': 'Essay',
    }
    source_models = {
        'qwen-7b': 'Qwen2.5', 
        'mistralai-7b': 'Mistral', 
        'llama3-8b': 'LLaMA3',
    }
    methods1 = {
        'likelihood': 'Likelihood',
        'entropy': 'Entropy',
        'logrank': 'LogRank',
        'lrr': 'LRR',
        'npr': 'NPR', 
        'dna_gpt': 'DNAGPT',
    }
    methods2 = {
        'perturbation_100': 'DetectGPT',
        'sampling_discrepancy_analytic': 'FastDetectGPT',
        'binoculars': 'Binoculars',
        'fluoroscopy': 'TextFluoroscopy',
        'radar': 'RADAR',
        'imbd': 'ImBD',
        'biscope': 'BiScope',
        'classification.bspline': 'AdaDetectGPT',
        }

    def _get_method_aurocs(dataset, method, filter=''):
        cols = []
        for model in source_models:
            result_file = f'{args.result_path}/{dataset}_{model}{filter}.{method}.json'
            if os.path.exists(result_file):
                auroc = get_auroc(result_file)
            else:
                auroc = 0.0
            cols.append(auroc)
        cols.append(np.mean(cols))
        return cols

    headers = ['Method'] + [source_models[model] for model in source_models] + ['Avg.']
    for dataset in datasets:
        print('----')
        print(datasets[dataset])
        print('----')
        print(' '.join(headers))
        # basic methods
        for method in methods1:
            method_name = methods1[method]
            cols = _get_method_aurocs(dataset, method)
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
        # white-box comparison
        results = {}
        for method in methods2:
            method_name = methods2[method]
            cols = _get_method_aurocs(dataset, method)
            results[method_name] = cols
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
        cols = np.array(results['AdaDetectGPT']) - np.array(results['FastDetectGPT'])
        cols = [f'{col:.4f}' for col in cols]
        relatives = np.array(results['AdaDetectGPT']) - np.array(results['FastDetectGPT'])
        relatives = 100 * relatives / (1.0 - np.array(results['FastDetectGPT']))
        relatives = [f'{relative:.4f}' for relative in relatives]
        print('Relative (over FastDetect)', ' '.join(relatives))

        relatives = np.array(results['AdaDetectGPT']) - np.array(results['Binoculars'])
        relatives = 100 * relatives / (1.0 - np.array(results['Binoculars']))
        relatives = [f'{relative:.4f}' for relative in relatives]
        print('Relative (over Binoculars)', ' '.join(relatives))

def report_chatgpt_gpt4_results(args):
    datasets = {
        'xsum': 'XSum',
        'writing': 'Writing',
        'pubmed': 'PubMed',
    }
    source_models = {
        'gpt-3.5-turbo': 'ChatGPT',
        'gpt-4': 'GPT-4',
    }
    score_models = { 't5-11b': 'T5-11B',
                     'gpt2-xl': 'GPT-2',
                     'opt-2.7b': 'OPT-2.7',
                     'gpt-neo-2.7B': 'Neo-2.7',
                     'gpt-j-6B': 'GPT-J',
                     'gpt-neox-20b': 'NeoX', 
                     'falcon-7b': 'Falcon-7B',
                     'falcon-7b-instruct': 'Falcon-7B-Instruct',
                     }
    methods1 = {
        'roberta-base-openai-detector': 'RoBERTa-base',
        'roberta-large-openai-detector': 'RoBERTa-large'
    }
    methods2 = {
        'likelihood': 'Likelihood', 
        'entropy': 'Entropy', 
        'logrank': 'LogRank',
    }
    methods3 = {
        'lrr': 'LRR', 
        'npr': 'NPR', 
        'perturbation_100': 'DetectGPT',
        'sampling_discrepancy_analytic': 'FastDetectGPT', 
        'classification.bspline': 'AdaDetectGPT', 
    }

    def _get_method_aurocs(method, filter=''):
        results = []
        for model in source_models:
            cols = []
            for dataset in datasets:
                result_file = f'{args.result_path}/{dataset}_{model}{filter}.{method}.json'
                if os.path.exists(result_file):
                    auroc = get_auroc(result_file)
                else:
                    auroc = 0.0
                cols.append(auroc)
            cols.append(np.mean(cols))
            results.extend(cols)
        return results

    def _get_method_aurocs2(method, filter=''):
        results = []
        for model in source_models:
            cols = []
            for dataset in datasets:
                result_file = f'{args.result_path}/{dataset}_{model}{filter}.{method}.json'
                if os.path.exists(result_file):
                    auroc = get_auroc(result_file)
                else:
                    auroc = 0.0
                cols.append(auroc)
            cols.append(np.mean(cols))
            results.extend(cols)
        return results

    headers1 = ['--'] + [source_models[model] for model in source_models]
    data_list = [datasets[d] for d in datasets] 
    data_list = data_list + ["Avg."] 
    headers2 = ['Method'] + data_list * len(source_models)
    print(' '.join(headers1))
    print(' '.join(headers2))
    # supervised methods
    for method in methods1:
        method_name = methods1[method]
        cols = _get_method_aurocs(method)
        cols = [f'{col:.4f}' for col in cols]
        print(method_name, ' '.join(cols))
    # zero-shot methods
    filters2 = {
        'likelihood': ['.gpt-j-6B'],
        'entropy': ['.gpt-j-6B'],
        'logrank': ['.gpt-j-6B'],
    }
    filters3 = {
        'lrr': ['.t5-11b_gpt2-xl', '.t5-11b_gpt-neo-2.7B', '.t5-11b_gpt-j-6B', '.t5-11b_gpt-neox-20b'],
        'npr': ['.t5-11b_gpt2-xl', '.t5-11b_gpt-neo-2.7B', '.t5-11b_gpt-j-6B', '.t5-11b_gpt-neox-20b'],
        'perturbation_100': ['.t5-11b_gpt2-xl', '.t5-11b_gpt-neo-2.7B', '.t5-11b_gpt-j-6B', '.t5-11b_gpt-neox-20b'],
        'sampling_discrepancy_analytic': [
            '.gpt-j-6B_gpt-neo-2.7B', 
        ], 
        'AdaDetectGPT.bspline': [
            ".gpt-j-6B_gpt-neo-2.7B", 
        ],
    }
    for method in methods2:
        for filter in filters2[method]:
            setting = score_models[filter[1:]]
            method_name = f'{methods2[method]}({setting})'
            cols = _get_method_aurocs(method, filter)
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
    for method in methods4:
        method_name = methods4[method]
        cols = _get_method_aurocs2(method)
        if method_name == "SuperAdaDetectGPT":
            super_ada_res = cols 
        if method_name == "ImBD":
            imbd_res = cols
        cols = [f'{col:.4f}' for col in cols]
        print(method_name, ' '.join(cols))
    for method in methods3:
        for filter in filters3[method]:
            setting = [score_models[model] for model in filter[1:].split('_')]
            method_name = f'{methods3[method]}({setting[0]}/{setting[1]})'
            cols = _get_method_aurocs(method, filter)
            if methods3[method] == "AdaDetectGPT":
                ada_res = cols 
            if method_name == "FastDetectGPT":
                fast_res1 = cols
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
    relatives = np.array(ada_res) - np.array(fast_res1)
    relatives = 100 * relatives / (1.0 - np.array(fast_res1))
    relatives = [f'{relative:.4f}' for relative in relatives]
    print('Relative (Ada over Fast)', ' '.join(relatives))

def report_advance_llm_results(args):
    datasets = {
        'xsum': 'XSum',
        'writing': 'Writing',
        'yelp': 'Yelp', 
        'essay': 'Essay',
    }
    source_models = {
        'gpt-4o': 'GPT-4o',
        'gemini-2.5-flash': 'Gemini-2.5',
        'claude-3-5-haiku': 'Claude-3.5',
    }
    score_models = {
        'gemma-9b': 'Gemma2-9B',
        'gemma-9b-instruct': 'Gemma2-9B-Instruct',
    }
    methods1 = {
        'roberta-base-openai-detector': 'RoBERTa-base',
        'roberta-large-openai-detector': 'RoBERTa-large'
    }
    methods2 = {
        'likelihood': 'Likelihood', 
        'entropy': 'Entropy', 
        'logrank': 'LogRank',
    }
    methods3 = {
        'sampling_discrepancy_analytic': 'FastDetectGPT', 
        'AdaDetectGPT.bspline': 'AdaDetectGPT', 
    }
    methods4 = {
        'radar': 'RADAR',
        'biscope': 'BiScope',
    }

    def _get_method_aurocs(method, filter=''):
        results = []
        for model in source_models:
            cols = []
            for dataset in datasets:
                result_file = f'{args.result_path}/{dataset}_{model}{filter}.{method}.json'
                if os.path.exists(result_file):
                    auroc = get_auroc(result_file)
                else:
                    auroc = 0.0
                cols.append(auroc)
            cols.append(np.mean(cols))
            results.extend(cols)
        return results

    def _get_method_aurocs2(method, filter=''):
        results = []
        for model in source_models:
            cols = []
            for dataset in datasets:
                result_file = f'{args.result_path}/{dataset}_{model}{filter}.{method}.json'
                if os.path.exists(result_file):
                    auroc = get_auroc(result_file)
                else:
                    auroc = 0.0
                cols.append(auroc)
            cols.append(np.mean(cols))
            results.extend(cols)
        return results

    headers1 = ['--'] + [source_models[model] for model in source_models]
    data_list = [datasets[d] for d in datasets] 
    data_list = data_list + ["Avg."] 
    headers2 = ['Method'] + data_list * len(source_models)
    print(' '.join(headers1))
    print(' '.join(headers2))
    # supervised methods
    for method in methods1:
        method_name = methods1[method]
        cols = _get_method_aurocs(method)
        cols = [f'{col:.4f}' for col in cols]
        print(method_name, ' '.join(cols))
    # zero-shot methods
    filters2 = {
        'likelihood': ['.gemma-9b'],
        'entropy': ['.gemma-9b'],
        'logrank': ['.gemma-9b'],
    }
    filters3 = {
        'sampling_discrepancy_analytic': [
            '.falcon-7b_falcon-7b-instruct', 
            ".gemma-9b_gemma-9b-instruct"
        ], 
        'AdaDetectGPT.bspline': [
            ".gemma-9b_gemma-9b-instruct", 
        ],
    }
    for method in methods2:
        for filter in filters2[method]:
            setting = score_models[filter[1:]]
            method_name = f'{methods2[method]}({setting})'
            cols = _get_method_aurocs(method, filter)
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
    for method in methods4:
        method_name = methods4[method]
        cols = _get_method_aurocs2(method)
        cols = [f'{col:.4f}' for col in cols]
        print(method_name, ' '.join(cols))
    for method in methods3:
        for filter in filters3[method]:
            setting = [score_models[model] for model in filter[1:].split('_')]
            method_name = f'{methods3[method]}({setting[0]}/{setting[1]})'
            cols = _get_method_aurocs(method, filter)
            if methods3[method] == "AdaDetectGPT":
                ada_res = cols 
            if method_name == "FastDetectGPT(Gemma2-9B/Gemma2-9B-Instruct)":
                fast_res1 = cols
            cols = [f'{col:.4f}' for col in cols]
            print(method_name, ' '.join(cols))
    relatives = np.array(ada_res) - np.array(fast_res1)
    relatives = 100 * relatives / (1.0 - np.array(fast_res1))
    relatives = [f'{relative:.4f}' if relative > 0 else '---' for relative in relatives]
    print('Relative (Ada over Fast)', ' '.join(relatives))

def get_power(result_file, alpha=0.1):
    critical_value = norm.ppf(1 - alpha, loc=0.0, scale=1.0)
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        fake_stats = np.array(res['predictions']['samples'])
        return np.mean(fake_stats > critical_value)  

def get_tpr(result_file, alpha=0.1):
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        real_stats = np.array(res['predictions']['real'])
        fake_stats = np.array(res['predictions']['samples'])
        critical_value = np.quantile(fake_stats, q=alpha)
        return np.mean(real_stats <= critical_value)

def get_fpr(result_file, alpha=0.1):
    with open(result_file, 'r') as fin:
        res = json.load(fin)
        real_stats = np.array(res['predictions']['real'])
        fake_stats = np.array(res['predictions']['samples'])
        critical_value = np.quantile(real_stats, q=1-alpha)
        return np.mean(fake_stats <= critical_value)  

def report_tpr_fpr(args):
    datasets = {'xsum': 'XSum',
                'squad': 'SQuAD',
                'writing': 'WritingPrompts'}
    source_models = {'gpt2-xl': 'GPT-2',
                     'opt-2.7b': 'OPT-2.7',
                     'gpt-neo-2.7B': 'Neo-2.7',
                    #  'gpt-j-6B': 'GPT-J',
                    #  'gpt-neox-20b': 'NeoX', 
                     }
    methods2 = {
        'AdaDetectGPT.bspline': 'AdaDetectGPT',
        'sampling_discrepancy_analytic': 'FastDetectGPT',
        }

    def _get_method_fpr(dataset, method, alpha):
        cols = []
        for model in source_models:
            result_file = f'{args.result_path}/{dataset}_{model}.{method}.json'
            if os.path.exists(result_file):
                error = get_fpr(result_file, alpha)
            else:
                error = 0.0
            cols.append(error)
        cols.append(np.mean(cols))
        return cols

    def _get_method_tpr(dataset, method, alpha):
        cols = []
        for model in source_models:
            result_file = f'{args.result_path}/{dataset}_{model}.{method}.json'
            if os.path.exists(result_file):
                error = get_tpr(result_file, alpha)
            else:
                error = 0.0
            cols.append(error)
        cols.append(np.mean(cols))
        return cols

    headers = ['Method'] + [source_models[model] for model in source_models] + ['Avg.']
    # white-box comparison
    for dataset in datasets:
        print('----')
        print(datasets[dataset] + ' (FPR)')
        print('----')
        print(' '.join(headers))
        results = {}
        for method in methods2:
            method_name = methods2[method]
            cols = _get_method_fpr(dataset, method, args.alpha)
            results[method_name] = cols
            cols = [f'{col:.3f}' for col in cols]
            print(method_name, ' '.join(cols))
    for dataset in datasets:
        print('----')
        print(datasets[dataset] + ' (TPR)')
        print('----')
        print(' '.join(headers))
        results_power = {}
        for method in methods2:
            method_name = methods2[method]
            cols = _get_method_tpr(dataset, method, args.alpha)
            results_power[method_name] = cols
            cols = [f'{col:.3f}' for col in cols]
            print(method_name, ' '.join(cols))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    if args.report_name == 'main_results':
        report_main_results(args)
    elif args.report_name == 'main_ext_results':
        report_main_ext_results(args)
    elif args.report_name == 'chatgpt_gpt4_results':
        report_chatgpt_gpt4_results(args)
    elif args.report_name == 'advance_llm_results':
        report_advance_llm_results(args)