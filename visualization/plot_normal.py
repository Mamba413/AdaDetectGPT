import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import json
from scipy import stats
from scipy.stats import norm

CASES = ['exact', 'inexact']
plot_path = 'visualization/normal_results'
include_legend = False

def get_results():
    datasets=["xsum", "squad", "writing"]
    source_models=["gpt2-xl", "gpt-neo-2.7B"]
    
    methods2 = {
        'classification': 'AdaDetectGPT',
        }
    
    result_dir_template = 'exp_normal/results_{}'

    def get_stats(result_file):
        with open(result_file, 'r') as fin:
            res = json.load(fin)
            return res['predictions']['samples']

    def _get_method_stats(dataset, model, method, cases, filter=''):
        res_path = result_dir_template.format(cases)
        result_file = f'{res_path}/{dataset}_{model}.{method}.json'
        if os.path.exists(result_file):
            stats = np.array(get_stats(result_file))
        else:
            stats = np.array([0.0])
        return stats

    result_list = []
    for cases in CASES:
        for dataset in datasets:
            for model in source_models:
                for method in methods2:
                    results = {'datasets': dataset, 'models': model, 'cases': cases}
                    method_name = methods2[method]
                    results['methods'] = method_name
                    cols = _get_method_stats(dataset, model, method, cases)
                    results['values'] = cols
                    # print(f"{method_name} with mean {np.mean(cols)} and std {np.std(cols)} on {dataset} with cases {cases}")
                    result_list.append(results)
    
    def merge_dicts_of_lists(dataset_list: list[dict]) -> dict:
        """
        将一系列 dict(键→list) 合并为一个 dict，
        同一个键对应的 list 会被 extend 到一起。
        """
        merged = defaultdict(list)
        for d in dataset_list:
            for key, value in d.items():
                # 如果 value 本身是 list，则 extend；否则 append
                if isinstance(value, list):
                    merged[key].extend(value)
                else:
                    merged[key].append(value)
        return dict(merged)
    
    result_list = merge_dicts_of_lists(result_list)
    df = pd.DataFrame(result_list)
    df['values'] = df['values'].apply(lambda arr: arr.tolist())
    df = df.explode('values').reset_index(drop=True)
    # print(df)
    return df

def plot_hist_by_dataset_and_method(df):
    datasets = sorted(df['datasets'].unique())
    models = sorted(df['models'].unique())
    # datasets.remove('xsum')
    methods  = sorted(df['methods'].unique())

    # pick a distinct color for each cases
    N = len(CASES)
    cmap   = plt.get_cmap('viridis')
    palette = cmap(np.linspace(1, 0, N))

    fig, axes = plt.subplots(
        nrows=len(methods),
        ncols=len(datasets),
        figsize=(3.9*len(datasets), 3.3*len(methods)),
        sharey=False,
    )

    # if only one row or one col, axes may be 1-D
    if len(methods) == 1 and len(datasets) == 1:
        axes = [[axes]]
    elif len(methods) == 1:
        axes = [axes]
    elif len(datasets) == 1:
        axes = [[ax] for ax in axes]

    for i, method in enumerate(methods):
        for j, dataset in enumerate(datasets):
            ax = axes[i][j]
            sub = df[(df['methods']==method) & (df['datasets']==dataset)]

            value_list = []
            model_list = []
            for l, model in enumerate(models):
                sub2 = sub[sub['models']==model]

                # draw one histogram per cases
                for k, n in enumerate(CASES):
                    vals = sub2[sub2['cases']==n]['values']
                    value_list.append(vals)
                    model_list.append(model)

                print(f"------------ Test on {model} at {dataset} ------------")
                test_result = stats.kstest(vals.to_numpy().astype(np.float16), stats.norm.cdf)
                print("KS test: ", test_result.pvalue)
                test_result = stats.shapiro(vals.to_numpy().astype(np.float16))
                print("SW test: ", test_result.pvalue)
                test_result = stats.anderson(vals.to_numpy().astype(np.float16), dist='norm')
                print("Anderson test: ", test_result.statistic, test_result.critical_values, test_result.significance_level)
                print(f"------------------------------------------------------")
                
            _, bins, _ = ax.hist(
                value_list,
                density=True, 
                bins=12,
                alpha=0.6,
                range=(-3, 3),
                histtype='stepfilled',
                color=['#0571b0', '#f4a582'],
                label=model_list,
            )
            # overlay the standard normal curve
            x = np.linspace(bins[0], bins[-1], 200)
            y = norm.pdf(x, loc=0, scale=1)   # standard normal
            ax.plot(
                x, y,
                color='darkred',
                linestyle='--',
                linewidth=2,
            )

            ax.set_ylabel('density', fontsize=12, fontweight='bold')
            # titles & labels
            if i == 0:
                pass
            if j == 0:
                pass
            if i == len(methods)-1:
                ax.set_xlabel('statistics', fontsize=13, fontweight='bold')

    # common legend on the right
    handles, labels = axes[0][-1].get_legend_handles_labels()
    if include_legend:
        fig.legend(
            handles, labels,
            # title="cases",
            loc='lower center',
            bbox_to_anchor=(0.5, -0.017), 
            ncol=5,
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.24)
    plt.savefig(f'{plot_path}/normal_histogram.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def plot_box_by_dataset_and_method(df):
    """
    df must have columns:
      - 'datasets' (categorical)
      - 'methods'  (categorical)
      - 'cases' (numeric, e.g. 2,4,8,16)
      - 'values'   (each entry is a 1D array or list of numbers)
    """
    datasets = sorted(df['datasets'].unique())
    methods  = sorted(df['methods'].unique())

    n_rows, n_cols = len(methods), len(datasets)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4*n_cols, 4*n_rows),
        sharey=False
    )

    # In case of single row/col, normalize axes to 2D list
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    # pick a color for each cases
    N = len(CASES)
    cmap   = plt.get_cmap('viridis')
    palette = cmap(np.linspace(1, 0, N))

    for i, method in enumerate(methods):
        for j, dataset in enumerate(datasets):
            ax = axes[i][j]
            sub = df[(df['datasets']==dataset) & (df['methods']==method)]

            # build list-of-lists for each prompt
            data_list = []
            for n in CASES:
                block = sub[sub['cases']==n]
                print("KS test for ", n, f"at {dataset} {method}")
                print(stats.kstest(block['values'], stats.norm.cdf))
                data_list.append(block['values'])

            # draw the boxplots at positions 0,1,2,3
            bp = ax.boxplot(
                data_list,
                positions=range(len(CASES)),
                patch_artist=True,
                widths=0.6,
                medianprops=dict(color="black")
            )
            # color them
            for patch, color in zip(bp['boxes'], palette):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # x‐ticks & labels
            ax.set_xticks(list(range(len(CASES))))
            ax.set_xticklabels([str(n) for n in CASES])
            ax.set_xlabel("cases")
            if j == 0:
                ax.set_ylabel(method, fontweight="bold")
            if i == 0:
                pass
                ax.set_title(dataset, fontweight="bold")

    # shared legend in upper right
    handles = [
        plt.Line2D([0],[0], color=palette[k], marker='s', linestyle='', alpha=0.7)
        for k in range(len(CASES))
    ]
    labels = [f"cases={n}" for n in CASES]
    fig.legend(
        handles, labels,
        # title="cases",
        loc='lower center',
        bbox_to_anchor=(0.5, 0.0), 
        ncol=5,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # leave extra space at the bottom
    plt.savefig(f'{plot_path}/normal_boxplot.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    df = get_results()
    plot_hist_by_dataset_and_method(df)
