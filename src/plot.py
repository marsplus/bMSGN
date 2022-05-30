
import argparse
import numpy as np
import pandas as pd
import dill as pickle
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--figure', type=int)
args = parser.parse_args()


def gen_figure1():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    for i in [0, 1, 2]:
        if i == 0:
            gType = 'BA'
        elif i == 1:
            gType = 'SW'
        else:
            gType = 'BTER'

        Data = pd.read_csv('../result/trend_{}.txt'.format(gType), header=None, sep=' ')
        Data = Data[[0, 2, 3, 4]]
        Data.columns = [0, 1, 2, 3]
        

        Data[[1, 2, 3]] = np.sqrt((Data[[1, 2, 3]] ** 2) / 100)
        Data.columns = ['length', '$\\mathbf{b_i-c_i}$', '$\\mathbf{\\beta_i}$', '$\\mathbf{\\eta_i}$']
        yerr = Data.groupby('length').sem() * 1.96
        Data.groupby('length').mean().plot(yerr=yerr, linewidth=4, ax=axes[i], linestyle='-.')
        plt.setp(axes[i].get_xticklabels(), fontsize=18, fontweight='bold')
        plt.setp(axes[i].get_yticklabels(), fontsize=18, fontweight='bold')
        axes[i].set_xlabel('', fontsize=24, fontweight='bold')
        axes[i].set_ylim([0, 2])
        if i == 0:
            axes[i].set_ylabel('RMSE', fontsize=24, fontweight='bold')
        plt.sca(axes[i])
        axes[i].get_legend().remove()
        axes[i].locator_params(axis='y', nbins=6)
        plt.xticks([250, 500, 750, 1000, 1250, 1500], rotation=45)
        
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.figlegend(by_label.values(), by_label.keys(), loc='upper center', \
                  bbox_to_anchor=(0.5, 1.25), ncol=3, fontsize=24)
    plt.tight_layout()
    plt.savefig('../result/figures/trend.pdf', dpi=300, bbox_inches='tight')



def gen_figure2():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for i in [0, 1, 2]:
        if i == 0:
            graph_type = 'BA'
        elif i == 1:
            graph_type = 'SW'
        else:
            graph_type = 'BTER'
            
        data = pd.read_csv('../result/liktest_withGroup.txt', sep=' ', header=None)
        data.columns = ['time step', 'graph', 'p-value']
        data = data[data['graph'] == graph_type]
        yerr = data.groupby('time step')['p-value'].sem() * 1.96
        data.groupby('time step')['p-value'].mean().plot(yerr=yerr, linestyle='solid', \
                                                                linewidth=4, label='with groups', capsize=10, ax=axes[i])

        data_no = pd.read_csv('../result/liktest_noGroup.txt', sep=' ', header=None)
        data_no.columns = ['time step', 'graph', 'p-value']
        data_no = data_no[data_no['graph'] == graph_type]
        yerr = data_no.groupby('time step')['p-value'].sem() * 1.96
        data_no.groupby('time step')['p-value'].mean().plot(yerr=yerr, linestyle='-.', \
                                                                   linewidth=4, label='without group', capsize=10, ax=axes[i])
        axes[i].hlines(0.05, 200, 2100, color='red', linestyle='--', linewidth=4, label='0.05')
        axes[i].set_xlim(200, 2100)
        axes[i].set_ylim(-0.05, 1.1)
        if i == 0:
            axes[i].set_ylabel('p-value', fontweight='bold', fontsize=24)
        axes[i].set_xlabel('', fontweight='bold')
        plt.sca(axes[i])
        plt.setp(axes[i].get_xticklabels(), fontsize=18, fontweight='bold')
        plt.setp(axes[i].get_yticklabels(), fontsize=18, fontweight='bold')
        plt.xticks([250, 500, 750, 1000, 1250, 1500, 2000], rotation=45)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.figlegend(by_label.values(), by_label.keys(), loc='upper center', \
                  bbox_to_anchor=(0.5, 1.15), ncol=3, prop={'weight': 'bold', 'size': 20})
    plt.tight_layout()
    plt.savefig('../result/figures/lik_ratio_test.pdf', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    if args.figure == 1:
        gen_figure1()
    elif args.figure == 2:
        gen_figure2()



