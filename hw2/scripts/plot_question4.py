
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_eventfile_results(file):
    """
        requires tensorflow==1.12.0
    """
    data = {}
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag in data.keys():
                data[v.tag].append(v.simple_value)
            else:
                data[v.tag] = [v.simple_value]
    return data


def plot_ensemble(data):
    fig = plt.figure()

    for k, v in data.items():
        x = np.arange(len(v['Eval_AverageReturn']))
        n_ensemble = k.split('ensemble')[-1].split('_')[0]
        plt.errorbar(x, v['Eval_AverageReturn'], yerr=v['Eval_StdReturn'], fmt='-o', label='Ensemble size: ' + n_ensemble)
    
    plt.ylabel('Average Reward', fontsize=10)
    plt.xlabel('Iteration Number', fontsize=10)
    plt.title('Effect of Ensemble Size', fontsize=15)
    plt.legend(loc='lower right')
    plt.savefig(plot_dir + 'q4_ensemble.png')

def plot_numseq(data):
    fig = plt.figure()

    for k, v in data.items():
        breakpoint()
        x = np.arange(len(v['Eval_AverageReturn']))
        n = k.split('numseq')[-1].split('_')[0]
        plt.errorbar(x, v['Eval_AverageReturn'], yerr=v['Eval_StdReturn'], fmt='-o', label='N Candidates: ' + n)
    
    plt.ylabel('Average Reward', fontsize=10)
    plt.xlabel('Iteration Number', fontsize=10)
    plt.title('Effect of Number of Candidate Actions', fontsize=15)
    plt.legend(loc='lower right')
    plt.savefig(plot_dir + 'q4_numseq.png')

def plot_horizon(data):
    fig = plt.figure()

    for k, v in data.items():
        x = np.arange(len(v['Eval_AverageReturn']))
        n = k.split('horizon')[-1].split('_')[0]
        plt.errorbar(x, v['Eval_AverageReturn'], yerr=v['Eval_StdReturn'], fmt='-o', label='Horizon: ' + n)
    
    plt.ylabel('Average Reward', fontsize=10)
    plt.xlabel('Iteration Number', fontsize=10)
    plt.title('Effect of Planning Horizon', fontsize=15)
    plt.legend(loc='lower right')
    plt.savefig(plot_dir + 'q4_horizon.png')


if __name__ == '__main__':
    import glob

    logdir = './run_logs/'
    plot_dir = './plots/'
    
    # Ensemble effect
    exps = glob.glob(logdir + 'hw2_q4*ensemble*/**/event*', recursive=True)
    results = {}
    for exp in exps:
        results[exp.split('/')[2]] = get_eventfile_results(exp)
    plot_ensemble(results)

    # N candidates effect
    exps = glob.glob(logdir + 'hw2_q4*numseq*/**/event*', recursive=True)
    results = {}
    for exp in exps:
        results[exp.split('/')[2]] = get_eventfile_results(exp)
    plot_numseq(results)

    # Horizon effect
    exps = glob.glob(logdir + 'hw2_q4*horizon*/**/event*', recursive=True)
    results = {}
    for exp in exps:
        results[exp.split('/')[2]] = get_eventfile_results(exp)
    plot_horizon(results)

    