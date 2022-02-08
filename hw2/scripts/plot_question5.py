
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


def plot_results(data):
    fig = plt.figure()

    for k, v in data.items():
        if 'cem' in k:
            n_cem = k.split('_')[-1].split('_')[4]
            label = 'CEM: ' + n_cem 
        else:
            label = 'Random-Shooting'
        x = np.arange(len(v['Eval_AverageReturn']))
        plt.errorbar(x, v['Eval_AverageReturn'], yerr=v['Eval_StdReturn'], fmt='-o', label=label)
    
    plt.ylabel('Average Reward', fontsize=10)
    plt.xlabel('Iteration Number', fontsize=10)
    plt.title('CEM vs Random-Shooting', fontsize=15)
    plt.legend(loc='lower right')
    plt.savefig(plot_dir + 'q5.png')


if __name__ == '__main__':
    import glob

    logdir = './run_logs/'
    plot_dir = './plots/'
    
    # Ensemble effect
    exps = glob.glob(logdir + 'hw2_q5*/**/event*', recursive=True)
    results = {}
    for exp in exps:
        results[exp.split('/')[2]] = get_eventfile_results(exp)
    plot_results(results)
    