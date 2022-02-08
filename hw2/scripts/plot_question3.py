
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

def plot_performance(exp, data):
    fig = plt.figure()
    x = np.arange(len(data['Eval_AverageReturn']))
    plt.errorbar(x, data['Eval_AverageReturn'], yerr=data['Eval_StdReturn'], fmt='-o', label='Eval Performance')
    plt.ylabel('Average Reward', fontsize=10)
    plt.xlabel('Iteration Number', fontsize=10)
    plt.title('Model-Based RL Performance: ' + exp.split('_')[2], fontsize=15)
    plt.legend(loc='lower right')
    plt.savefig(plot_dir + 'q3_' + exp.split('_')[2] + '.png')


if __name__ == '__main__':
    import glob

    logdir = './run_logs/'
    plot_dir = './plots/'
    
    # Identify question runs
    exps = glob.glob(logdir + 'hw2_q3*/**/event*', recursive=True)
    
    results = {}
    for exp in exps:
        results[exp.split('/')[2]] = get_eventfile_results(exp)

    for k, v in results.items():
        plot_performance(k, v)

    