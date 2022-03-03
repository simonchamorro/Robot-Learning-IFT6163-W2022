
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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


def plot_results(data, title, name):
    fig = plt.figure()

    for k, v in data.items():
        x = np.arange(len(v['Eval_AverageReturn']))
        y = np.array(v['Eval_AverageReturn'])
        std = np.array(v['Eval_StdReturn'])
        b = k.split('_b')[-1].split('_lr')[0]
        lr = k.split('_lr')[-1].split('_rtg')[0]
        label = 'b=' + b + ', lr=' + lr
        plt.plot(x, y, label=label)
        plt.fill_between(x, y-std, y+std, alpha=0.3)

    plt.ylabel('Average Reward', fontsize=10)
    plt.xlabel('Iteration Number', fontsize=10)
    plt.title(title, fontsize=15)
    plt.legend(loc='lower right')
    plt.savefig(plot_dir + name)


if __name__ == '__main__':
    import glob

    logdir = './run_logs/'
    plot_dir = './plots/'
    
    # HP search
    exps = glob.glob(logdir + 'hw3_q4_search*/**/event*', recursive=True)
    results = {}
    for exp in exps:
        results[exp.split('/')[2]] = get_eventfile_results(exp)
    plot_results(results, 'Hyperparameter Search in HalfCheetah', 'q4-search.png')

    # TODO plot results for final HPs


    