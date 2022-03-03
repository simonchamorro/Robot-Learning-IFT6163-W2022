
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


def plot_results(data, title, i):
    fig = plt.figure()

    for k, v in data.items():
        x = np.arange(len(v['Eval_AverageReturn']))
        y = np.array(v['Eval_AverageReturn'])
        std = np.array(v['Eval_StdReturn'])
        lr = k.split('_r')[-1].split('lb_')[-1].split('_Inverted')[0]
        if lr == '02':
            label = 'lr: 0.02'
        elif lr == '01':
            label = 'lr: 0.01'
        else:
            label = 'lr: 0.005'
        plt.plot(x, y, label=label)
        plt.fill_between(x, y-std, y+std, alpha=0.3)

    plt.ylabel('Average Reward', fontsize=10)
    plt.xlabel('Iteration Number', fontsize=10)
    plt.title(title, fontsize=15)
    plt.legend(loc='lower right')
    plt.savefig(plot_dir + 'q2-' + str(i) + '.png')


if __name__ == '__main__':
    import glob

    logdir = './run_logs/'
    plot_dir = './plots/'
    
    # Batch 256
    exps = glob.glob(logdir + 'hw3_q2_b256*/**/event*', recursive=True)
    results = {}
    for exp in exps:
        results[exp.split('/')[2]] = get_eventfile_results(exp)
    plot_results(results, 'Batch size of 256', '0256')

    # TODO plot aggregated results for 256 and 0.01


    