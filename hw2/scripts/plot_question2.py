
import os
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


if __name__ == '__main__':
    import glob

    logdir = './run_logs/'
    plot_dir = './plots/'
    
    # Identify question runs
    exps = glob.glob(logdir + 'hw2_q2*/**/event*', recursive=True)
    
    # Only one run for question 2
    results = get_eventfile_results(exps[0])

    fig = plt.figure()
    plt.errorbar(1, results['Eval_AverageReturn'], yerr=results['Eval_StdReturn'], fmt='-o', label='Eval')
    plt.errorbar(0, results['Train_AverageReturn'], yerr=results['Train_StdReturn'], fmt='-o', label='Train')
    plt.ylabel('Average Reward', fontsize=10)
    plt.xlabel('Iteration Number', fontsize=10)
    plt.title('Untrained vs Trained Dynamics Model', fontsize=15)
    plt.legend(loc='lower right')
    plt.savefig(plot_dir + 'q2.png')