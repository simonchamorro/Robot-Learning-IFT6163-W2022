
import os
import glob
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt

logdir = './run_logs/'
plot_dir = './plots/'

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


def plot_results(data, title, save, label_colors, 
                 x_axis=(None, 'Iteration Number'), 
                 y_axis=('Eval_AverageReturn', 'Average Reward'), 
                 y_std='Eval_StdReturn'):
    fig = plt.figure()
    i = 0
    for k, v in data.items():
        x = np.arange(len(v[y_axis[0]])) if x_axis[0] is None else np.array(v[x_axis[0]])
        y = np.array(v[y_axis[0]])
        std = np.zeros_like(y) if y_std is None else np.array(v[y_std])
        colors = None
        label = None
        color = 'C' + str(i)

        if label_colors: 
            label, colors = label_colors(k)
        
        if colors:
            color=colors[label]
        
        plt.plot(x, y, label=label, color=color)
        plt.fill_between(x, y-std, y+std, alpha=0.3, color=color)
        i += 1

    plt.ylabel(y_axis[1], fontsize=10)
    plt.xlabel(x_axis[1], fontsize=10)
    plt.title(title, fontsize=15)
    plt.legend(loc='lower right')
    plt.savefig(plot_dir + save)


def plot_agg_results(data, title, save, label):
    fig = plt.figure()

    len_train = len(next(iter(data.values()))['Eval_AverageReturn'])
    all_runs = np.zeros((len(data.keys()), len_train)) 
    i = 0
    for k, v in data.items():
        all_runs[i,:] = np.array(v['Eval_AverageReturn'])
        i += 1
    x = np.arange(len(v['Eval_AverageReturn']))
    y = np.mean(all_runs, axis=0)
    std = np.std(all_runs, axis=0)

    plt.plot(x, y, label=label)
    plt.fill_between(x, y-std, y+std, alpha=0.3)

    plt.ylabel('Average Reward', fontsize=10)
    plt.xlabel('Iteration Number', fontsize=10)
    plt.title(title, fontsize=15)
    plt.legend(loc='lower right')
    plt.savefig(plot_dir + save)


def plot_graph(name, fig_title, save_to, label_colors=None, 
               x_axis=(None, 'Iteration Number'), 
               y_axis=('Eval_AverageReturn', 'Average Reward'), 
               y_std='Eval_StdReturn'):

    exps = glob.glob(logdir + name + '/**/event*', recursive=True)
    results = {}
    for exp in exps:
        results[exp.split('/')[2]] = get_eventfile_results(exp)
    plot_results(results, fig_title, save_to, label_colors, x_axis=x_axis, 
                 y_axis=y_axis, y_std=y_std)


def plot_agg(name, fig_title, save_to, label=None):
    exps = glob.glob(logdir + name + '/**/event*', recursive=True)
    results = {}
    for exp in exps:
        results[exp.split('/')[2]] = get_eventfile_results(exp)
    plot_agg_results(results, fig_title, save_to, label)


def q1(exp_name):
    label = exp_name.split('sb_')[-1].split('lb_')[-1].split('_Cart')[0]
    colors = {'rtg_dsa':'C0', 'no_rtg_dsa':'C1', 'rtg_na':'C2'}
    return label, colors

def q2(exp_name):
    label = exp_name.split('_r')[-1].split('lb_')[-1].split('_Inverted')[0]
    return label, None

def q3(exp_name):
    label = 'b=40000, lr=0.005'
    return label, None

def q4_search(exp_name):
    b = exp_name.split('_b')[-1].split('_lr')[0]
    lr = exp_name.split('_lr')[-1].split('_rtg')[0]
    label = 'b=' + b + ', lr=' + lr
    return label, None

def q4_final(exp_name):
    if '_rtg_nnbaseline' in exp_name:
        label = 'rtg + baseline'
    elif '_rtg' in exp_name:
        label = 'rtg'
    elif '_nnbaseline' in exp_name:
        label = 'baseline'
    else:
        label = 'vanilla'
    return label, None

def q5(exp_name):
    lambda_value = exp_name.split('_lambda')[-1].split('_Hopper')[0]
    label = 'lambda=' + lambda_value
    colors = {'lambda=0.0':'C0', 'lambda=0.95':'C1', 'lambda=0.99':'C2', 'lambda=1.0':'C3'}
    return label, colors

def q6(exp_name):
    ntu = exp_name.split('_Cart')[0].split('_')[-2]
    ngsptu = exp_name.split('_Cart')[0].split('_')[-1]
    label = 'ntu=' + ntu + ', ngsptu=' + ngsptu
    return label, None

def q8(exp_name):
    b = exp_name.split('_b')[-1].split('_')[0]
    label = 'b=' + b 
    colors = {'b=2000':'C0', 'b=5000':'C1'}
    return label, colors

def q9(exp_name):
    lr = exp_name.split('_lr')[-1].split('_')[0]
    step = exp_name.split('_step')[-1].split('_')[0]
    label = 'lr=' + lr + ', num_steps=' + step
    return label, None

def q10(exp_name):
    label = 'clipped' if 'clip' in exp_name else 'vanilla'
    return label, None

def q11(exp_name):
    proc = exp_name.split('_proc')[-1].split('_')[0]
    label = 'num_proc=' + proc
    return label, None

def get_time_info(file):
    with open(file) as f:
        lines = f.readlines()
    
    values = []
    for l in lines:
        if 'Data collection' in l:
            values.append(float(lines[0].split('time:')[-1]))

    return np.mean(values), np.std(values) 

def plot_time(name, fig_title, save_to):
    x = []
    times = []
    time_std = []
    exps = glob.glob(logdir + name + '/**/run_hw3.log', recursive=True)
    results = {}
    for exp in exps:
        exp_name = exp.split('/')[2]
        num_proc = exp_name.split('_proc')[-1].split('_')[0]
        x.append(int(num_proc))
        collect_time, collect_std = get_time_info(exp)
        times.append(collect_time)
        time_std.append(collect_std)

    x, times, time_std = zip(*sorted(zip(x, times, time_std)))

    fig = plt.figure()
    plt.errorbar(x, times, yerr=time_std, fmt='-o')
    plt.ylabel('Data Collection Time', fontsize=10)
    plt.xlabel('Number of Processes', fontsize=10)
    plt.title(fig_title, fontsize=15)
    plt.savefig(plot_dir + save_to)


if __name__ == '__main__':    
    # Q1
    plot_graph('hw3_q1_sb*', 'Small Batch Experiments', 'q1-01.png', q1)
    plot_graph('hw3_q1_lb*', 'Large Batch Experiments', 'q1-02.png', q1)

    # Q2
    plot_graph('hw3_q2_b256*', 'Batch size of 256', 'q2-0256.png', q2)
    plot_agg('hw3_q2_final*', 'Final Config Over 5 Seeds', 'q2-final.png', label='b=256, lr=0.01')

    # Q3    
    plot_graph('hw3_q3_*', 'Performance on Lunar Lander', 'q3-lunar-lander.png', q3)

    # Q4 
    plot_graph('hw3_q4_search*', 'Hyperparameter Search in HalfCheetah', 'q4-search.png', q4_search)
    plot_graph('hw3_q4_b*', 'Policy Gradient in HalfCheetah', 'q4-final.png', q4_final)
    
    # Q5
    plot_graph('hw3_q5_b*', 'GAE in Hopper', 'q5-normal.png', q5)
    plot_graph('hw3_q5_noisy*', 'GAE in Noisy Hopper', 'q5-noisy.png', q5)

    # Q6
    plot_graph('hw3_q6*', 'Actor-Critic HP Search in Cartpole', 'q6-search.png', q6)

    # Q7
    plot_graph('hw3_q7_10_10_Inverted*', 'Actor-Critic in InvertedPendulum', 'q7-pendulum.png')
    plot_graph('hw3_q7_10_10_Half*', 'Actor-Critic in HalfCheetah', 'q7-cheetah.png')

    # Q8
    plot_graph('hw3_q8*00_cheeta*', 'Dyna Agent in Cheetah', 'q8-dyna.png', q8)
    plot_graph('hw3_q8*all*', 'Dyna Agent in Cheetah (All data for critic training)', 'q8-dyna-all.png', q8)

    # Bonus - multistep pg
    plot_graph('hw3_q9*lr0.005*', 'Multi-Step PG in LunarLander', 'q9-005.png', q9)
    plot_graph('hw3_q9*lr0.001*', 'Multi-Step PG in LunarLander with Smaller LR', 'q9-001.png', q9)

    # Bonus - ac + clip
    plot_graph('hw3_q10*CartPole*', 'AC with Gradient Clipping PG in CartPole', 'q10-cartpole-rew.png', q10)
    plot_graph('hw3_q10*CartPole*', 'AC with Gradient Clipping PG in CartPole', 'q10-cartpole-loss.png', q10, 
                y_axis=('Actor_Loss','Actor Loss'), y_std=None)

    plot_graph('hw3_q10*InvertedPendulum*', 'AC with Gradient Clipping PG in InvertedPendulum', 'q10-pendulum-rew.png', q10)
    plot_graph('hw3_q10*InvertedPendulum*', 'AC with Gradient Clipping PG in InvertedPendulum', 'q10-pendulum-loss.png', q10, 
                y_axis=('Actor_Loss','Actor Loss'), y_std=None)

    # Bonus - parallel data collection
    plot_graph('hw3_q11*', 'Performance', 'q11-rew.png', q11)
    plot_time('hw3_q11*', 'Data Collection Time', 'q11-collect.png')
    plot_graph('hw3_q11*', 'Training Time', 'q11-train.png', q11, 
            y_axis=('TimeSinceStart','Training Time'), y_std=None)