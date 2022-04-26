
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


def smooth_data(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plot_results(data, title, save, label_colors, 
                 x_axis=(None, 'Iteration Number'), 
                 y_axis=('Eval_AverageReturn', 'Average Reward'), 
                 y_std='Eval_StdReturn',
                 smooth=False):
    fig = plt.figure()
    i = 0
    for k, v in data.items():
        x = np.arange(len(v[y_axis[0]])) if x_axis[0] is None else np.array(v[x_axis[0]])/2 
        # because num_agent_train_steps_per_iter=2
        y = np.array(v[y_axis[0]])
        y = np.where(y < -5000.0, np.nan, y)
        std = np.zeros_like(y) if y_std is None else np.array(v[y_std])
        if smooth:
            y = smooth_data(y, 5)
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
    if label:
        plt.legend(loc='lower right')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.grid()    
    plt.savefig(plot_dir + save)


def plot_graph(name, fig_title, save_to, label_colors=None, 
               x_axis=(None, 'Iteration Number'), 
               y_axis=('Eval_AverageReturn', 'Average Reward'), 
               y_std='Eval_StdReturn',
               smooth=False):

    exps = glob.glob(logdir + name + '/**/event*', recursive=True)
    results = {}
    for exp in exps:
        results[exp.split('/')[2]] = get_eventfile_results(exp)
    plot_results(results, fig_title, save_to, label_colors, x_axis=x_axis, 
                 y_axis=y_axis, y_std=y_std, smooth=smooth)


def plot_dqn_1(name, title, save):
    exps = glob.glob(logdir + name + '/**/event*', recursive=True)
    data = {}
    for exp in exps:
        data[exp.split('/')[2]] = get_eventfile_results(exp)

    for k, v in data.items():
        x = np.array(v['Train_EnvstepsSoFar'])/2 
        avg = np.array(v['Train_AverageReturn'])
        avg = np.where(avg < -5000.0, np.nan, avg)
        best = np.array(v['Train_BestReturn'])
        best = np.where(best < -5000.0, np.nan, best)
    
    plt.figure()
    plt.plot(x, avg, label='Avg')
    plt.plot(x, best, label='Best')
    plt.ylabel('Reward', fontsize=10)
    plt.xlabel('Env Steps', fontsize=10)
    plt.title(title, fontsize=15)
    plt.legend(loc='lower right')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.grid()    
    plt.savefig(plot_dir + save)


def plot_dqn_2(name, title, save):
    exps = glob.glob(logdir + name + '/**/event*', recursive=True)
    data = {}
    for exp in exps:
        data[exp.split('/')[2]] = get_eventfile_results(exp)

    i = 0
    j = 0
    len_exp = len(data[exp.split('/')[2]]['Train_EnvstepsSoFar'])
    dqn_avg = np.zeros((3,len_exp))
    dqn_best = np.zeros((3,len_exp))
    double_avg = np.zeros((3,len_exp))
    double_best = np.zeros((3,len_exp))
    for k, v in data.items():
        x = np.array(v['Train_EnvstepsSoFar'])/2 
        avg = np.array(v['Train_AverageReturn'])
        avg = np.where(avg < -5000.0, np.nan, avg)
        best = np.array(v['Train_BestReturn'])
        best = np.where(best < -5000.0, np.nan, best)
        if 'double' in k:
            dqn_avg[i,:] = avg
            dqn_best[i,:] = best
            i += 1
        else:
            double_avg[j,:] = avg
            double_best[j,:] = best
            j += 1
    dqn_avg = np.mean(dqn_avg, axis=0)
    dqn_best = np.mean(dqn_best, axis=0)
    double_avg = np.mean(double_avg, axis=0)
    double_best = np.mean(double_best, axis=0)

    plt.figure()
    plt.plot(x, dqn_avg, label='Vanilla Avg')
    plt.plot(x, dqn_best, label='Vanilla Best')
    plt.plot(x, double_avg, label='Double Avg')
    plt.plot(x, double_best, label='Double Best')
    plt.ylabel('Reward', fontsize=10)
    plt.xlabel('Env Steps', fontsize=10)
    plt.title(title, fontsize=15)
    plt.legend(loc='lower right')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.grid()    
    plt.savefig(plot_dir + save)


def q3(exp_name):
    eps = exp_name.split('_eps')[-1].split('_')[0]
    label = 'start eps=' + eps
    return label, None

def q4_lr(exp_name):
    lr = exp_name.split('_lr')[-1].split('_')[0]
    label = 'lr=' + lr
    return label, None

def q4_up(exp_name):
    lr = exp_name.split('_up')[-1].split('_')[0]
    label = 'update=' + lr
    return label, None

def q6_rho(exp_name):
    rho = exp_name.split('_rho')[-1].split('_')[0]
    label = 'rho=' + rho
    return label, None

def q6_shape(exp_name):
    shape = exp_name.split('_shape')[-1].split('_')[0]
    label = 'shape=' + shape
    return label, None

def q5_7(exp_name):
    if 'ddpg' in exp_name:
        label = 'DDPG'
    else:
        label = 'TD3'
    return label, None


if __name__ == '__main__':  
    
    # DQN
    # Q1
    plot_dqn_1('hw4_q1*', 'DQN on Pacman', 'q1.png')

    # Q2
    plot_dqn_2('hw4_q2*Lunar*', 'DQN vs Double DQN on LunarLander', 'q2-lander.png')
    plot_dqn_2('hw4_q2*Pac*', 'DQN vs Double DQN on Pacman', 'q2-pacman.png')

    # Q3
    plot_graph('hw4_q3*', 'DQN: Exploration Study', 'q3.png', q3, x_axis=('Train_EnvstepsSoFar', 'Env Steps'),
                                                                     y_axis=('Train_AverageReturn', 'Reward'),
                                                                     y_std=None)  
    
    # DDPG
    # Q4 
    plot_graph('hw4_q4_ddpg_lr*', 'DDPG: Learning Rate', 'q4-lr.png', q4_lr, x_axis=('Train_EnvstepsSoFar', 'Env Steps'))  
    plot_graph('hw4_q4_ddpg_up*', 'DDPG: Update Frequency', 'q4-up.png', q4_up, x_axis=('Train_EnvstepsSoFar', 'Env Steps'))  

    # Q5
    plot_graph('hw4_q5*', 'DDPG on HalfCheetah', 'q5.png', x_axis=('Train_EnvstepsSoFar', 'Env Steps'), smooth=True)    

    # TD3
    # Q6
    plot_graph('hw4_q6_td3_rho*', 'TD3: Traget Policy Noise', 'q6-rho.png', q6_rho, x_axis=('Train_EnvstepsSoFar', 'Env Steps'))  
    plot_graph('hw4_q6_td3_shape*', 'TD3: Q-Function Structure', 'q6-shape.png', q6_shape, x_axis=('Train_EnvstepsSoFar', 'Env Steps'))  

    # Q7
    plot_graph('hw4_q7*', 'TD3 on HalfCheetah', 'q7.png', x_axis=('Train_EnvstepsSoFar', 'Env Steps'), smooth=True) 

    # DDPG and TD3 Comparison
    plot_graph('*HalfCheetah*', 'DDPG vs TD3', 'q5-7.png', q5_7, x_axis=('Train_EnvstepsSoFar', 'Env Steps'), smooth=True)  
