import csv 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(logdir, env):
    avg_reward_path = logdir + 'Q2-' + env + '-Eval_AverageReturn.csv' 
    std_reward_path = logdir + 'Q2-' + env + '-Eval_StdReturn.csv'

    itrs = []
    avg_reward = []
    std_reward = []
    with open(avg_reward_path, 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            itrs.append(int(row[1]))
            avg_reward.append(float(row[2]))

    with open(std_reward_path, 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            std_reward.append(float(row[2]))

    return itrs, avg_reward, std_reward

def plot_data(env, expert, itr, avg_reward, std_reward, plot_dir):
    fig = plt.figure()
    plt.errorbar(itr, avg_reward, yerr=std_reward, fmt='-o', label='DAgger')
    plt.plot(itr, [avg_reward[0]]*len(itr), label='BC')
    plt.plot(itr, [expert]*len(itr), label='Expert')
    plt.ylabel('Average Reward', fontsize=10)
    plt.xlabel('Iteration Number', fontsize=10)
    plt.title('DAgger in'+ env, fontsize=15)
    plt.legend(loc='lower right')
    plt.savefig(plot_dir + 'q2-' + env + '.png')


def main():
    envs = ['Ant', 'Hopper']
    expert = [4713.65, 3772.67]
    logdir = './run_logs/'
    plot_dir = './plots/'

    for i in range(len(envs)):
        itr, avg_reward, std_reward = load_data(logdir, envs[i])
        plot_data(envs[i], expert[i], itr, avg_reward, std_reward, plot_dir)


if __name__ == "__main__":
    main()
