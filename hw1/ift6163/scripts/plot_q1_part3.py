import csv 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(logdir, steps):

    avg_rewards = []
    std_rewards = []

    for step in steps:
        avg_reward_path = logdir + 'steps' + step + '-avg-reward.csv' 
        std_reward_path = logdir + 'steps' + step + '-std-reward.csv'

        with open(avg_reward_path, 'r') as file:
            csvreader = csv.reader(file)
            header = next(csvreader)
            for row in csvreader:
                avg_rewards.append(float(row[2]))

        with open(std_reward_path, 'r') as file:
            csvreader = csv.reader(file)
            header = next(csvreader)
            for row in csvreader:
                std_rewards.append(float(row[2]))

    return avg_rewards, std_rewards

def plot_data(env, expert, steps, avg_reward, std_reward, plot_dir):
    fig = plt.figure()
    plt.errorbar(steps, avg_reward, yerr=std_reward, fmt='-o', label='BC')
    plt.plot(steps, [expert]*len(steps), label='Expert')
    plt.xscale('log')
    plt.ylabel('Average Reward', fontsize=10)
    plt.xlabel('Number of Training Steps', fontsize=10)
    plt.title(env + ' Performance', fontsize=15)
    plt.legend(loc='lower right')
    plt.savefig(plot_dir + 'q1_3-' + env + '.png')


def main():
    env = 'Hopper'
    expert = 3772.67
    logdir = './run_logs/q1_part3_csv/'
    steps = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
    step_names = ['10', '100', '1k', '10k', '100k', '1M', '10M']
    plot_dir = './plots/'

    avg_reward, std_reward = load_data(logdir, step_names)
    plot_data(env, expert, steps, avg_reward, std_reward, plot_dir)


if __name__ == "__main__":
    main()
