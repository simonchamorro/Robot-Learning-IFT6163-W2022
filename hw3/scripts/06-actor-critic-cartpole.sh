
# Experiment 6

python run_hw3.py env_name=CartPole-v0 exp_name=q6_ac_1_1 rl_alg=ac
python run_hw3.py env_name=CartPole-v0 exp_name=q6_100_1  rl_alg=ac num_target_updates=100 num_grad_steps_per_target_update=1
python run_hw3.py env_name=CartPole-v0 exp_name=q6_1_100  rl_alg=ac num_target_updates=1   num_grad_steps_per_target_update=100
python run_hw3.py env_name=CartPole-v0 exp_name=q6_10_10  rl_alg=ac num_target_updates=10  num_grad_steps_per_target_update=10


