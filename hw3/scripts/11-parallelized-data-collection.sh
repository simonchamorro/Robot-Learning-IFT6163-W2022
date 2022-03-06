
# Experiment 11

python run_hw3.py env_name=CartPole-v0 n_iter=100 batch_size=10000 reward_to_go=true exp_name=q11_proc1 rl_alg=reinforce no_gpu=true num_proc_data_collection=1
python run_hw3.py env_name=CartPole-v0 n_iter=100 batch_size=10000 reward_to_go=true exp_name=q11_proc2 rl_alg=reinforce no_gpu=true num_proc_data_collection=2
python run_hw3.py env_name=CartPole-v0 n_iter=100 batch_size=10000 reward_to_go=true exp_name=q11_proc6 rl_alg=reinforce no_gpu=true num_proc_data_collection=6

