
# Experiment 5

# TODO action noise = 0.5

python run_hw3.py env_name=Hopper-v2 ep_len=1000 discount=0.99 n_iter=300 size=32 reward_to_go=true nn_baseline=true batch_size=2000 learning_rate=0.001 exp_name=q5_b2000_r0.001_lambda0.0 gae_lambda=0.0 rl_alg=reinforce
python run_hw3.py env_name=Hopper-v2 ep_len=1000 discount=0.99 n_iter=300 size=32 reward_to_go=true nn_baseline=true batch_size=2000 learning_rate=0.001 exp_name=q5_b2000_r0.001_lambda0.95 gae_lambda=0.95 rl_alg=reinforce
python run_hw3.py env_name=Hopper-v2 ep_len=1000 discount=0.99 n_iter=300 size=32 reward_to_go=true nn_baseline=true batch_size=2000 learning_rate=0.001 exp_name=q5_b2000_r0.001_lambda0.99 gae_lambda=0.99 rl_alg=reinforce
python run_hw3.py env_name=Hopper-v2 ep_len=1000 discount=0.99 n_iter=300 size=32 reward_to_go=true nn_baseline=true batch_size=2000 learning_rate=0.001 exp_name=q5_b2000_r0.001_lambda1.0 gae_lambda=1.0 rl_alg=reinforce




