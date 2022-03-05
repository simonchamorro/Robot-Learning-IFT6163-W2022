
# Experiment 9

#python run_hw3.py env_name=LunarLanderContinuous-v2 ep_len=1000 discount=0.99 reward_to_go=true nn_baseline=true batch_size=40000 learning_rate=0.005 exp_name=q9_step1 rl_alg=reinforce
#python run_hw3.py env_name=LunarLanderContinuous-v2 ep_len=1000 discount=0.99 reward_to_go=true nn_baseline=true batch_size=40000 learning_rate=0.005 exp_name=q9_step2 rl_alg=reinforce pg_steps=2
#python run_hw3.py env_name=LunarLanderContinuous-v2 ep_len=1000 discount=0.99 reward_to_go=true nn_baseline=true batch_size=40000 learning_rate=0.005 exp_name=q9_step5 rl_alg=reinforce pg_steps=5

python run_hw3.py env_name=LunarLanderContinuous-v2 ep_len=1000 discount=0.99 reward_to_go=true nn_baseline=true batch_size=40000 learning_rate=0.001 exp_name=q9_step1_lr0.001 rl_alg=reinforce
python run_hw3.py env_name=LunarLanderContinuous-v2 ep_len=1000 discount=0.99 reward_to_go=true nn_baseline=true batch_size=40000 learning_rate=0.001 exp_name=q9_step2_lr0.001 rl_alg=reinforce pg_steps=2
python run_hw3.py env_name=LunarLanderContinuous-v2 ep_len=1000 discount=0.99 reward_to_go=true nn_baseline=true batch_size=40000 learning_rate=0.001 exp_name=q9_step5_lr0.001 rl_alg=reinforce pg_steps=5


