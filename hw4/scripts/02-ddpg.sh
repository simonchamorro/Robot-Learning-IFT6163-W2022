
# Experiment 2: DDPG

# Learning Rate
python run_hw4.py exp_name=q4_ddpg_up1_lr0.0005 rl_alg=ddpg n_iter=50000 env_name=InvertedPendulum-v2 atari=false learning_rate=0.0005
python run_hw4.py exp_name=q4_ddpg_up1_lr0.001  rl_alg=ddpg n_iter=50000 env_name=InvertedPendulum-v2 atari=false learning_rate=0.001
python run_hw4.py exp_name=q4_ddpg_up1_lr0.002  rl_alg=ddpg n_iter=50000 env_name=InvertedPendulum-v2 atari=false learning_rate=0.002

# Update Frequency
python run_hw4.py exp_name=q4_ddpg_up2_lr0.001 rl_alg=ddpg n_iter=50000 env_name=InvertedPendulum-v2 atari=false learning_freq=2
python run_hw4.py exp_name=q4_ddpg_up4_lr0.001 rl_alg=ddpg n_iter=50000 env_name=InvertedPendulum-v2 atari=false learning_freq=4

# HalfCheetah
python run_hw4.py exp_name=q5_ddpg_hard_up1_lr0.001 rl_alg=ddpg n_iter=300000 env_name=HalfCheetah-v2 atari=false 



