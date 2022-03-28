
# Experiment 3: TD3

# Target Policy Noise
python run_hw4.py exp_name=q6_td3_shape2x64_rho0.0  rl_alg=ddpg n_iter=50000 env_name=InvertedPendulum-v2 atari=false td3_target_policy_noise=0.0
python run_hw4.py exp_name=q6_td3_shape2x64_rho0.05 rl_alg=ddpg n_iter=50000 env_name=InvertedPendulum-v2 atari=false td3_target_policy_noise=0.05
python run_hw4.py exp_name=q6_td3_shape2x64_rho0.1  rl_alg=ddpg n_iter=50000 env_name=InvertedPendulum-v2 atari=false td3_target_policy_noise=0.1
python run_hw4.py exp_name=q6_td3_shape2x64_rho0.2  rl_alg=ddpg n_iter=50000 env_name=InvertedPendulum-v2 atari=false td3_target_policy_noise=0.2

# Update Frequency
python run_hw4.py exp_name=q6_td3_shape1x32_rho0.1   rl_alg=ddpg n_iter=50000 env_name=InvertedPendulum-v2 atari=false n_layers=1 size=32
python run_hw4.py exp_name=q6_td3_shape2x32_rho0.1   rl_alg=ddpg n_iter=50000 env_name=InvertedPendulum-v2 atari=false n_layers=2 size=32
python run_hw4.py exp_name=q6_td3_shape2x128_rho0.1  rl_alg=ddpg n_iter=50000 env_name=InvertedPendulum-v2 atari=false n_layers=2 size=128

# HalfCheetah
python run_hw4.py exp_name=q7_td3_shape2x128_rho0.05 rl_alg=td3 n_iter=300000 env_name=HalfCheetah-v2 atari=false td3_target_policy_noise=0.05 n_layers=2 size=128




