
# Experiment 2

python run_hw3.py env_name=InvertedPendulum-v2 ep_len=1000 discount=0.9 reward_to_go=true batch_size=512 learning_rate=0.005 exp_name=q2_b512_r005 rl_alg=reinforce
python run_hw3.py env_name=InvertedPendulum-v2 ep_len=1000 discount=0.9 reward_to_go=true batch_size=1024 learning_rate=0.005 exp_name=q2_b1024_r005 rl_alg=reinforce
python run_hw3.py env_name=InvertedPendulum-v2 ep_len=1000 discount=0.9 reward_to_go=true batch_size=2048 learning_rate=0.005 exp_name=q2_b2048_r005 rl_alg=reinforce

python run_hw3.py env_name=InvertedPendulum-v2 ep_len=1000 discount=0.9 reward_to_go=true batch_size=512 learning_rate=0.01 exp_name=q2_b512_r01 rl_alg=reinforce
python run_hw3.py env_name=InvertedPendulum-v2 ep_len=1000 discount=0.9 reward_to_go=true batch_size=1024 learning_rate=0.01 exp_name=q2_b1024_r01 rl_alg=reinforce
python run_hw3.py env_name=InvertedPendulum-v2 ep_len=1000 discount=0.9 reward_to_go=true batch_size=2048 learning_rate=0.01 exp_name=q2_b2048_r01 rl_alg=reinforce

python run_hw3.py env_name=InvertedPendulum-v2 ep_len=1000 discount=0.9 reward_to_go=true batch_size=512 learning_rate=0.02 exp_name=q2_b512_r02 rl_alg=reinforce
python run_hw3.py env_name=InvertedPendulum-v2 ep_len=1000 discount=0.9 reward_to_go=true batch_size=1024 learning_rate=0.02 exp_name=q2_b1024_r02 rl_alg=reinforce
python run_hw3.py env_name=InvertedPendulum-v2 ep_len=1000 discount=0.9 reward_to_go=true batch_size=2048 learning_rate=0.02 exp_name=q2_b2048_r02 rl_alg=reinforce

