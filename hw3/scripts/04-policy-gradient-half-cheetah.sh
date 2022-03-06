
# Experiment 4

# Hyper parameter search
python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 discount=0.95 size=32 reward_to_go=true nn_baseline=true batch_size=10000 learning_rate=0.005 exp_name=q4_search_b10000_lr0.005_rtg_nnbaseline rl_alg=reinforce
python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 discount=0.95 size=32 reward_to_go=true nn_baseline=true batch_size=30000 learning_rate=0.005 exp_name=q4_search_b30000_lr0.005_rtg_nnbaseline rl_alg=reinforce
python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 discount=0.95 size=32 reward_to_go=true nn_baseline=true batch_size=50000 learning_rate=0.005 exp_name=q4_search_b50000_lr0.005_rtg_nnbaseline rl_alg=reinforce

python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 discount=0.95 size=32 reward_to_go=true nn_baseline=true batch_size=10000 learning_rate=0.01 exp_name=q4_search_b10000_lr0.01_rtg_nnbaseline rl_alg=reinforce
python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 discount=0.95 size=32 reward_to_go=true nn_baseline=true batch_size=30000 learning_rate=0.01 exp_name=q4_search_b30000_lr0.01_rtg_nnbaseline rl_alg=reinforce
python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 discount=0.95 size=32 reward_to_go=true nn_baseline=true batch_size=50000 learning_rate=0.01 exp_name=q4_search_b50000_lr0.01_rtg_nnbaseline rl_alg=reinforce

python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 discount=0.95 size=32 reward_to_go=true nn_baseline=true batch_size=10000 learning_rate=0.02 exp_name=q4_search_b10000_lr0.02_rtg_nnbaseline rl_alg=reinforce
python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 discount=0.95 size=32 reward_to_go=true nn_baseline=true batch_size=30000 learning_rate=0.02 exp_name=q4_search_b30000_lr0.02_rtg_nnbaseline rl_alg=reinforce
python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 discount=0.95 size=32 reward_to_go=true nn_baseline=true batch_size=50000 learning_rate=0.02 exp_name=q4_search_b50000_lr0.02_rtg_nnbaseline rl_alg=reinforce

# Final parameters
python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 discount=0.95 size=32 batch_size=50000 learning_rate=0.02 exp_name=q4_b50000_lr0.02 rl_alg=reinforce
python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 discount=0.95 size=32 reward_to_go=true batch_size=50000 learning_rate=0.02 exp_name=q4_b50000_lr0.02_rtg rl_alg=reinforce
python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 discount=0.95 size=32 nn_baseline=true batch_size=50000 learning_rate=0.02 exp_name=q4_b50000_lr0.02_nnbaseline rl_alg=reinforce
python run_hw3.py env_name=HalfCheetah-v2 ep_len=150 discount=0.95 size=32 reward_to_go=true nn_baseline=true batch_size=50000 learning_rate=0.02 exp_name=q4_b50000_lr0.02_rtg_nnbaseline rl_alg=reinforce




