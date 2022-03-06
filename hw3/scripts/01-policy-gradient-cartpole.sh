
# Experiment 1

# Small batch
python run_hw3.py env_name=CartPole-v0 n_iter=100 batch_size=1000 dont_standardize_advantages=true exp_name=q1_sb_no_rtg_dsa rl_alg=reinforce

python run_hw3.py env_name=CartPole-v0 n_iter=100 batch_size=1000 dont_standardize_advantages=true reward_to_go=true exp_name=q1_sb_rtg_dsa rl_alg=reinforce

python run_hw3.py env_name=CartPole-v0 n_iter=100 batch_size=1000 reward_to_go=true exp_name=q1_sb_rtg_na rl_alg=reinforce

# Large batch
python run_hw3.py env_name=CartPole-v0 n_iter=100 batch_size=5000 dont_standardize_advantages=true exp_name=q1_lb_no_rtg_dsa rl_alg=reinforce

python run_hw3.py env_name=CartPole-v0 n_iter=100 batch_size=5000 dont_standardize_advantages=true reward_to_go=true exp_name=q1_lb_rtg_dsa rl_alg=reinforce

python run_hw3.py env_name=CartPole-v0 n_iter=100 batch_size=5000 reward_to_go=true exp_name=q1_lb_rtg_na rl_alg=reinforce
