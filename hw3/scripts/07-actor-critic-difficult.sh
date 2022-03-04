
# Experiment 7

python run_hw3.py env_name=InvertedPendulum-v2 rl_alg=ac ep_len=1000 discount=0.95 batch_size=5000 learning_rate=0.01 exp_name=q7_10_10 num_target_updates=10  num_grad_steps_per_target_update=10
python run_hw3.py env_name=HalfCheetah-v2 rl_alg=ac ep_len=150 discount=0.9 n_iter=150 size=32 batch_size=30000 learning_rate=0.02 exp_name=q7_10_10 num_target_updates=10  num_grad_steps_per_target_update=10


