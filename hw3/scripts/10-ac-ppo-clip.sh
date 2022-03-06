
# Experiment 10

python run_hw3.py env_name=CartPole-v0 exp_name=q10_10_10  rl_alg=ac num_target_updates=10  num_grad_steps_per_target_update=10
python run_hw3.py env_name=CartPole-v0 exp_name=q10_10_10_clip  rl_alg=ac num_target_updates=10  num_grad_steps_per_target_update=10 clip_policy_loss=true

python run_hw3.py env_name=InvertedPendulum-v2 rl_alg=ac ep_len=1000 discount=0.95 batch_size=5000 learning_rate=0.01 exp_name=q10_10_10 num_target_updates=10  num_grad_steps_per_target_update=10
python run_hw3.py env_name=InvertedPendulum-v2 rl_alg=ac ep_len=1000 discount=0.95 batch_size=5000 learning_rate=0.01 exp_name=q10_10_10_clip num_target_updates=10  num_grad_steps_per_target_update=10 clip_policy_loss=true

