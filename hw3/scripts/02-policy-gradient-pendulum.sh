
# Experiment 2

python run_hw3.py --multirun env_name=InvertedPendulum-v2 ep_len=1000 discount=0.9 n_iter=100 size=64 batch_size=256,512,1024,4096 learning_rate=0.005,0.01,0.02 reward_to_go=true exp_name=q2_b<batch_size>_r<learning_rate> rl_alg=reinforce 
