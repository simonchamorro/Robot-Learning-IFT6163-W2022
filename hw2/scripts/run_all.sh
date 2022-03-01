
# Question 1
python run_hw2_mb.py exp_name=q1_cheetah_n500_arch1x32 env_name=cheetah-ift6163-v0 num_agent_train_steps_per_iter=500 n_layers=1 size=32

python run_hw2_mb.py exp_name=q1_cheetah_n5_arch2x250 env_name=cheetah-ift6163-v0 num_agent_train_steps_per_iter=5 n_layers=2 size=250

python run_hw2_mb.py exp_name=q1_cheetah_n500_arch2x250 env_name=cheetah-ift6163-v0 num_agent_train_steps_per_iter=500 n_layers=2 size=250


# Question 2

python run_hw2_mb.py exp_name=q2_obstacles_singleiteration env_name=obstacles-ift6163-v0 num_agent_train_steps_per_iter=20 batch_size_initial=5000 batch_size=1000 mpc_horizon=10 video_log_freq=-1


# Question 3

python run_hw2_mb.py exp_name=q3_obstacles env_name=obstacles-ift6163-v0 num_agent_train_steps_per_iter=20 batch_size_initial=5000 batch_size=1000 mpc_horizon=10 n_iter=12 video_log_freq=-1

python run_hw2_mb.py exp_name=q3_reacher env_name=reacher-ift6163-v0 mpc_horizon=10 num_agent_train_steps_per_iter=1000 batch_size_initial=5000 batch_size=5000 n_iter=15 video_log_freq=-1

python run_hw2_mb.py exp_name=q3_cheetah env_name=cheetah-ift6163-v0 mpc_horizon=15 num_agent_train_steps_per_iter=1500 batch_size_initial=5000 batch_size=5000 n_iter=20 video_log_freq=-1


# Question 4

python run_hw2_mb.py exp_name=q4_reacher_horizon5 env_name=reacher-ift6163-v0  mpc_horizon=5 mpc_action_sampling_strategy='random' num_agent_train_steps_per_iter=1000 batch_size=800 n_iter=15 video_log_freq=-1 mpc_action_sampling_strategy='random'

python run_hw2_mb.py exp_name=q4_reacher_horizon15 env_name=reacher-ift6163-v0 mpc_horizon=15 num_agent_train_steps_per_iter=1000 batch_size=800 n_iter=15 video_log_freq=-1 mpc_action_sampling_strategy='random'

python run_hw2_mb.py exp_name=q4_reacher_horizon30 env_name=reacher-ift6163-v0 mpc_horizon=30 num_agent_train_steps_per_iter=1000 batch_size=800 n_iter=15 video_log_freq=-1 mpc_action_sampling_strategy='random'

python run_hw2_mb.py exp_name=q4_reacher_numseq100 env_name=reacher-ift6163-v0 mpc_horizon=10 num_agent_train_steps_per_iter=1000 batch_size=800 n_iter=15 mpc_num_action_sequences=100 mpc_action_sampling_strategy='random'

python run_hw2_mb.py exp_name=q4_reacher_numseq1000 env_name=reacher-ift6163-v0 mpc_horizon=10 num_agent_train_steps_per_iter=1000 batch_size=800 n_iter=15 video_log_freq=-1 mpc_num_action_sequences=1000 mpc_action_sampling_strategy='random'

python run_hw2_mb.py exp_name=q4_reacher_ensemble1 env_name=reacher-ift6163-v0 ensemble_size=1 mpc_horizon=10 num_agent_train_steps_per_iter=1000 batch_size=800 n_iter=15 video_log_freq=-1 mpc_action_sampling_strategy='random'

python run_hw2_mb.py exp_name=q4_reacher_ensemble3 env_name=reacher-ift6163-v0 ensemble_size=3 mpc_horizon=10 num_agent_train_steps_per_iter=1000 batch_size=800 n_iter=15 video_log_freq=-1 mpc_action_sampling_strategy='random'

python run_hw2_mb.py exp_name=q4_reacher_ensemble5 env_name=reacher-ift6163-v0 ensemble_size=5 mpc_horizon=10 num_agent_train_steps_per_iter=1000 batch_size=800 n_iter=15 video_log_freq=-1 mpc_action_sampling_strategy='random'


# Question 5

python run_hw2_mb.py exp_name=q5_cheetah_random env_name='cheetah-ift6163-v0' mpc_horizon=15 num_agent_train_steps_per_iter=1500 batch_size_initial=5000 batch_size=5000 n_iter=5 video_log_freq=-1 mpc_action_sampling_strategy='random'

python run_hw2_mb.py exp_name=q5_cheetah_cem_2 env_name='cheetah-ift6163-v0' mpc_horizon=15 num_agent_train_steps_per_iter=1500 batch_size_initial=5000 batch_size=5000 n_iter=5 video_log_freq=-1 mpc_action_sampling_strategy='cem' cem_iterations=2

python run_hw2_mb.py exp_name=q5_cheetah_cem_4 env_name='cheetah-ift6163-v0' mpc_horizon=15 num_agent_train_steps_per_iter=1500 batch_size_initial=5000 batch_size=5000 n_iter=5 video_log_freq=-1 mpc_action_sampling_strategy='cem' cem_iterations=4