
# Experiment 1: DQN
# Pacman
python run_hw4.py env_name=MsPacman-v0 n_iter=1000000 exp_name=q1_1 seed=1

python run_hw4.py env_name=MsPacman-v0 n_iter=1000000 exp_name=q2_dqn_1 seed=1
python run_hw4.py env_name=MsPacman-v0 n_iter=1000000 exp_name=q2_dqn_2 seed=2
python run_hw4.py env_name=MsPacman-v0 n_iter=1000000 exp_name=q2_dqn_3 seed=3

python run_hw4.py env_name=MsPacman-v0 n_iter=1000000 exp_name=q2_doubledqn_1 seed=1 double_q=true
python run_hw4.py env_name=MsPacman-v0 n_iter=1000000 exp_name=q2_doubledqn_2 seed=2 double_q=true
python run_hw4.py env_name=MsPacman-v0 n_iter=1000000 exp_name=q2_doubledqn_3 seed=3 double_q=true

# Lunar lander
python run_hw4.py env_name=LunarLander-v3 n_iter=500000 exp_name=q2_dqn_1 seed=1
python run_hw4.py env_name=LunarLander-v3 n_iter=500000 exp_name=q2_dqn_2 seed=2
python run_hw4.py env_name=LunarLander-v3 n_iter=500000 exp_name=q2_dqn_3 seed=3

python run_hw4.py env_name=LunarLander-v3 n_iter=500000 exp_name=q2_doubledqn_1 seed=1 double_q=true
python run_hw4.py env_name=LunarLander-v3 n_iter=500000 exp_name=q2_doubledqn_2 seed=2 double_q=true
python run_hw4.py env_name=LunarLander-v3 n_iter=500000 exp_name=q2_doubledqn_3 seed=3 double_q=true

# Hyper-parameter: Exploration rate
python run_hw4.py env_name=MsPacman-v0 n_iter=1000000 exp_name=q3_start_eps1.0 start_eps_dqn=1.0
python run_hw4.py env_name=MsPacman-v0 n_iter=1000000 exp_name=q3_start_eps0.5 start_eps_dqn=0.5
python run_hw4.py env_name=MsPacman-v0 n_iter=1000000 exp_name=q3_start_eps0.2 start_eps_dqn=0.2
python run_hw4.py env_name=MsPacman-v0 n_iter=1000000 exp_name=q3_start_eps0.1 start_eps_dqn=0.1

