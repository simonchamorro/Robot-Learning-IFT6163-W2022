
# Experiment 8
python run_hw3.py exp_name=q8_cheetah_n500_arch1x32_b5000_critic_all_data env_name=cheetah-ift6163-v0 discount=0.95 size=32 learning_rate=0.01 batch_size=5000 training_batch_size_dyna=1024 rl_alg=dyna train_critic_with_new_data=true
python run_hw3.py exp_name=q8_cheetah_n500_arch1x32_b2000_critic_all_data env_name=cheetah-ift6163-v0 discount=0.95 size=32 learning_rate=0.01 batch_size=2000 training_batch_size_dyna=1024 rl_alg=dyna train_critic_with_new_data=true

python run_hw3.py exp_name=q8_cheetah_n500_arch1x32_b5000 env_name=cheetah-ift6163-v0 discount=0.95 size=32 learning_rate=0.01 batch_size=5000 training_batch_size_dyna=1024 rl_alg=dyna train_critic_with_new_data=false
python run_hw3.py exp_name=q8_cheetah_n500_arch1x32_b2000 env_name=cheetah-ift6163-v0 discount=0.95 size=32 learning_rate=0.01 batch_size=2000 training_batch_size_dyna=1024 rl_alg=dyna train_critic_with_new_data=false 


