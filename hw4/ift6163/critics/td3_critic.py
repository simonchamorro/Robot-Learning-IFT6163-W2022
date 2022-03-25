from .base_critic import BaseCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn
import copy

from ift6163.infrastructure import pytorch_util as ptu
from ift6163.policies.MLP_policy import ConcatMLP


class TD3Critic(BaseCritic):

    def __init__(self, actor, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.env_name = hparams['env_name']
        self.ob_dim = hparams['ob_dim']
        self.learning_rate = hparams['critic_learning_rate']

        if isinstance(self.ob_dim, int):
            self.input_shape = (self.ob_dim,)
        else:
            self.input_shape = hparams['input_shape']

        self.ac_dim = hparams['ac_dim']
        self.grad_norm_clipping = hparams['grad_norm_clipping']
        self.gamma = hparams['gamma']

        self.optimizer_spec = optimizer_spec
        hparams = copy.deepcopy(hparams)
        hparams['ob_dim'] = hparams['ob_dim'] + hparams['ac_dim']
        hparams['ac_dim'] = 1
        self.q_net_1 = ConcatMLP(   
            hparams['ac_dim'],
            hparams['ob_dim'],
            hparams['n_layers_critic'],
            hparams['size_hidden_critic'],
            discrete=False,
            learning_rate=hparams['critic_learning_rate'],
            nn_baseline=False,
            deterministic=True
            )
        self.q_net_2 = ConcatMLP(   
            hparams['ac_dim'],
            hparams['ob_dim'],
            hparams['n_layers_critic'],
            hparams['size_hidden_critic'],
            discrete=False,
            learning_rate=hparams['critic_learning_rate'],
            nn_baseline=False,
            deterministic=True
            )
        self.q_net_target_1 = copy.deepcopy(self.q_net_1)
        self.q_net_target_2 = copy.deepcopy(self.q_net_2)

        # self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
        #     self.optimizer,
        #     self.optimizer_spec.learning_rate_schedule,
        # 
        self.optimizer = optim.Adam(
            list(self.q_net_1.parameters()) + list(self.q_net_2.parameters()),
            self.learning_rate,
            )
        self.loss = nn.SmoothL1Loss()  # AKA Huber loss
        self.q_net_1.to(ptu.device)
        self.q_net_2.to(ptu.device)
        self.q_net_target_1.to(ptu.device)
        self.q_net_target_2.to(ptu.device)
        self.actor = actor
        self.actor_target = copy.deepcopy(actor) 
        self.polyak_avg = hparams['polyak_avg']
        self.target_policy_noise = hparams['td3_target_policy_noise']

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
            Update the parameters of the critic.
            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories
            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                ac_na: length: sum_of_path_lengths. The action taken at the current step.
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end
            returns:
                nothing
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)

        q_t_values_1 = self.q_net_1(ob_no, ac_na).squeeze()
        q_t_values_2 = self.q_net_2(ob_no, ac_na).squeeze()
        
        # Compute the Q-values from the target network 
        ## Hint: you will need to use the target policy
        # Add noise to target
        noise = (torch.randn_like(ac_na, dtype=torch.float) * self.target_policy_noise)
        ac_next_target = self.actor_target(next_ob_no) + noise

        q_tp1_values_1 = self.q_net_target_1(next_ob_no, ac_next_target).squeeze()
        q_tp1_values_2 = self.q_net_target_2(next_ob_no, ac_next_target).squeeze()
        q_tp1_values = torch.min(q_tp1_values_1, q_tp1_values_2)

        # TODO compute targets for minimizing Bellman error
        # HINT: as you saw in lecture, this would be:
            #currentReward + self.gamma * qValuesOfNextTimestep * (not terminal)
        target = reward_n + self.gamma * q_tp1_values * (1 - terminal_n)
        target = target.detach()

        assert q_t_values_1.shape == target.shape
        loss = self.loss(q_t_values_1, target) + self.loss(q_t_values_2, target) 

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net_1.parameters(), self.grad_norm_clipping)
        utils.clip_grad_value_(self.q_net_2.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        return ptu.to_numpy(loss)

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target_1.parameters(), self.q_net_1.parameters()
        ):
            ## Perform Polyak averaging
            target_param.data.copy_(
                target_param.data * (1.0 - self.polyak_avg) + param.data * self.polyak_avg
            )
        for target_param, param in zip(
                self.q_net_target_2.parameters(), self.q_net_2.parameters()
        ):
            ## Perform Polyak averaging
            target_param.data.copy_(
                target_param.data * (1.0 - self.polyak_avg) + param.data * self.polyak_avg
            )
        for target_param, param in zip(
                self.actor_target.parameters(), self.actor.parameters()
        ):
            ## Perform Polyak averaging for the target policy
            target_param.data.copy_(
                target_param.data * (1.0 - self.polyak_avg) + param.data * self.polyak_avg
            )


