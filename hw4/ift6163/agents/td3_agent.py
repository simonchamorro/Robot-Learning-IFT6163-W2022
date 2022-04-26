import numpy as np

from ift6163.infrastructure.replay_buffer import ReplayBuffer
from ift6163.policies.MLP_policy import MLPPolicyDeterministic
from ift6163.critics.td3_critic import TD3Critic
import copy

from ift6163.agents.ddpg_agent import DDPGAgent

class TD3Agent(DDPGAgent):
    def __init__(self, env, agent_params):

        super().__init__(env, agent_params)
        
        self.q_fun = TD3Critic(self.actor, 
                               agent_params, 
                               self.optimizer_spec)
        self.policy_freq = agent_params['td3_policy_freq']

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):

            # Fill in the call to the update function using the appropriate tensors
            log['QF Loss'] = self.q_fun.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)
            
            if self.t % self.policy_freq == 0:
                # Fill in the call to the update function using the appropriate tensors
                ## Hint the actor will need a copy of the q_net to maximize the Q-function
                log['Policy Loss'] = self.actor.update(ob_no, self.q_fun.q_net_1)

            # Update the target network periodically 
            # HINT: your critic already has this functionality implemented
            if self.num_param_updates % self.target_update_freq == 0:
                self.q_fun.update_target_network()

            self.num_param_updates += 1

        self.t += 1
        return log
        