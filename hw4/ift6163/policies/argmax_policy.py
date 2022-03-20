
import torch
import numpy as np
from ift6163.infrastructure import pytorch_util as ptu


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs): 
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## Return the action that maxinmizes the Q-value 
        # at the current observation as the output
        observation = ptu.from_numpy(observation)
        q_val = self.critic.q_net(observation)
        action = np.argmax(ptu.to_numpy(q_val))
        return action