from collections import OrderedDict

from .base_agent import BaseAgent
from ift6163.models.ff_model import FFModel
from ift6163.policies.MPC_policy import MPCPolicy
from ift6163.policies.MLP_policy import MLPPolicyPG, MLPPolicyAC
from ift6163.infrastructure.replay_buffer import ReplayBuffer
from ift6163.infrastructure.utils import *
from ift6163.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from ift6163.infrastructure import pytorch_util as ptu


class MBAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(MBAgent, self).__init__()

        self.env = env.unwrapped
        self.agent_params = agent_params
        self.ensemble_size = self.agent_params['ensemble_size']
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        self.dyn_models = []
        for i in range(self.ensemble_size):
            model = FFModel(
                self.agent_params['ac_dim'],
                self.agent_params['ob_dim'],
                self.agent_params['n_layers'],
                self.agent_params['size'],
                self.agent_params['learning_rate'],
            )
            self.dyn_models.append(model)

        # actor/policy
        self.actor = MLPPolicyAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline'],
        )
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):

        # training a MB agent refers to updating the predictive model using observed state transitions
        # NOTE: each model in the ensemble is trained on a different random batch of size batch_size
        losses = []
        num_data = ob_no.shape[0]
        num_data_per_ens = int(num_data / self.ensemble_size)

        for i in range(self.ensemble_size):

            # select which datapoints to use for this model of the ensemble
            # you might find the num_data_per_env variable defined above useful
            rand_indices = np.random.permutation(num_data)[:num_data_per_ens]
            observations = ob_no[rand_indices,:]
            actions = ac_na[rand_indices,:]
            next_observations = next_ob_no[rand_indices,:]

            # Use datapoints to update one of the dyn_models
            model =  self.dyn_models[i]
            log = model.update(observations, actions, next_observations,
                                self.data_statistics)
            loss = log['Training Loss']
            losses.append(loss)
            
        # TODO Pick a model at random
        # TODO Use that model to generate one additional next_ob_no for every state in ob_no (using the policy distribution) 
        # Hint: You may need the env to label the rewards
        # Hint: Keep things on policy
        # TODO add this generated data to the real data
        # TODO Perform a policy gradient update         
        # Hint: Should the critic be trained with this generated data? Try with and without and include your findings in the report.

        model_idx = np.random.randint(self.ensemble_size) 
        model = self.dyn_models[model_idx]

        observations = ptu.from_numpy(ob_no)
        act_distribution = self.actor(observations)
        new_act = ptu.to_numpy(act_distribution.sample())
        next_obs_pred =  model.get_prediction(ob_no, new_act, self.data_statistics)
        new_rew, new_terminals = self.env.get_reward(ob_no, new_act)

        # Add to existing data
        all_obs = np.concatenate((ob_no, ob_no))
        all_act = np.concatenate((ac_na, new_act))
        all_rew = np.concatenate((re_n, new_rew))
        all_next_ob = np.concatenate((next_ob_no, next_obs_pred))
        all_terminals = np.concatenate((terminal_n, new_terminals))

        # Train
        critic_loss = 0
        actor_loss = 0
        train_critic_with_new_data = self.agent_params['train_critic_with_new_data']
        for i in range(self.agent_params['num_critic_updates_per_agent_update']):
            if train_critic_with_new_data:
                critic_loss = self.critic.update(all_obs, all_act, all_next_ob, all_rew, all_terminals)
            else:
                critic_loss = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)

        advantage = self.estimate_advantage(all_obs, all_next_ob, all_rew, all_terminals)

        for i in range(self.agent_params['num_actor_updates_per_agent_update']):
            actor_loss = self.actor.update(all_obs, all_act, advantage)['Training Loss']   

        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_loss
        loss['FD_Loss'] = np.mean(losses)
        return loss

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        # TODO Implement the following pseudocode:
        # 1) query the critic with ob_no, to get V(s)
        # 2) query the critic with next_ob_no, to get V(s')
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)
        ob_no = ptu.from_numpy(ob_no)
        next_ob_no = ptu.from_numpy(next_ob_no)
        re_n = ptu.from_numpy(re_n)
        terminal_n = ptu.from_numpy(terminal_n)
        terminal_n = 1 - terminal_n

        values = self.critic(ob_no)
        values_next = self.critic(next_ob_no)
        q_values = re_n + self.gamma * values_next * terminal_n
        adv_n = ptu.to_numpy(q_values - values)

        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        return adv_n

    def add_to_replay_buffer(self, paths, add_sl_noise=False):

        # add data to replay buffer
        self.replay_buffer.add_rollouts(paths, noised=add_sl_noise)

        # get updated mean/std of the data in our replay buffer
        self.data_statistics = {
            'obs_mean': np.mean(self.replay_buffer.obs, axis=0),
            'obs_std': np.std(self.replay_buffer.obs, axis=0),
            'acs_mean': np.mean(self.replay_buffer.acs, axis=0),
            'acs_std': np.std(self.replay_buffer.acs, axis=0),
            'delta_mean': np.mean(
                self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0),
            'delta_std': np.std(
                self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0),
        }

        # update the actor's data_statistics too, so actor.get_action can be calculated correctly
        self.actor.data_statistics = self.data_statistics

    def sample(self, batch_size):
        # NOTE: sampling batch_size * ensemble_size,
        # so each model in our ensemble can get trained on batch_size data
        return self.replay_buffer.sample_random_data(
            batch_size * self.ensemble_size)

    def save(self, path):
        print("Save not implemented")
