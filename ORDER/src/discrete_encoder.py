import numpy as np
import torch
import torch.optim as optim
import os

from dreamerv2.utils.module import get_parameters, FreezeParameters
from dreamerv2.utils.algorithm import compute_return

from dreamerv2.models.actor import DiscreteActionModel
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.rssm import RSSM
from dreamerv2.models.pixel import ObsDecoder, ObsEncoder
from dreamerv2.utils.buffer import TransitionBuffer


class DiscreteEncoder(object):
    def __init__(
            self,
            config,
            device,
    ):
        self.device = device
        self.config = config
        self.action_size = config.action_size
        self.pixel = config.pixel
        self.kl_info = config.kl
        #self.seq_len = config.seq_len
        self.batch_size = config.batch_size
        #self.collect_intervals = config.collect_intervals
        #self.seed_steps = config.seed_steps
        self.discount = config.discount_
        #self.lambda_ = config.lambda_
        #self.horizon = config.horizon
        self.loss_scale = config.loss_scale
        #self.actor_entropy_scale = config.actor_entropy_scale
        self.grad_clip_norm = config.grad_clip
        self.use_prev_rewards = config.use_prev_rewards

        self._model_initialize(config)
        self._optim_initialize(config)


    def get_state_from_obs(self, obs, prev_action, done, prev_rssmstate, prev_reward):
        embed = self.ObsEncoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device))
        if self.use_prev_rewards:
            _, posterior_rssm_state = self.RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate, prev_reward)
        else:
            _, posterior_rssm_state = self.RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate, None)
        model_state = self.RSSM.get_model_state(posterior_rssm_state)

        return model_state, posterior_rssm_state

    def train_batch(self, obs, actions, rewards, terms):
        """
        trains the world model and imagination actor and critic for collect_interval times using sequence-batch data from buffer
        """
        train_metrics = {}

        obs_l = []
        model_l = []
        reward_l = []
        prior_ent_l = []
        post_ent_l = []
        kl_l = []
        pcont_l = []



        #obs, actions, rewards, terms = self.buffer.sample()
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)  # t, t+seq_len
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)  # t-1, t+seq_len-1
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device) # t-1 to t+seq_len-1
        nonterms = 1 - torch.tensor(terms, dtype=torch.float32).to(self.device)  # t-1 to t+seq_len-1

        model_loss, kl_loss, obs_loss, reward_loss, pcont_loss, prior_dist, post_dist, posterior = self.representation_loss(
            obs, actions, rewards, nonterms)

        self.model_optimizer.zero_grad()
        model_loss.backward()
        grad_norm_model = torch.nn.utils.clip_grad_norm_(get_parameters(self.world_list), self.grad_clip_norm)
        self.model_optimizer.step()


        with torch.no_grad():
            prior_ent = torch.mean(prior_dist.entropy())
            post_ent = torch.mean(post_dist.entropy())

        prior_ent_l.append(prior_ent.item())
        post_ent_l.append(post_ent.item())
        obs_l.append(obs_loss.item())
        model_l.append(model_loss.item())
        reward_l.append(reward_loss.item())
        kl_l.append(kl_loss.item())
        pcont_l.append(pcont_loss.item())


        train_metrics['model_loss'] = np.mean(model_l)
        train_metrics['kl_loss'] = np.mean(kl_l)
        train_metrics['reward_loss'] = np.mean(reward_l)
        train_metrics['obs_loss'] = np.mean(obs_l)
        train_metrics['prior_entropy'] = np.mean(prior_ent_l)
        train_metrics['posterior_entropy'] = np.mean(post_ent_l)
        train_metrics['pcont_loss'] = np.mean(pcont_l)

        #with torch.no_grad():
            #batched_posterior = self.RSSM.rssm_detach(
                #self.RSSM.rssm_seq_to_batch(posterior, self.batch_size, self.seq_len - 1))

        with FreezeParameters(self.world_list):
            rssm_states = self.RSSM.get_model_state(self.RSSM.rssm_detach(posterior))



        return train_metrics, rssm_states



    def representation_loss(self, obs, actions, rewards, nonterms):

        embed = self.ObsEncoder(obs)  # t to t+seq_len
        prev_rssm_state = self.RSSM._init_rssm_state(self.batch_size)
        if self.use_prev_rewards:
            prior, posterior = self.RSSM.rollout_observation( embed, actions, nonterms, prev_rssm_state, None)
        else:
            prior, posterior = self.RSSM.rollout_observation( embed, actions, nonterms, prev_rssm_state)
        post_modelstate = self.RSSM.get_model_state(posterior)  # t to t+seq_len
        obs_dist = self.ObsDecoder(post_modelstate[:-1])  # t to t+seq_len-1
        reward_dist = self.RewardDecoder(post_modelstate[:-1])  # t to t+seq_len-1
        pcont_dist = self.DiscountModel(post_modelstate[:-1])  # t to t+seq_len-1

        obs_loss = self._obs_loss(obs_dist, obs[:-1])
        reward_loss = self._reward_loss(reward_dist, rewards[1:])
        pcont_loss = self._pcont_loss(pcont_dist, nonterms[1:])
        prior_dist, post_dist, div = self._kl_loss(prior, posterior)

        model_loss = self.loss_scale['kl'] * div + reward_loss + obs_loss + self.loss_scale['discount'] * pcont_loss
        return model_loss, div, obs_loss, reward_loss, pcont_loss, prior_dist, post_dist, posterior


    def _obs_loss(self, obs_dist, obs):
        obs_loss = -torch.mean(obs_dist.log_prob(obs))
        return obs_loss

    def _kl_loss(self, prior, posterior):
        prior_dist = self.RSSM.get_dist(prior)
        post_dist = self.RSSM.get_dist(posterior)
        if self.kl_info['use_kl_balance']:
            alpha = self.kl_info['kl_balance_scale']
            kl_lhs = torch.mean(
                torch.distributions.kl.kl_divergence(self.RSSM.get_dist(self.RSSM.rssm_detach(posterior)), prior_dist))
            kl_rhs = torch.mean(
                torch.distributions.kl.kl_divergence(post_dist, self.RSSM.get_dist(self.RSSM.rssm_detach(prior))))
            if self.kl_info['use_free_nats']:
                free_nats = self.kl_info['free_nats']
                kl_lhs = torch.max(kl_lhs, kl_lhs.new_full(kl_lhs.size(), free_nats))
                kl_rhs = torch.max(kl_rhs, kl_rhs.new_full(kl_rhs.size(), free_nats))
            kl_loss = alpha * kl_lhs + (1 - alpha) * kl_rhs

        else:
            kl_loss = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
            if self.kl_info['use_free_nats']:
                free_nats = self.kl_info['free_nats']
                kl_loss = torch.max(kl_loss, kl_loss.new_full(kl_loss.size(), free_nats))
        return prior_dist, post_dist, kl_loss

    def _reward_loss(self, reward_dist, rewards):
        reward_loss = -torch.mean(reward_dist.log_prob(rewards))
        return reward_loss

    def _pcont_loss(self, pcont_dist, nonterms):
        pcont_target = nonterms.float()
        pcont_loss = -torch.mean(pcont_dist.log_prob(pcont_target))
        return pcont_loss



    def save_model(self, iter):
        save_dict = self.get_save_dict()
        model_dir = self.config.model_dir
        save_path = os.path.join(model_dir, 'models_%d.pth' % iter)
        torch.save(save_dict, save_path)

    def get_save_dict(self):
        return {
            "RSSM": self.RSSM.state_dict(),
            "ObsEncoder": self.ObsEncoder.state_dict(),
            "ObsDecoder": self.ObsDecoder.state_dict(),
            "RewardDecoder": self.RewardDecoder.state_dict(),
            "DiscountModel": self.DiscountModel.state_dict(),
        }

    def load_save_dict(self, saved_dict):
        self.RSSM.load_state_dict(saved_dict["RSSM"])
        self.ObsEncoder.load_state_dict(saved_dict["ObsEncoder"])
        self.ObsDecoder.load_state_dict(saved_dict["ObsDecoder"])
        self.RewardDecoder.load_state_dict(saved_dict["RewardDecoder"])
        self.DiscountModel.load_state_dict(saved_dict['DiscountModel'])

    def _model_initialize(self, config):
        obs_shape = config.obs_shape
        action_size = config.action_size
        deter_size = config.rssm_info['deter_size']
        if config.rssm_type == 'continuous':
            stoch_size = config.rssm_info['stoch_size']
        elif config.rssm_type == 'discrete':
            category_size = config.rssm_info['category_size']
            class_size = config.rssm_info['class_size']
            stoch_size = category_size * class_size

        embedding_size = config.embedding_size
        rssm_node_size = config.rssm_node_size
        modelstate_size = stoch_size + deter_size

        self.embedding_size = embedding_size
        self.rssm_node_size = rssm_node_size
        self.modelstate_size = modelstate_size


        self.RSSM = RSSM(action_size, rssm_node_size, embedding_size, self.device, config.rssm_type,
                         config.rssm_info).to(self.device)
        self.RewardDecoder = DenseModel((1,), modelstate_size, config.reward).to(self.device)

        if config.discount_['use']:
            self.DiscountModel = DenseModel((1,), modelstate_size, config.discount_).to(self.device)
        if config.pixel:
            self.ObsEncoder = ObsEncoder(obs_shape, embedding_size, config.obs_encoder).to(self.device)
            self.ObsDecoder = ObsDecoder(obs_shape, modelstate_size, config.obs_decoder).to(self.device)
        else:
            self.ObsEncoder = DenseModel((embedding_size,), obs_shape, config.obs_encoder).to(self.device)
            self.ObsDecoder = DenseModel((obs_shape,), modelstate_size, config.obs_decoder).to(self.device)

    def _optim_initialize(self, config):
        model_lr = config.lr['model']
        self.world_list = [self.ObsEncoder, self.RSSM, self.RewardDecoder, self.ObsDecoder, self.DiscountModel]
        self.model_optimizer = optim.Adam(get_parameters(self.world_list), model_lr)


    def _print_summary(self):
        print('\n Obs encoder: \n', self.ObsEncoder)
        print('\n RSSM model: \n', self.RSSM)
        print('\n Reward decoder: \n', self.RewardDecoder)
        print('\n Obs decoder: \n', self.ObsDecoder)
        if self.config.discount_['use']:
            print('\n Discount decoder: \n', self.DiscountModel)
