import torch
import torch.nn as nn
from torch.nn import functional as F
from util import helpers as utl
from util.torchkit.constant import *
from util.torchkit.networks import FlattenMlp
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.ObsRepEncoder import ObsRepMLP
import copy
from util.actor import TanhGaussianPolicy


class POMDP_policy(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        token_dim,
        action_embedding_size,
        observ_embedding_size,
        reward_embedding_size,
        rnn_hidden_size,
        policy_layers,
        rnn_num_layers=1,
        qf=None,
        vf=None,
        device='cuda',
        max_steps=10000,
        deterministic=False,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(device)
        self.qf=qf
        self.vf=vf
        self.deterministic = deterministic


        ### Build Model
        ## 1. embed action, state, reward (Feed-forward layers first)



        self.observ_embedder = ObsRepMLP(
            obs_dim=obs_dim,
            token_dim=token_dim,
            emb_dim=observ_embedding_size,
            both=True
        )


        self.action_embedder = utl.FeatureExtractor(
            action_dim, action_embedding_size, F.relu
        )
        self.reward_embedder = utl.FeatureExtractor(1, reward_embedding_size, F.relu)

        ## 2. build RNN model
        rnn_input_size = (
            action_embedding_size + observ_embedding_size + reward_embedding_size
        )
        self.rnn_hidden_size = rnn_hidden_size

        self.num_layers = rnn_num_layers

        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.num_layers,
            batch_first=False,
            bias=True,
        )
        # never add activation after GRU cell, cuz the last operation of GRU is tanh

        # default gru initialization is uniform, not recommended
        # https://smerity.com/articles/2016/orthogonal_init.html orthogonal has eigenvalue = 1
        # to prevent grad explosion or vanishing
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)


        self.current_observ_embedder = utl.FeatureExtractor(
            obs_dim, observ_embedding_size, F.relu
        )

        ## 4. build policy
        self.policy = self.build_actor(
            input_size=self.rnn_hidden_size + observ_embedding_size,
            action_dim=self.action_dim,
            hidden_sizes=policy_layers,
        )

        self.optimizer = torch.optim.Adam(self.parameters())
        self.lr_schedule = CosineAnnealingLR(self.optimizer, max_steps)
        self.device = device
        self.to(device)

    def build_actor(self, input_size, action_dim, hidden_sizes, **kwargs):
        return TanhGaussianPolicy(
            obs_dim=input_size,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            **kwargs,
        )
    def _get_obs_embedding(self, observs):
            return self.observ_embedder(observs)


    def _get_shortcut_obs_embedding(self, observs):
            return self.current_observ_embedder(observs)

    def get_hidden_states(
        self, prev_actions, rewards, observs, initial_internal_state=None
    ):
        # all the input have the shape of (1 or T+1, B, *)
        # get embedding of initial transition
        input_a = self.action_embedder(prev_actions)
        input_r = self.reward_embedder(rewards)
        #input_s = self._get_obs_embedding(observs)
        inputs = torch.cat((input_a, input_r, observs), dim=-1)

        # feed into RNN: output (T+1, B, hidden_size)
        if initial_internal_state is None:  # initial_internal_state is zeros
            output, _ = self.rnn(inputs)
            return output
        else:  # useful for one-step rollout
            output, current_internal_state = self.rnn(inputs, initial_internal_state)
            return output, current_internal_state

    def forward(self, prev_actions, rewards, observs, mask_scheme=None):
        """
        For prev_actions a, rewards r, observs o: (T+1, B, dim)
                a[t] -> r[t], o[t]

        return current actions a' (T+1, B, dim) based on previous history

        """
        assert prev_actions.dim() == rewards.dim() == observs.dim() == 3
        assert prev_actions.shape[0] == rewards.shape[0] == observs.shape[0]
        obs_emb, curr_embed, mask = self.observ_embedder(observs, mask_scheme)

        ### 1. get hidden/belief states of the whole/sub trajectories, aligned with states
        # return the hidden states (T+1, B, dim)
        hidden_states = self.get_hidden_states(
            prev_actions=prev_actions, rewards=rewards, observs=obs_emb
        )

        # 2. another branch for current obs
        #curr_embed = self._get_shortcut_obs_embedding(observs)  # (T+1, B, dim)

        # 3. joint embed
        joint_embeds = torch.cat((hidden_states, curr_embed), dim=-1)  # (T+1, B, dim)

        # 4. Actor
        new_actions, _, _, log_probs, tanh_normal = self.policy(joint_embeds, return_log_prob=True)
        return new_actions, log_probs, tanh_normal  # (T+1, B, dim), (T+1, B, 1)


    @torch.no_grad()
    def get_initial_info(self):
        # here we assume batch_size = 1

        ## here we set the ndim = 2 for action and reward for compatibility
        ## here we set the ndim = 2 for action and reward for compatibility
        prev_action = torch.zeros((1, self.action_dim), device=self.device).float()
        reward = torch.zeros((1, 1), device=self.device).float()

        hidden_state = torch.zeros((self.num_layers, 1, self.rnn_hidden_size), device=self.device).float()



        return prev_action, reward, hidden_state

    @torch.no_grad()
    def forward_single(self, prev_internal_state, prev_action, reward, obs, deterministic, return_log_prob, mask_scheme):
        # for evaluation (not training), so no target actor, and T = 1
        # a function that generates action, works like a pytorch module

        # 1. get hidden state and current internal state
        ## NOTE: in T=1 step rollout (and RNN layers = 1), for GRU they are the same,
        # for LSTM, current_internal_state also includes cell state, i.e.
        # hidden state: (1, B, dim)
        # current_internal_state: (layers, B, dim) or ((layers, B, dim), (layers, B, dim))
        obs_emb, curr_embed, _ = self.observ_embedder(obs.reshape(1,1,-1), mask_scheme)

        hidden_state, current_internal_state = self.get_hidden_states(
            prev_actions=prev_action.reshape(1,1,-1),
            rewards=reward.reshape(1,1,-1),
            observs=obs_emb,
            initial_internal_state=prev_internal_state,
        )
        # 2. another branch for current obs
        #curr_embed = self._get_shortcut_obs_embedding(obs)  # (1, B, dim)

        # 3. joint embed
        joint_embeds = torch.cat((hidden_state, curr_embed), dim=-1)  # (1, B, dim)
        if joint_embeds.dim() == 3:
            joint_embeds = joint_embeds.squeeze(0)  # (B, dim)

        # 4. Actor head, generate action tuple
        action_tuple = self.policy(joint_embeds, False, deterministic, return_log_prob)


        return action_tuple, current_internal_state
    
    def update(self, prev_actions, rewards, observs,  mask_scheme=None):
        with torch.no_grad():
            target_q = self.q_target(observs[:-1], prev_actions[1:])
            v = self.vf(observs[:-1])
            adv = target_q - v
            exp_adv = torch.exp(3.0 * adv.detach()).clamp(max=100)

        new_actions, _, tanh_normal = self.forward(prev_actions[:-1], rewards[:-1], observs[:-1], mask_scheme)
        if not self.deterministic:
            bc_losses = -tanh_normal.log_prob(prev_actions[:-1]).sum(dim=-1)
        else:
            bc_losses = torch.sum((new_actions - prev_actions[:-1]) ** 2, dim=-1)

        policy_loss = torch.mean(exp_adv.squeeze(dim=-1) * bc_losses)
        self.optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        grad_norms = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0, norm_type=2)
        self.optimizer.step()
        self.lr_schedule.step()
        

        return {'loss/'+mask_scheme['name']: float(policy_loss.mean().detach().cpu())}
