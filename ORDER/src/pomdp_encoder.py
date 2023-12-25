import torch
import torch.nn as nn
from torch.nn import functional as F
from util import helpers as utl
from util.torchkit.constant import *
from util.torchkit.networks import FlattenMlp
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.ObsRepEncoder import ObsRepMLP

# a new calss named POMDP_policy
'''
I want to define a class named POMDP_policy, which is a subclass of nn.Module. you should modify 
the code in the class POMDP_encoder, and add a new class named POMDP_policy.
1) you should create a module called self.policy instead of self.encoder, and so, the output dim of the self.policy is self.action_dim
2) the forward function and the forward_single function should be modified accordingly.
'''


class POMDP_encoder(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        n_code,
        action_embedding_size,
        observ_embedding_size,
        reward_embedding_size,
        rnn_hidden_size,
        encoder_layers,
        rnn_num_layers=1,
        action_discretizer=None,
        reward_discretizer=None,
        obs_discretizer=None,
        max_steps=10000,
        device='cuda',
        token_dim=3,
        continuous=False,
        use_obs_as_labels=False,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_code = n_code
        self.obs_discretizer = obs_discretizer


        ### Build Model
        ## 1. embed action, state, reward (Feed-forward layers first)


        self.observ_embedder = ObsRepMLP(
            obs_dim=obs_dim,
            token_dim=token_dim,
            emb_dim=observ_embedding_size,
            both=True
        )


        if action_discretizer is not None:
            self.action_embedder = action_discretizer
        else:
            self.action_embedder = utl.FeatureExtractor(
                action_dim, action_embedding_size, F.relu
            )
        if reward_discretizer is not None:
            self.reward_embedder = reward_discretizer
        else:
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
            num_layers=rnn_num_layers,
            batch_first=False,
            bias=True,
        )

        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

        ## 3. build another obs+act branch
        shortcut_embedding_size = observ_embedding_size



        self.encoder = self.build_encoder(
            input_size=self.rnn_hidden_size + shortcut_embedding_size,
            hidden_sizes=encoder_layers,
            action_dim=action_dim,
            continuous=continuous,
        )

        self.continuous = continuous
        self.use_obs_as_labels = use_obs_as_labels
        if continuous:
            self.loss_fn = torch.nn.MSELoss(reduction='mean')
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.parameters())
        self.lr_schedule = CosineAnnealingLR(self.optimizer, max_steps)
        self.device = device
        self.to(device)

    def build_encoder(self, hidden_sizes, input_size=None, obs_dim=None, action_dim=None, continuous=False):

        if obs_dim is not None and action_dim is not None:
            input_size = obs_dim + action_dim
        if continuous:
            output_size = self.obs_dim
        else:
            output_size = self.obs_dim * self.n_code
        encoder = FlattenMlp(
            input_size=input_size, output_size=output_size, hidden_sizes=hidden_sizes
        )
        return encoder

    @torch.no_grad()
    def get_initial_info(self):
        # here we assume batch_size = 1

        ## here we set the ndim = 2 for action and reward for compatibility
        prev_action = torch.zeros((1, self.action_dim), device=self.device).float()
        reward = torch.zeros((1, 1), device=self.device).float()

        hidden_state = torch.zeros((self.num_layers, 1, self.rnn_hidden_size),device=self.device).float()

        internal_state = hidden_state

        return prev_action, reward, internal_state

    def get_hidden_states(self, prev_actions, rewards, obs_emb, initial_internal_state=None):
        # all the input have the shape of (1 or T+1, B, *)
        # get embedding of initial transition
        input_a = self.action_embedder(prev_actions)
        input_r = self.reward_embedder(rewards)
        #input_s = self._get_obs_embedding(observs)
        inputs = torch.cat((input_a, input_r, obs_emb), dim=-1)

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
        current_actions (or action probs for discrete actions) a': (T or T+1, B, dim)
                o[t] -> a'[t]
        NOTE: there is one timestep misalignment in prev_actions and current_actions
        """
        assert prev_actions.shape[0] == rewards.shape[0] == observs.shape[0]

        obs_emb, shortcut_obs_emb, mask = self.observ_embedder(observs, mask_scheme)

        hidden_states = self.get_hidden_states(
            prev_actions=prev_actions, rewards=rewards, obs_emb=obs_emb
        )

        # 3. joint embeds
        joint_embeds = torch.cat(
            (hidden_states, shortcut_obs_emb), dim=-1
        )  # (T+1, B, dim)

        # 4. logits
        if self.continuous:
            logits = self.encoder(joint_embeds)
            return logits, logits, mask
        else:
            logits = self.encoder(joint_embeds).reshape(joint_embeds.shape[0], joint_embeds.shape[1],
                                                        self.obs_dim, self.n_code)

            predicted_codes = logits.argmax(dim=-1).reshape(joint_embeds.shape[0], joint_embeds.shape[1],
                                                        self.obs_dim)


            return logits, predicted_codes, mask



    def update(self, prev_actions, rewards, observs, labels, mask_scheme=None):
        labels = labels.unsqueeze(dim=-1)
        logits, predicted_codes, mask = self.forward(prev_actions, rewards, observs, mask_scheme)

        if self.continuous:
            if self.use_obs_as_labels:
                labels = observs
            loss = self.loss_fn(logits[mask], labels[mask].float().reshape(-1))
        else:
            loss = self.loss_fn(logits[mask], labels[mask].long().reshape(-1))

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return {'loss/'+mask_scheme['name']: float(loss.mean().detach().cpu())}

    @torch.no_grad()
    def forward_single(self, prev_action, reward, obs, prev_internal_state=None, mask_scheme=None, v2=False):
        """
        For prev_action a, reward r, observ o: (1, 1, dim)
                a -> r, o
        current_action (or action probs for discrete actions) a': (1 or 2, 1, dim)
                o -> a'
        """
        obs_emb, shortcut_obs_emb, mask = self.observ_embedder(obs.reshape(1,1,-1), mask_scheme)

        hidden_state, current_internal_state = self.get_hidden_states(
            prev_actions=prev_action.reshape(1,1,-1),
            rewards=reward.reshape(1,1,-1),
            obs_emb=obs_emb,
            initial_internal_state=prev_internal_state,
        )


        joint_embeds = torch.cat((hidden_state, shortcut_obs_emb), dim=-1)  # (1, 1, dim)



        if self.continuous:
            logits = self.encoder(joint_embeds).reshape(1, 1, self.obs_dim)
            predicted_codes = logits
        else:
            logits = self.encoder(joint_embeds).reshape(1, 1, self.obs_dim, self.n_code)
            predicted_codes = logits.argmax(dim=-1).reshape(1, 1, self.obs_dim)

        if v2:
           result, predicted_codes  = self.obs_discretizer.get_predicted_emb(obs, predicted_codes, mask)
        else:
            if self.use_obs_as_labels:
                true_codes = obs.reshape(predicted_codes.shape)
            else:
                true_codes = self.obs_discretizer.encode(obs).reshape(predicted_codes.shape)
            true_codes[mask] = predicted_codes[mask].float()
            result = true_codes

        return result, current_internal_state




