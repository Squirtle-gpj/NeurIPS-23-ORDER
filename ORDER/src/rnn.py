
import torch
import torch.nn as nn
from util.util import mlp





import torch
import torch.nn as nn
from torch.nn import functional as F
from util import helpers as utl
from util.torchkit.constant import *
from util.torchkit.networks import FlattenMlp



class Twin_Q_RNN(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        action_embedding_size,
        observ_embedding_size,
        reward_embedding_size,
        rnn_hidden_size,
        dqn_layers,
        rnn_num_layers=1,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        ### Build Model
        ## 1. embed action, state, reward (Feed-forward layers first)


        self.observ_embedder = utl.FeatureExtractor(
            obs_dim, observ_embedding_size, F.relu
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
        shortcut_embedding_size = rnn_input_size

        self.current_shortcut_embedder = utl.FeatureExtractor(
            obs_dim + action_dim, shortcut_embedding_size, F.relu
        )
        ## 4. build q networks
        self.qf1, self.qf2 = self.build_critic(
            input_size=self.rnn_hidden_size + shortcut_embedding_size,
            hidden_sizes=dqn_layers,
            action_dim=action_dim,
        )




    def build_critic(self, hidden_sizes, input_size=None, obs_dim=None, action_dim=None):
        if obs_dim is not None and action_dim is not None:
            input_size = obs_dim + action_dim
        qf1 = FlattenMlp(
            input_size=input_size, output_size=1, hidden_sizes=hidden_sizes
        )
        qf2 = FlattenMlp(
            input_size=input_size, output_size=1, hidden_sizes=hidden_sizes
        )
        return qf1, qf2

    def _get_obs_embedding(self, observs):
            return self.observ_embedder(observs)

    def _get_shortcut_obs_act_embedding(self, observs, current_actions):

        # for vector-based continuous action problems
        return self.current_shortcut_embedder(
            torch.cat([observs, current_actions], dim=-1)
        )


    def get_hidden_states(self, prev_actions, rewards, observs):
        # all the input have the shape of (T+1, B, *)
        # get embedding of initial transition
        input_a = self.action_embedder(prev_actions)
        input_r = self.reward_embedder(rewards)
        input_s = self._get_obs_embedding(observs)
        inputs = torch.cat((input_a, input_r, input_s), dim=-1)

        # feed into RNN: output (T+1, B, hidden_size)
        output, _ = self.rnn(inputs)  # initial hidden state is zeros
        return output

    def both(self, prev_actions, rewards, observs, current_actions):
        """
        For prev_actions a, rewards r, observs o: (T+1, B, dim)
                a[t] -> r[t], o[t]
        current_actions (or action probs for discrete actions) a': (T or T+1, B, dim)
                o[t] -> a'[t]
        NOTE: there is one timestep misalignment in prev_actions and current_actions
        """
        assert (
            prev_actions.dim()
            == rewards.dim()
            == observs.dim()
            == current_actions.dim()
            == 3
        )
        assert prev_actions.shape[0] == rewards.shape[0] == observs.shape[0]

        ### 1. get hidden/belief states of the whole/sub trajectories, aligned with observs
        # return the hidden states (T+1, B, dim)
        hidden_states = self.get_hidden_states(
            prev_actions=prev_actions, rewards=rewards, observs=observs
        )

        # 2. another branch for state & **current** action
        if current_actions.shape[0] == observs.shape[0]:
            # current_actions include last obs's action, i.e. we have a'[T] in reaction to o[T]
            curr_embed = self._get_shortcut_obs_act_embedding(
                observs, current_actions
            )  # (T+1, B, dim)
            # 3. joint embeds
            joint_embeds = torch.cat(
                (hidden_states, curr_embed), dim=-1
            )  # (T+1, B, dim)
        else:
            # current_actions does NOT include last obs's action
            curr_embed = self._get_shortcut_obs_act_embedding(
                observs[:-1], current_actions
            )  # (T, B, dim)
            # 3. joint embeds
            joint_embeds = torch.cat(
                (hidden_states[:-1], curr_embed), dim=-1
            )  # (T, B, dim)

        # 4. q value
        q1 = self.qf1(joint_embeds)
        q2 = self.qf2(joint_embeds)

        return q1, q2  # (T or T+1, B, 1 or A)

    def forward(self, prev_actions, rewards, observs, current_actions):
        return torch.min(*self.both(prev_actions, rewards, observs, current_actions))

class V_RNN(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        action_embedding_size,
        observ_embedding_size,
        reward_embedding_size,
        rnn_hidden_size,
        dqn_layers,
        rnn_num_layers=1,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        ### Build Model
        ## 1. embed action, state, reward (Feed-forward layers first)


        self.observ_embedder = utl.FeatureExtractor(
            obs_dim, observ_embedding_size, F.relu
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
        shortcut_embedding_size = rnn_input_size

        self.current_shortcut_embedder = utl.FeatureExtractor(
            obs_dim, shortcut_embedding_size, F.relu
        )
        ## 4. build q networks
        self.v = self.build_critic(
            input_size=self.rnn_hidden_size + shortcut_embedding_size,
            hidden_sizes=dqn_layers,
        )




    def build_critic(self, hidden_sizes, input_size=None):
        v = FlattenMlp(
            input_size=input_size, output_size=1, hidden_sizes=hidden_sizes
        )

        return v

    def _get_obs_embedding(self, observs):
        return self.observ_embedder(observs)


    def _get_shortcut_obs_embedding(self, observs):

        # for vector-based continuous action problems
        return self.current_shortcut_embedder(
            observs
        )


    def get_hidden_states(self, prev_actions, rewards, observs):
        # all the input have the shape of (T+1, B, *)
        # get embedding of initial transition
        input_a = self.action_embedder(prev_actions)
        input_r = self.reward_embedder(rewards)
        input_s = self._get_obs_embedding(observs)
        inputs = torch.cat((input_a, input_r, input_s), dim=-1)

        # feed into RNN: output (T+1, B, hidden_size)
        output, _ = self.rnn(inputs)  # initial hidden state is zeros
        return output

    def forward(self, prev_actions, rewards, observs):
        """
        For prev_actions a, rewards r, observs o: (T+1, B, dim)
                a[t] -> r[t], o[t]
        current_actions (or action probs for discrete actions) a': (T or T+1, B, dim)
                o[t] -> a'[t]
        NOTE: there is one timestep misalignment in prev_actions and current_actions
        """
        assert (
            prev_actions.dim()
            == rewards.dim()
            == observs.dim()
            == 3
        )
        assert prev_actions.shape[0] == rewards.shape[0] == observs.shape[0]

        ### 1. get hidden/belief states of the whole/sub trajectories, aligned with observs
        # return the hidden states (T+1, B, dim)
        hidden_states = self.get_hidden_states(
            prev_actions=prev_actions, rewards=rewards, observs=observs
        )

        # 2. another branch for state & **current** action

            # current_actions include last obs's action, i.e. we have a'[T] in reaction to o[T]
        curr_embed = self._get_shortcut_obs_embedding(
            observs
        )  # (T+1, B, dim)
        # 3. joint embeds
        joint_embeds = torch.cat(
            (hidden_states, curr_embed), dim=-1
        )  # (T+1, B, dim)


        # 4. v value
        v = self.v(joint_embeds)

        return v  # (T or T+1, B, 1 or A)
