import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from util.util import DEFAULT_DEVICE, update_exponential_moving_average
from util import exp_logger
from src.discrete_encoder import  DiscreteEncoder

EXP_ADV_MAX = 100.


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class ImplicitQLearning(nn.Module):
    def __init__(self, qf, vf, policy, optimizer_factory, max_steps,
                 tau, beta, discount=0.99, alpha=0.005):
        super().__init__()
        self.qf = qf.to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = vf.to(DEFAULT_DEVICE)
        self.policy = policy.to(DEFAULT_DEVICE)
        self.v_optimizer = optimizer_factory(self.vf.parameters())
        self.q_optimizer = optimizer_factory(self.qf.parameters())
        self.policy_optimizer = optimizer_factory(self.policy.parameters())
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.alpha = alpha

    def update(self, observations, actions, next_observations, rewards, terminals):
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
            next_v = self.vf(next_observations)

        # v, next_v = compute_batched(self.vf, [observations, next_observations])

        # Update value function
        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # Update Q function
        targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

        # Update policy
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.policy(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        elif torch.is_tensor(policy_out):
            assert policy_out.shape == actions.shape
            bc_losses = torch.sum((policy_out - actions)**2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

class ImplicitQLearning_obs_encoder(nn.Module):
    def __init__(self, obs_encoder, qf, vf, policy, optimizer_factory, max_steps,
                 tau, beta, discount=0.99, alpha=0.005):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.qf = qf.to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)
        #self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = vf.to(DEFAULT_DEVICE)
        self.policy = policy.to(DEFAULT_DEVICE)
        #self.v_optimizer = optimizer_factory(self.vf.parameters())
        #self.q_optimizer = optimizer_factory(self.qf.parameters())
        self.policy_optimizer = optimizer_factory(self.policy.parameters())
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.alpha = alpha

    def update_v2(self, observations, actions, label_input=False):
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
            v = self.vf(observations)
            adv = target_q - v
            exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
            if not label_input:
                obs_emb = self.obs_encoder.encode(observations, clip=True)
            else:
                _, obs_emb = self.obs_encoder.encode(observations, get_idx=True, clip=True)

        policy_out = self.policy(obs_emb.float().detach())
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        elif torch.is_tensor(policy_out):
            assert policy_out.shape == actions.shape
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

        train_metrics = {}
        train_metrics['policy_loss'] = float(policy_loss.mean().detach().cpu())
        return train_metrics

    """
    def update_v2(self, observations, actions, next_observations, rewards, terminals, label_input=False):
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
            next_v = self.vf(next_observations)
            if not label_input:
                obs_emb = self.obs_encoder.encode(observations)
            else:
                _, obs_emb = self.obs_encoder.encode(observations, get_idx=True)

        # Update value function
        v = self.vf(observations)
        adv = target_q - v
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        v_loss = asymmetric_l2_loss(adv, self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # Update Q function
        targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

        policy_out = self.policy(obs_emb.float())
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        elif torch.is_tensor(policy_out):
            assert policy_out.shape == actions.shape
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

        train_metrics = {}
        train_metrics['policy_loss'] = float(policy_loss.mean().detach().cpu())
        train_metrics['v_loss'] = float(v_loss.mean().detach().cpu())
        train_metrics['q_loss'] = float(q_loss.mean().detach().cpu())
        return train_metrics
    """

    def update(self, observations, actions, next_observations, rewards, terminals,obs_quantile_idx, **kwargs):
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
            #next_v = self.vf(next_observations)

        # v, next_v = compute_batched(self.vf, [observations, next_observations])

        # Update value function
            v = self.vf(observations)
            adv = target_q - v
            exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
            #obs_emb = self.obs_encoder.encode(observations)
            obs_emb=torch.tensor(obs_quantile_idx, dtype=torch.float32).to(DEFAULT_DEVICE)
            #obs_emb[:,:] = observations[:,:]
            #obs_emb = self.obs_encoder.decode(observations).detach()


        # Update Q function
        #targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
        #qs = self.qf.both(observations, actions)
        #q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)


        # Update target Q network
        #update_exponential_moving_average(self.q_target, self.qf, self.alpha)

        # Update policy

        policy_out = self.policy(obs_emb)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        elif torch.is_tensor(policy_out):
            assert policy_out.shape == actions.shape
            bc_losses = torch.sum((policy_out - actions)**2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

        train_metrics = {}
        train_metrics['policy_loss'] = float(policy_loss.mean().detach().cpu())
        return train_metrics


class ImplicitQLearning_recurrent(nn.Module):
    def __init__(self, qf, vf, policy, optimizer_factory, max_steps,
                 tau, beta, deterministic=True, discount=0.99, alpha=0.005, grad_norm=1.0):
        super().__init__()
        self.qf = qf.to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = vf.to(DEFAULT_DEVICE)
        self.policy = policy.to(DEFAULT_DEVICE)
        self.v_optimizer = optimizer_factory(self.vf.parameters())
        self.q_optimizer = optimizer_factory(self.qf.parameters())
        self.policy_optimizer = optimizer_factory(self.policy.parameters())
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.alpha = alpha
        self.deterministic = deterministic
        self.grad_norm = grad_norm

    def update(self, observations, actions, next_observations, rewards, terminals, shaped_rewards, **kwargs):
    #def update(self):
        seq_len, batch_size, _ = observations.shape
        device = observations.device
        prev_actions = torch.cat([torch.zeros(1, batch_size,actions.shape[-1], device=device),
                             actions[:-1]], dim=0)
        prev_rewards = torch.cat([torch.zeros(1, batch_size,rewards.shape[-1], device=device),
                             rewards[:-1]], dim=0)

        #observations = torch.ones(64, 64, 11, device='cuda')
        #next_observations = torch.ones(64, 64, 11, device='cuda')
        #actions = torch.ones(64, 64, 3, device='cuda')
        #prev_actions = torch.ones(64, 64, 3, device='cuda')
        #prev_rewards = torch.ones(64, 64, 1, device='cuda')
        #rewards = torch.ones(64, 64, 1, device='cuda')
        #terminals = torch.ones(64, 64, 1, device='cuda')


        with torch.no_grad():

            target_q = self.q_target(prev_actions=prev_actions,
                                     rewards=prev_rewards,
                                     observs=observations,
                                     current_actions=actions
            )

            next_v = self.vf(prev_actions=prev_actions,
                             rewards=prev_rewards,
                             observs=next_observations,
            )

        # v, next_v = compute_batched(self.vf, [observations, next_observations])

        # Update value function
        v = self.vf(prev_actions=prev_actions,
                    rewards=prev_rewards,
                    observs=observations
        )
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        if self.grad_norm > 0:
            grad_norms = torch.nn.utils.clip_grad_norm_(self.vf.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.v_optimizer.step()

        # Update Q function
        targets = shaped_rewards + (1. - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(prev_actions=prev_actions,
                          rewards=prev_rewards,
                          observs=observations,
                          current_actions=actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        if self.grad_norm > 0:
            grad_norms = torch.nn.utils.clip_grad_norm_(self.qf.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.q_optimizer.step()

        # Update target Q network
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

        # Update policy
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)

        new_actions, _, tanh_normal = self.policy(observations)
        if not self.deterministic:
            bc_losses = -tanh_normal.log_prob(actions)
        else:
            bc_losses = torch.sum((new_actions - actions) ** 2, dim=-1)

        policy_loss = torch.mean(exp_adv.squeeze(dim=-1) * bc_losses)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        if self.grad_norm > 0:
            grad_norms = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

        return {'policy_loss': policy_loss, 'v_loss': v_loss, 'q_loss': q_loss}


class ImplicitQLearning_discrete(nn.Module):
    def __init__(self, discrete_encoder, qf, vf, policy, optimizer_factory, max_steps,
                 tau, beta, deterministic=True, discount=0.99, alpha=0.005, grad_norm=1.0):
        super().__init__()
        self.discrete_encoder = discrete_encoder
        self.qf = qf.to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = vf.to(DEFAULT_DEVICE)
        self.policy = policy.to(DEFAULT_DEVICE)
        self.v_optimizer = optimizer_factory(self.vf.parameters())
        self.q_optimizer = optimizer_factory(self.qf.parameters())
        self.policy_optimizer = optimizer_factory(self.policy.parameters())
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.alpha = alpha
        self.deterministic = deterministic
        self.grad_norm = grad_norm

    def update(self, observations, actions,  rewards, terminals, shaped_rewards, **kwargs):
    #def update(self):
        seq_len, batch_size, _ = observations.shape
        device = observations.device
        seq_len -= 1
        #observations = observations[1:] #t to t+seq_len
        #next_observations = next_observations[1:] #t+1 to t+seq_len + 1
        #actions = actions[:-1]  # t-1, t+seq_len-1
        #rewards = rewards[:-1]
        #terminals = terminals[:-1]
        #shaped_rewards = shaped_rewards[:-1]

        train_metrics, rssm_states = self.discrete_encoder.train_batch(obs=observations[1:], actions=actions[:-1], rewards=rewards[:-1], terms=terminals[:-1])
        rssm_states = rssm_states.detach()
        #observations = torch.ones(64, 64, 11, device='cuda')
        #next_observations = torch.ones(64, 64, 11, device='cuda')
        #actions = torch.ones(64, 64, 3, device='cuda')
        #prev_actions = torch.ones(64, 64, 3, device='cuda')
        #prev_rewards = torch.ones(64, 64, 1, device='cuda')
        #rewards = torch.ones(64, 64, 1, device='cuda')
        #terminals = torch.ones(64, 64, 1, device='cuda')

        #obs: rssm_states[:-1] next_obs: rssm_states[1:]
        observations = rssm_states[:-1]
        next_observations = rssm_states[1:]
        actions = actions[1:-1]
        shaped_rewards = shaped_rewards[1:-1].squeeze(-1)
        terminals = terminals[1:-1].squeeze(-1)

        with torch.no_grad():
            target_q = self.q_target(observations, actions)
            next_v = self.vf(next_observations) #next state

        # v, next_v = compute_batched(self.vf, [observations, next_observations])

        # Update value function
        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # Update Q function
        targets = shaped_rewards + (1. - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()

        self.q_optimizer.step()

        # Update target Q network
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

        # Update policy
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)

        policy_out = self.policy(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        elif torch.is_tensor(policy_out):
            assert policy_out.shape == actions.shape
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError

        policy_loss = torch.mean(exp_adv.squeeze(dim=-1) * bc_losses)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

        train_metrics['policy_loss'] = float(policy_loss.mean().detach().cpu())
        train_metrics['v_loss'] = float(v_loss.mean().detach().cpu())
        train_metrics['q_loss'] = float(q_loss.mean().detach().cpu())
        return train_metrics