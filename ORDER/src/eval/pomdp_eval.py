import numpy as np
import torch
import d4rl
import math
import gym
from util.util import return_range, set_seed,torchify,to_torch
from src.discretization import QuantileDiscretizer

# dataset is a dict, values of which are tensors of same first dimension
def sample_batch(dataset, batch_size, sample_seq_length=None):
    k = list(dataset.keys())[0]
    n, device = len(dataset[k]), dataset[k].device
    for v in dataset.values():
        assert len(v) == n, 'Dataset values must have same length'

    if sample_seq_length is not None:
        valid_starts_indices = torch.where(dataset['valid_starts'] > 0.0)[0]
        sampled_indices_indices = torch.randint(low=0, high=len(valid_starts_indices), size=(batch_size,), device=device)
        sampled_start_indices = valid_starts_indices[sampled_indices_indices]
        indices = []
        for start in sampled_start_indices:
            end = start + sample_seq_length
            indices.append(torch.arange(start, end, device=device))
        indices = torch.cat(indices, dim=0)
        tmp = {}
        #tmp['indices'] = indices.detach().cpu()
        for k,v in dataset.items():
            tmp[k] = v[indices].reshape(batch_size, sample_seq_length, -1).permute(1, 0, 2)
        #tmp = {k: v[indices].reshape(batch_size, sample_seq_length, -1).permute(1, 0, 2) for k, v in dataset.items()}

        return tmp
    else:
        indices = torch.randint(low=0, high=n, size=(batch_size,), device=device)
        return {k: v[indices] for k, v in dataset.items()}

def get_env_and_dataset(env_name, max_episode_steps, sample_seq_length=None, codebook_size=50, quantile=True):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    # seq_dataset = d4rl.sequence_dataset(env)

    if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
        min_ret, max_ret = return_range(dataset, max_episode_steps)
        print(f'Dataset returns have range [{min_ret}, {max_ret}]')
        dataset['shaped_rewards'] = dataset['rewards'] / (max_ret - min_ret)
        dataset['shaped_rewards'] *= max_episode_steps
    elif 'antmaze' in env_name:
        dataset['shaped_rewards'] = dataset['rewards'] - 1.

    if sample_seq_length is not None:
        dataset['valid_starts'] = np.ones(shape=dataset['terminals'].shape)
        terminal_ids = np.where(dataset['terminals'] > 0)[0]

        last_end = -1
        terminal_ids = list(terminal_ids)
        terminal_ids.append(len(dataset['rewards'])-1)
        #terminal_ids.append(len(dataset['rewards'])-1)
        for i, terminal_idx in enumerate(terminal_ids):
            max_start = terminal_idx - sample_seq_length + 1
            if max_start <= last_end:
                dataset['valid_starts'][last_end + 1:terminal_idx + 1] = 0
                last_end = terminal_idx
                continue
            else:
                last_end = terminal_idx
                dataset['valid_starts'][max_start+1:terminal_idx + 1] = 0

    if  quantile:
        obs_discretizer = QuantileDiscretizer(dataset['observations'], N=codebook_size)
        #action_discretizer = QuantileDiscretizer(dataset['actions'], N=codebook_size)
        #reward_dicretizer = QuantileDiscretizer(dataset['reward'], N=codebook_size)
        dataset['obs_quantile_idx'] = obs_discretizer.discretize(x=dataset['observations'])
        dataset['next_obs_quantile_idx'] = obs_discretizer.discretize(x=dataset['next_observations'][-1])

    for k, v in dataset.items():
        dataset[k] = torchify(v)
    if  quantile:
        dataset['next_obs_quantile_idx']  = torch.cat([dataset['obs_quantile_idx'][1:],dataset['next_obs_quantile_idx']], dim=0)
        return env, dataset, obs_discretizer
    else:
        return env, dataset, None


def eval_policy(env, obs_encoder, policy, config, observation_manager, encoder_recurrent=False, deterministic=False, use_obs_encoder=True, v2=False, padding_zero=False, get_info=False):
    all_results = {}
    total_return_mean = 0
    total_return_std = 0
    total_normalized_return_mean = 0
    total_normalized_return_std = 0
    all_info = {}

    if use_obs_encoder:
         eval_fn = evaluate_policy_obs_encoder
         eval_returns = []
         org = []
         recon = []
         for _ in range(config.n_eval_episodes):

             cur_eval_returns, cur_info = eval_fn(env, obs_encoder, policy, config.max_episode_steps,
                                                  {'observable_type': 'full', 'name': 'full'},
                                                  encoder_recurrent=False,
                                                  deterministic=deterministic, v2=v2, get_info=get_info)
             if get_info:
                 cur_org = torch.cat(cur_info[0], dim=0).cpu()
                 cur_recon = torch.cat(cur_info[1], dim=0).cpu()
                 org.append(cur_org)
                 recon.append(cur_recon)
             eval_returns.append(cur_eval_returns)
         eval_returns = np.array(eval_returns)
         if get_info:
             org = torch.cat(org, dim=0)
             recon = torch.cat(recon, dim=0)
    else:
        obs_encoder = None
        eval_fn = evaluate_policy_pomdp
        eval_returns, info = np.array([eval_fn(env, obs_encoder, policy, config.max_episode_steps,
                                         {'observable_type': 'full', 'name': 'full'},
                                         encoder_recurrent=encoder_recurrent,
                                         deterministic=deterministic) \
                                 for _ in range(config.n_eval_episodes)])



    normalized_returns = d4rl.get_normalized_score(config.env_name, eval_returns) * 100.0

    results = {
        'return mean/full': eval_returns.mean(),
        'return std/full': eval_returns.std(),
        'normalized return mean/full': normalized_returns.mean(),
        'normalized return std/full': normalized_returns.std(),
    }

    all_results.update(results)
    all_info.update({'full':[org, recon]})


    for i_scheme in range(len(observation_manager.schemes)):
        observation_manager.set_scheme(i_scheme)

        eval_fn = evaluate_policy_obs_encoder
        eval_returns = []
        org = []
        recon = []
        for _ in range(config.n_eval_episodes):
            cur_eval_returns, cur_info = eval_fn(env, obs_encoder, policy, config.max_episode_steps,
                                                 observation_manager.get_current_scheme(),
                                                 encoder_recurrent=encoder_recurrent,
                                                 deterministic=deterministic, v2=v2, get_info=get_info)
            if get_info:
                cur_org = torch.cat(cur_info[0], dim=0).cpu()
                cur_recon = torch.cat(cur_info[1], dim=0).cpu()
                org.append(cur_org)
                recon.append(cur_recon)
            eval_returns.append(cur_eval_returns)

        eval_returns = np.array(eval_returns)
        if get_info:
            org = torch.cat(org, dim=0)
            recon = torch.cat(recon, dim=0)

        normalized_returns = d4rl.get_normalized_score(config.env_name, eval_returns) * 100.0

        results = {
            'return mean/' + observation_manager.get_current_scheme()['name']: eval_returns.mean(),
            'return std/' + observation_manager.get_current_scheme()['name']: eval_returns.std(),
            'normalized return mean/' + observation_manager.get_current_scheme()['name']: normalized_returns.mean(),
            'normalized return std/' + observation_manager.get_current_scheme()['name']: normalized_returns.std(),
        }

        all_results.update(results)
        all_info.update({observation_manager.get_current_scheme()['name']: [org, recon]})

        total_return_mean += eval_returns.mean()
        total_return_std += eval_returns.std()
        total_normalized_return_mean += normalized_returns.mean()
        total_normalized_return_std += normalized_returns.std()

    all_results.update({
        'total return mean': total_return_mean/len(observation_manager.schemes),
        'total return std': total_return_std/len(observation_manager.schemes),
        'total normalized return mean': total_normalized_return_mean/len(observation_manager.schemes),
        'total normalized return std': total_normalized_return_std/len(observation_manager.schemes),
    })



    if get_info:
        return all_results, all_info
    else:
        return all_results

def evaluate_policy_obs_encoder(env, obs_encoder, policy, max_episode_steps,
                                mask_scheme=None,
                                encoder_recurrent=False,
                                deterministic=False, v2=False, get_info=False):
    info={}
    if get_info:
        org = []
        recon = []

    full_observation = env.reset()
    #full_observation, mask_indices, masked_observation = observation_manager.get_observation(observation)
    if encoder_recurrent:
        action, reward, prev_internal_state = obs_encoder.get_initial_info()
    done = False
    total_reward = 0.0
    step = 0

    while not done and step < max_episode_steps:
        if encoder_recurrent:
            prev_action = action
            encoded_observation, prev_internal_state = obs_encoder.forward_single( to_torch(prev_action),
                                                                                   to_torch(reward),
                                                                                   to_torch(full_observation),
                                                                                   mask_scheme=mask_scheme,
                                                                                   prev_internal_state=prev_internal_state, v2=v2)
            if get_info:
                org.append(obs_encoder.obs_discretizer.encode(to_torch(full_observation)).detach())
                recon.append(encoded_observation.detach())
        else:
            if mask_scheme['observable_type'] == 'full':
                encoded_observation = obs_encoder.obs_discretizer.encode(to_torch(full_observation))
                if get_info:
                    org.append(obs_encoder.obs_discretizer.encode(to_torch(full_observation)).detach())
                    recon.append(encoded_observation.detach())
            #encoded_observation = obs_encoder.encode(full_observation, mask_scheme)

        with torch.no_grad():
            #action = policy(encoded_observation.float())
            action = policy.act(encoded_observation.float(), deterministic=deterministic).reshape(-1).cpu().numpy()
            full_observation, reward, done, _ = env.step(action)
            #full_observation, mask_indices, masked_observation = observation_manager.get_observation(observation)


        total_reward += reward
        step += 1
    if get_info:
        info = [org, recon]
    return total_reward, info

def evaluate_policy_pomdp(env, obs_encoder, policy, max_episode_steps,
                                mask_scheme=None,
                                encoder_recurrent=False,
                                deterministic=False, v2=False, padding_zero=True, get_info=False):
    info = {}
    full_observation = env.reset()
    last_observation = torch.zeros(to_torch(full_observation).shape, device='cuda')
    obs_dim = full_observation.shape[0]
    #full_observation, mask_indices, masked_observation = observation_manager.get_observation(observation)
    if encoder_recurrent:
        action, reward, prev_internal_state = policy.get_initial_info()
    done = False
    total_reward = 0.0
    step = 0

    while not done and step < max_episode_steps:
        with torch.no_grad():
            if encoder_recurrent:
                prev_action = action
                (action,_,_,_,_), prev_internal_state = policy.forward_single(prev_internal_state, to_torch(prev_action), to_torch(reward), to_torch(full_observation), deterministic, False, mask_scheme)
                action = action.reshape(-1).cpu().numpy()
            else:
                full_observation = to_torch(full_observation)
                observable_type = mask_scheme.get('observable_type', 'full')
                mask_entry = mask_scheme.get('mask_entry', None)
                mask_ratio = mask_scheme.get('mask_ratio', None)
                if observable_type == 'random_step':
                    num_masked_elements_per_row = min(int(obs_dim * mask_ratio) + 1, obs_dim)
                    mask_entry = torch.randint(0, obs_dim, (num_masked_elements_per_row,))
                    if padding_zero:
                        full_observation[mask_entry] = 0
                    else:
                        full_observation[mask_entry] = last_observation[mask_entry]
                        last_observation=full_observation
                elif observable_type == 'random_episode' or observable_type=='fixed':
                    full_observation[mask_entry] = 0


                action = policy.act(to_torch(full_observation), deterministic=deterministic).reshape(-1).cpu().numpy()
                #encoded_observation = obs_encoder.encode(full_observation, mask_scheme)

            
                #action = policy(encoded_observation.float())
                
            full_observation, reward, done, _ = env.step(action)
            #full_observation, mask_indices, masked_observation = observation_manager.get_observation(observation)


            total_reward += reward
            step += 1

    return total_reward, info

class ObservationManager:
    def __init__(self, obs_dim,   env_name='hopper', train_mask_ratio=0.5):

        #self.fixed_mask_indices = fixed_mask_indices
        self.random_episode_indices = None
        self.current_scheme = None
        self.env_name = env_name
        self.train_mask_ratio=train_mask_ratio
        self.obs_dim = obs_dim

        self.initial_schemes()

    def set_scheme(self, scheme_index=None, scheme=None):
        if scheme_index is not None:
            if scheme_index >= len(self.schemes):
                raise ValueError(f"Invalid scheme_name: {scheme_index}")

            self.current_scheme = self.schemes[scheme_index]
        else:
            self.current_scheme = scheme

        observable_type = self.current_scheme['observable_type']
        if observable_type == 'random_episode':
            mask_ratio = self.current_scheme.get('mask_ratio', 0)
            num_masked_elements = math.ceil(self.obs_dim * mask_ratio)
            self.current_scheme.update({'mask_entry': torch.randperm(self.obs_dim)[:num_masked_elements]})

        return self.current_scheme

    def add_scheme(self, new_schemes):
        self.schemes.update(new_schemes)

    def get_current_scheme(self):
        return self.current_scheme

    def get_observation(self, full_observation):
        if not self.current_scheme:
            raise RuntimeError("No current_scheme is set. Use set_scheme() to set the current scheme.")

        observable_type = self.current_scheme['observable_type']

        if observable_type == 'full':
            return full_observation, torch.tensor([]).long(), full_observation

        D = full_observation.shape[0]
        mask_ratio = self.current_scheme.get('mask_ratio', 0)

        if observable_type == 'fixed':
            mask_indices = self.current_scheme['mask_indices']
        elif observable_type == 'random_step':
            num_masked_elements = math.ceil(D * mask_ratio)
            mask_indices = torch.randperm(D)[:num_masked_elements]
        elif observable_type == 'random_episode':
            if self.random_episode_indices is None:
                num_masked_elements = math.ceil(D * mask_ratio)
                self.random_episode_indices = torch.randperm(D)[:num_masked_elements]
            mask_indices = self.random_episode_indices
        else:
            raise ValueError(f"Invalid observable_type: {observable_type}")

        masked_observation = self.mask_and_add_noise(full_observation, mask_indices)
        return full_observation, mask_indices, masked_observation

    def mask_and_add_noise(self, observation, mask_indices):
        masked_observation = observation.clone()

        if self.current_scheme.get('mask_fill_type', 'zero') == 'zero':
            masked_observation[mask_indices] = 0
        elif self.current_scheme.get('mask_fill_type', 'zero') == 'noise':
            noise_scale = self.current_scheme.get('noise_scale', 0)
            noise = torch.randn(mask_indices.shape) * noise_scale
            masked_observation[mask_indices] +=  noise
        else:
            raise ValueError(f"Invalid mask_fill_type: {self.current_scheme.get('mask_fill_type', 'zero')}")

        return masked_observation

    def reset(self):
        if self.current_scheme and self.current_scheme['observable_type'] == 'random_episode':
            self.random_episode_indices = None

    def initial_schemes(self):
        self.train_scheme_info = {
            "halfcheetah": {
                "random": [0.1, 0.3, 0.5, 0.7, 0.9],
                "P": [0, 1, 2, 3, 4, 5, 6, 7],
                "V": [8, 9, 10, 11, 12, 13, 14, 15, 16],
            },
            "walker": {
                "random": [0.1, 0.3, 0.5, 0.7, 0.9],
                "P": [0, 1, 2, 3, 4, 5, 6, 7],
                "V": [8, 9, 10, 11, 12, 13, 14, 15, 16],
            },
            "hopper": {
                "random": [0.1, 0.3, 0.5],
                "P": [0, 1, 2, 3, 4],
                "V": [5, 6, 7, 8, 9, 10],
            },
            "toy": {
                "random": [0.1, 0.3, 0.5],
                "x": [0],
                "y": [1],
            },
        }
        self.train_schemes = []


        tmp = {"observable_type": "random_step",
               "mask_ratio": self.train_mask_ratio,
               "mask_entry": [0],
               "name": "random_step_" + str(self.train_mask_ratio)}
        self.train_schemes.append(tmp)


        tmp = {"observable_type": "random_episode",
               "mask_ratio": self.train_mask_ratio,
               "mask_entry": [0],
               "name": "random_episode_" + str(self.train_mask_ratio)}
        self.train_schemes.append(tmp)


        self.scheme_info = {
            "halfcheetah": {
                "random": [0.1,  0.3,  0.5, 0.7, 0.9],
                "P": [0, 1, 2, 3, 4, 5, 6, 7],
                "V": [8, 9, 10, 11, 12, 13, 14, 15, 16],
            },
            "walker": {
                "random": [0.1,  0.3,  0.5, 0.7, 0.9],
                "P": [0, 1, 2, 3, 4, 5, 6, 7],
                "V": [8, 9, 10, 11, 12, 13, 14, 15, 16],
            },
            "hopper": {
                "random": [0.1,  0.3,  0.5, 0.7, 0.9],
                "P": [0, 1, 2, 3, 4],
                "V": [5, 6, 7, 8, 9, 10],
            },
            "maze": {
                "random": [0.2, 0.4, 0.6],
                "P": [0, 1],
                "V": [2, 3],
            },
            "toy": {
                "random": [0.1,  0.3,  0.5],
                "x": [0],
                "y": [1],
            },
        }
        self.schemes = []
        for k, v in self.scheme_info[self.env_name].items():
            if k == "random":
                for i_ratio in v:
                    tmp = {"observable_type": "random_step",
                           "mask_ratio": i_ratio,
                           "mask_entry": [0],
                           "name": "random_step_" + str(i_ratio)}
                    self.schemes.append(tmp)

                for i_ratio in v:
                    tmp = {"observable_type": "random_episode",
                           "mask_ratio": i_ratio,
                           "mask_entry": [0],
                           "name": "random_episode_" + str(i_ratio)}
                    self.schemes.append(tmp)

            else:
                tmp = {"observable_type": "fixed",
                       "mask_ratio": 0,
                       "mask_entry": v,
                       "name": 'mask_' + k}
                self.schemes.append(tmp)
