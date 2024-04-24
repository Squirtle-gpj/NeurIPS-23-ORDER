import csv
from datetime import datetime
import json
from pathlib import Path
import random
import string
import sys

import numpy as np
import torch
import torch.nn as nn


DTYPE = torch.float
DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_torch(x, dtype=None, device=None):
	dtype = dtype or DTYPE
	device = device or DEFAULT_DEVICE
	if type(x) is dict:
		return {k: to_torch(v, dtype, device) for k, v in x.items()}
	elif torch.is_tensor(x):
		return x.to(device).type(dtype)
	return torch.tensor(x, dtype=dtype, device=device)

class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


def mlp(dims, activation=nn.ReLU, output_activation=None, squeeze_output=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


def compute_batched(f, xs):
    return f(torch.cat(xs, dim=0)).split([len(x) for x in xs])


def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)


def torchify(x):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)
        if x.dtype is torch.float64:
            x = x.float()
        x = x.to(device=DEFAULT_DEVICE)
    return x



def return_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, d in zip(dataset['rewards'], dataset['terminals']):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    # returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)      # but still keep track of number of steps
    assert sum(lengths) == len(dataset['rewards'])
    return min(returns), max(returns)


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




def evaluate_policy(env, policy, max_episode_steps, deterministic=True, policy_recurrent=False):
    obs = env.reset()
    total_reward = 0.
    if policy_recurrent:
        action, reward, internal_state = policy.get_initial_info()
    for _ in range(max_episode_steps):
        with torch.no_grad():
            if policy_recurrent:
                (action, _, _, _, _), internal_state = policy.act(
                    prev_internal_state=internal_state.reshape(1,1,-1),
                    prev_action=to_torch(action).reshape(1,1,-1),
                    reward=to_torch(reward).reshape(1,1,-1),
                    obs=to_torch(obs).reshape(1,1,-1),
                    deterministic=deterministic,
                )
                action = action.cpu().numpy()
            else:
                action = policy.act(torchify(obs), deterministic=deterministic).cpu().numpy()
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
        else:
            obs = next_obs
    return total_reward

def evaluate_policy_obs_encoder(env, obs_encoder, policy, max_episode_steps, deterministic=True, policy_recurrent=False, label_input=False):
    obs = env.reset()
    total_reward = 0.
    if policy_recurrent:
        action, reward, internal_state = policy.get_initial_info()
    for _ in range(max_episode_steps):
        with torch.no_grad():
            if policy_recurrent:
                (action, _, _, _, _), internal_state = policy.act(
                    prev_internal_state=internal_state.reshape(1,1,-1),
                    prev_action=to_torch(action).reshape(1,1,-1),
                    reward=to_torch(reward).reshape(1,1,-1),
                    obs=to_torch(obs).reshape(1,1,-1),
                    deterministic=deterministic,
                )
                action = action.cpu().numpy()
            else:
                if not label_input:
                    obs_emb = obs_encoder.encode(to_torch(obs), clip=True)
                else:
                    _, obs_emb = obs_encoder.encode(to_torch(obs), get_idx=True, clip=True)
                action = policy.act(obs_emb.float(), deterministic=deterministic).cpu().numpy()
        next_obs, reward, done, info = env.step(np.squeeze(action))
        total_reward += reward
        if done:
            break
        else:
            obs = next_obs
    return total_reward



def evaluate_policy_discrete(env, encoder, policy, max_episode_steps, deterministic=True,):
    obs = env.reset()
    total_reward = 0.
    done = False
    prev_rssmstate = encoder.RSSM._init_rssm_state(1)
    prev_action = torch.zeros(1, encoder.action_size).to(encoder.device)
    prev_reward = 0
    for _ in range(max_episode_steps):
        with torch.no_grad():
            model_state, rssm_state = encoder.get_state_from_obs(to_torch(obs),
                                                                 to_torch(prev_action),
                                                                 to_torch(done),
                                                                 prev_rssmstate,
                                                                 to_torch(prev_reward))
            action = policy.act(model_state, deterministic=deterministic).cpu().numpy()
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        prev_reward = reward
        prev_rssmstate = rssm_state
        prev_action = action
        if done:
            break
        else:
            obs = next_obs
    return total_reward


def set_seed(seed, env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)


def _gen_dir_name():
    now_str = datetime.now().strftime('%m-%d-%y_%H.%M.%S')
    rand_str = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f'{now_str}_{rand_str}'

class Log:
    def __init__(self, root_log_dir, cfg_dict,
                 txt_filename='log.txt',
                 csv_filename='progress.csv',
                 cfg_filename='config.json',
                 flush=True):
        self.dir = Path(root_log_dir)/_gen_dir_name()
        self.dir.mkdir(parents=True)
        self.txt_file = open(self.dir/txt_filename, 'w')
        self.csv_file = None
        (self.dir/cfg_filename).write_text(json.dumps(cfg_dict))
        self.txt_filename = txt_filename
        self.csv_filename = csv_filename
        self.cfg_filename = cfg_filename
        self.flush = flush

    def write(self, message, end='\n'):
        now_str = datetime.now().strftime('%H:%M:%S')
        message = f'[{now_str}] ' + message
        for f in [sys.stdout, self.txt_file]:
            print(message, end=end, file=f, flush=self.flush)

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)

    def row(self, dict):
        if self.csv_file is None:
            self.csv_file = open(self.dir/self.csv_filename, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, list(dict.keys()))
            self.csv_writer.writeheader()

        self(str(dict))
        self.csv_writer.writerow(dict)
        if self.flush:
            self.csv_file.flush()

    def close(self):
        self.txt_file.close()
        if self.csv_file is not None:
            self.csv_file.close()