import os
import gym
import d4rl
import numpy as np
import torch
from tqdm import trange
from util import exp_logger
from util.Config import Config, get_parser
from src.vqvae import VQ_VAE

from src.iql import  ImplicitQLearning_discrete
from src.value_functions import TwinQ, ValueFunction
from src.policy import  DeterministicPolicy, GaussianPolicy
from util.util import return_range, set_seed, sample_batch, torchify, evaluate_policy_discrete
from src.discrete_encoder import  DiscreteEncoder
import torch.nn as nn


def get_env_and_dataset(env_name, max_episode_steps, sample_seq_length=None):
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

    for k, v in dataset.items():
        dataset[k] = torchify(v)

    return env, dataset


def main(config):
    torch.set_num_threads(1)
    # log = Log(Path(args.log_dir)/args.env_name, vars(args))
    # log(f'Log dir: {log.dir}')

    env, dataset = get_env_and_dataset(config.env_name, config.max_episode_steps,)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]  # this assume continuous actions
    set_seed(config.seed, env=env)
    config.add('action_size', act_dim)
    config.add('obs_shape', obs_dim)


    vq_vae = VQ_VAE(
        obs_dim=obs_dim,
        n_code=config.n_code,
        code_dim=config.code_dim,
        lr=config.learning_rate,
        max_steps=config.n_steps,
    )

    for step in trange(config.n_steps):

        results = vq_vae.update(**sample_batch(dataset, config.batch_size))
        #iql.update()
        if (step==0) or ((step + 1) % config.log_period == 0):
            exp_logger.record_step(step)
            for k, v in results.items():
                exp_logger.record_tabular(k, v)
            exp_logger.dump_tabular()
        if step==0:
            best_result = results["loss"]
            torch.save(vq_vae.state_dict(), os.path.join(config.paths['ckpt'], 'best.pt'))
        if (step+1)% config.save_period == 0:
            if results["loss"] <= best_result:
                best_result = results["loss"]
                torch.save(vq_vae.state_dict(), os.path.join(config.paths['ckpt'], 'best.pt'))
            torch.save(vq_vae.state_dict(), os.path.join(config.paths['ckpt'], 'final.pt'))

    torch.save(vq_vae.state_dict(), os.path.join(config.paths['ckpt'], 'final.pt'))
    # log.close()


if __name__ == '__main__':
    parser = get_parser(['default'])
    parser.add_argument('--env_name', required=True)
    parser.add_argument('--deterministic_policy', action='store_true')
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    os.environ['MUJOCO_GL'] = "egl"
    import util.torchkit.pytorch_utils as ptu
    ptu.set_gpu_mode(True)

    config = Config(args)
    config.load_config("vq_vae", config.cfg_filename, format="yaml")


    # learnt world-models desc

    logger_formats = ["stdout", "tensorboard", 'csv']
    exp_logger.configure(dir=config.paths["tb"], format_strs=logger_formats, precision=4)



    main(config)