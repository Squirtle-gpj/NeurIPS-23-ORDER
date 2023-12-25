import os
os.add_dll_directory("C://Users//Administrator//.mujoco//mujoco210//bin")
os.add_dll_directory("C://Users//Administrator//.mujoco//mujoco-py//mujoco_py") #manually importing mujoco-py dll files for windows, in Linux you can delete these two lines
import gym
import d4rl
import numpy as np
import torch
from tqdm import trange
from util import exp_logger
from util.Config import Config, get_parser



from src.iql import ImplicitQLearning
from src.policy import  DeterministicPolicy, GaussianPolicy
from src.value_functions import TwinQ, ValueFunction
from util.util import return_range, set_seed, sample_batch, torchify, evaluate_policy



def get_env_and_dataset(env_name, max_episode_steps, sample_seq_length=None):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    #seq_dataset = d4rl.sequence_dataset(env)

    if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
        min_ret, max_ret = return_range(dataset, max_episode_steps)
        print(f'Dataset returns have range [{min_ret}, {max_ret}]')
        dataset['rewards'] /= (max_ret - min_ret)
        dataset['rewards'] *= max_episode_steps
    elif 'antmaze' in env_name:
        dataset['rewards'] -= 1.

    if sample_seq_length is not None:
        dataset['valid_starts'] = np.ones(shape=dataset['terminals'].shape)
        terminal_ids = np.where(dataset['terminals']>0)[0]

        last_end = -1
        for i, terminal_idx in enumerate(terminal_ids):
            max_start = terminal_idx -sample_seq_length+1
            if max_start<=last_end:
                last_end = terminal_idx
                continue
            else:
                last_end = terminal_idx
                dataset['valid_starts'][max_start:terminal_idx+1] = 0




    for k, v in dataset.items():
        dataset[k] = torchify(v)

    return env, dataset


def main(config):


    torch.set_num_threads(1)
    #log = Log(Path(args.log_dir)/args.env_name, vars(args))
    #log(f'Log dir: {log.dir}')

    env, dataset = get_env_and_dataset( config.env_name, config.max_episode_steps)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]   # this assume continuous actions
    set_seed(config.seed, env=env)

    if config.deterministic_policy:
        policy = DeterministicPolicy(obs_dim, act_dim, hidden_dim=config.hidden_dim, n_hidden=config.n_hidden)
    else:
        policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=config.hidden_dim, n_hidden=config.n_hidden)
    def eval_policy(step):
        eval_returns = np.array([evaluate_policy(env, policy, config.max_episode_steps) \
                                 for _ in range(config.n_eval_episodes)])
        normalized_returns = d4rl.get_normalized_score(config.env_name, eval_returns) * 100.0
        return {
            'return mean': eval_returns.mean(),
            'return std': eval_returns.std(),
            'normalized return mean': normalized_returns.mean(),
            'normalized return std': normalized_returns.std(),
        }

    iql = ImplicitQLearning(
        qf=TwinQ(obs_dim, act_dim, hidden_dim=config.hidden_dim, n_hidden=config.n_hidden),
        vf=ValueFunction(obs_dim, hidden_dim=config.hidden_dim, n_hidden=config.n_hidden),
        policy=policy,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=config.learning_rate),
        max_steps=config.n_steps,
        tau=config.tau,
        beta=config.beta,
        alpha=config.alpha,
        discount=config.discount
    )

    for step in trange(config.n_steps):
        iql.update(**sample_batch(dataset, config.batch_size))


        if (step == 0) or ((step + 1) % config.eval_period == 0):
            results = eval_policy(step)
            exp_logger.record_step(step)
            for k, v in results.items():
                exp_logger.record_tabular(k, v)
            exp_logger.dump_tabular()
            if step == 0:
                best_result = results["return mean"]
                torch.save(iql.state_dict(), os.path.join(config.paths['ckpt'], 'best.pt'))
            if (step + 1) % config.save_period == 0:
                if results["return mean"] >= best_result:
                    best_result = results["return mean"]
                    torch.save(iql.state_dict(), os.path.join(config.paths['ckpt'], 'best.pt'))
                torch.save(iql.state_dict(), os.path.join(config.paths['ckpt'], 'final.pt'))

    torch.save(iql.state_dict(), os.path.join(config.paths['ckpt'], 'final.pt'))


    #log.close()


if __name__ == '__main__':

    parser = get_parser(['default'])
    
    parser.add_argument('--env_name', required=True)
    parser.add_argument('--deterministic_policy', action='store_true')
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    os.environ['MUJOCO_GL'] = "egl"
    config = Config(args)
    config.load_config("iql", config.cfg_filename, format="yaml")
    #config.add("state_dim",env_list[config.env_name]["state_dim"])
    #config.add("action_dim", env_list[config.env_name]["action_dim"])

    # logger_formats = ["stdout", "log", "csv", "tensorboard"]
    logger_formats = ["stdout", "tensorboard",'csv']
    exp_logger.configure(dir=config.paths["tb"], format_strs=logger_formats, precision=4)



    main(config)