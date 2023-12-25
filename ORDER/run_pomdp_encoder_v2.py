import os
from tqdm import trange
from util import exp_logger
from util.Config import Config, get_parser
from src.policy import  DeterministicPolicy, GaussianPolicy
from src.eval.pomdp_eval import *
from src.pomdp_encoder import POMDP_encoder
from src.vqvae_v2.vq_vae import VQ_VAE

from src.iql import ImplicitQLearning_obs_encoder
from src.policy import  DeterministicPolicy, GaussianPolicy
from src.value_functions import TwinQ, ValueFunction






def main(config):
    torch.set_num_threads(1)
    # log = Log(Path(args.log_dir)/args.env_name, vars(args))
    # log(f'Log dir: {log.dir}')

    env, dataset, discretizer = get_env_and_dataset(config.env_name, config.max_episode_steps, config.sample_seq_length, config.n_code, quantile=False)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]  # this assume continuous actions
    set_seed(config.seed, env=env)
    #config.add('action_size', act_dim)
    #config.add('obs_shape', obs_dim)

    #obs_encoder = discretizer
    #obs_encoder.load_state_dict(torch.load(config.obs_encoder_path))
    #obs_encoder.set_encode_type('recon')
    #obs_encoder.set_encode_type('identity')
    emb_dim = obs_dim*config.code_dim
    #emb_dim = config.code_dim
    obs_encoder = VQ_VAE(
        obs_dim=obs_dim,
        n_code=config.n_code,
        code_dim=config.code_dim,
        lr=config.learning_rate,
        max_steps=config.n_steps,
        share_codebook=config.share_codebook,
        vq_coef=config.vq_coef,
        commit_coef=config.commit_coef,
    )
    # obs_encoder = discretizer
    #obs_encoder.load_state_dict(torch.load(config.obs_encoder_path))

    if config.deterministic_policy:
        policy = DeterministicPolicy(emb_dim, act_dim, hidden_dim=config.hidden_dim, n_hidden=config.n_hidden)
    else:
        policy = GaussianPolicy(emb_dim, act_dim, hidden_dim=config.hidden_dim, n_hidden=config.n_hidden)

    ckpt = ImplicitQLearning_obs_encoder(
        obs_encoder=obs_encoder,
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
    ckpt.load_state_dict(torch.load(config.policy_path))
    obs_encoder=ckpt.obs_encoder
    policy = ckpt.policy


    pomdp_encoder = POMDP_encoder(
        obs_dim=obs_dim,
        action_dim=act_dim,
        n_code=config.n_code,
        token_dim=config.token_dim,
        action_embedding_size=act_dim,
        observ_embedding_size=config.observ_embedding_size,
        reward_embedding_size=config.reward_embedding_size,
        rnn_hidden_size=config.rnn_hidden_size,
        encoder_layers=config.encoder_layers,
        rnn_num_layers=1,
        action_discretizer=None,
        reward_discretizer=None,
        obs_discretizer=obs_encoder,
        max_steps=config.n_steps,
    )

    observation_manager = ObservationManager(obs_dim=obs_dim, env_name=config.env_code)
    for step in trange(config.n_steps):
        cur_scheme_idx = step % len(observation_manager.schemes)
        cur_mask_scheme = observation_manager.set_scheme(cur_scheme_idx)

        batch = sample_batch(dataset, config.batch_size, config.sample_seq_length)
        observations = batch['observations']
        seq_len, batch_size, _ = observations.shape
        device = observations.device
        prev_actions = torch.cat([torch.zeros(1, batch_size, batch['actions'].shape[-1], device=device),
                                  batch['actions'][:-1]], dim=0)
        prev_rewards = torch.cat([torch.zeros(1, batch_size, batch['rewards'].shape[-1], device=device),
                                  batch['rewards'][:-1]], dim=0)
        with torch.no_grad():
            _, labels = obs_encoder.encode(batch['observations'].reshape(-1, obs_dim), get_idx=True)
            labels = labels.reshape(config.sample_seq_length, -1, obs_dim)
        #labels = torch.tensor(batch['obs_quantile_idx'], dtype=torch.float32)

        results = pomdp_encoder.update(prev_actions, prev_rewards, observations, labels, mask_scheme=cur_mask_scheme)
        #iql.update()
        if (step == 0) or ((step + 1) % config.log_period == 0):
            exp_logger.record_step(step)
            for k, v in results.items():
                exp_logger.record_tabular(k, v)
            exp_logger.dump_tabular()

        if (step == 0) or ((step + 1) % config.eval_period == 0):
            results = eval_policy(env, pomdp_encoder, policy, config, observation_manager, encoder_recurrent=True, deterministic=True, v2=True)
            exp_logger.record_step(step)
            for k, v in results.items():
                exp_logger.record_tabular(k, v)
            exp_logger.dump_tabular()
            if (step + 1) % config.save_period == 0:
                torch.save(pomdp_encoder.state_dict(), os.path.join(config.paths['ckpt'], 'final.pt'))

    torch.save(pomdp_encoder.state_dict(), os.path.join(config.paths['ckpt'], 'final.pt'))


    # log.close()


if __name__ == '__main__':
    parser = get_parser(['default'])
    parser.add_argument('--env_name', required=True)
    parser.add_argument('--env_code', required=True)
    parser.add_argument('--deterministic_policy', action='store_true')
    parser.add_argument("--n_code", default=50, help="", type=int)
    parser.add_argument("--policy_path", default='/home/pengjie/Experiments/test_vae_policy/Checkpoints/test_code10_1234/best.pt', help="", type=str)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    os.environ['MUJOCO_GL'] = "egl"
    import util.torchkit.pytorch_utils as ptu
    ptu.set_gpu_mode(True)

    config = Config(args)
    config.load_config("pomdp_encoder_v2", config.cfg_filename, format="yaml")


    # learnt world-models desc

    logger_formats = ["stdout", "tensorboard", 'csv']
    exp_logger.configure(dir=config.paths["tb"], format_strs=logger_formats, precision=4)



    main(config)