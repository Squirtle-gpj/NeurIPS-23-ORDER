






def main(config):
    torch.set_num_threads(1)
    # log = Log(Path(args.log_dir)/args.env_name, vars(args))
    # log(f'Log dir: {log.dir}')

    env, dataset, discretizer = get_env_and_dataset(config.env_name, config.max_episode_steps, config.sample_seq_length, config.n_code)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]  # this assume continuous actions
    set_seed(config.seed, env=env)
    #config.add('action_size', act_dim)
    #config.add('obs_shape', obs_dim)

    #obs_encoder = discretizer
    #obs_encoder.load_state_dict(torch.load(config.obs_encoder_path))
    #obs_encoder.set_encode_type('recon')
    #obs_encoder.set_encode_type('identity')
    emb_dim = obs_dim
    #emb_dim = config.code_dim

    if config.deterministic_policy:
        policy = DeterministicPolicy(emb_dim, act_dim, hidden_dim=config.hidden_dim, n_hidden=config.n_hidden)
    else:
        policy = GaussianPolicy(emb_dim, act_dim, hidden_dim=config.hidden_dim, n_hidden=config.n_hidden)
    ckpt = ImplicitQLearning(
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
        obs_discretizer=discretizer,
        max_steps=config.n_steps,
        continuous=config.continuous,
        use_obs_as_labels=config.use_obs_as_labels,
    )
    pomdp_encoder.load_state_dict(torch.load(config.encoder_path))
    observation_manager = ObservationManager(obs_dim=obs_dim, env_name=config.env_code)


    results, info = eval_policy(env, pomdp_encoder, policy, config, observation_manager, encoder_recurrent=True, deterministic=True, get_info=True)
    exp_logger.record_step(0)
    for k, v in results.items():
        exp_logger.record_tabular(k, v)
    exp_logger.dump_tabular()
    torch.save(info, os.path.join(config.paths['results'], 'info.pt'))


    # log.close()


if __name__ == '__main__':
    import os

    from util.Config import Config, get_parser

    parser = get_parser(['default'])
    parser.add_argument('--env_name', required=True)
    parser.add_argument('--env_code', required=True)
    parser.add_argument('--continuous', default=False)
    parser.add_argument('--use_obs_as_labels', default=False)
    parser.add_argument('--deterministic_policy', action='store_true')
    parser.add_argument("--n_code", default=50, help="", type=int)
    parser.add_argument("--policy_path", default='/home/pengjie/Experiments/test_vae_policy/Checkpoints/test_code10_1234/best.pt', help="", type=str)
    parser.add_argument("--encoder_path",
                        default='/home/pengjie/Experiments/test_vae_policy/Checkpoints/test_code10_1234/best.pt',
                        help="", type=str)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    os.environ['MUJOCO_GL'] = "egl"
    import util.torchkit.pytorch_utils as ptu
    ptu.set_gpu_mode(True)

    from tqdm import trange

    from src.policy import DeterministicPolicy, GaussianPolicy
    from src.eval.pomdp_eval import *
    from src.pomdp_encoder import POMDP_encoder

    from src.iql import ImplicitQLearning
    from src.policy import DeterministicPolicy, GaussianPolicy
    from src.value_functions import TwinQ, ValueFunction

    config = Config(args)
    config.load_config("pomdp_encoder", config.cfg_filename, format="yaml")

    from util import exp_logger
    # learnt world-models desc

    logger_formats = ["stdout", "tensorboard", 'csv']
    exp_logger.configure(dir=config.paths["tb"], format_strs=logger_formats, precision=4)



    main(config)