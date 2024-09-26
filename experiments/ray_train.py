import os
import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from deeptrade.cmd_utils import make_tradeenv
from deeptrade.args import parse_all_the_args, extract_args, save_args, dump_git_info
from deeptrade.envs.wrappers import RewardScaler


def on_episode_end(info):
    episode=info['episode']
    env_info=episode.last_info_for()
    for k in ['apv','inv']:
        episode.custom_metrics[k] = env_info[k]


if __name__ == "__main__":
    args = parse_all_the_args()
    assert args.env
    env_args = extract_args('env', args)
    #model_args = extract_args('model', args)
    data_path = os.environ.get('DEEPTRADE_DATA')
    seed = 1
    is_training = True
    ray.init()
    #make_tradeenv(env_id, data, level, env_args, seed_, is_training)
    register_env("deeptrade", lambda env_conf: \
        RewardScaler(make_tradeenv(args.env, data_path, args.level, env_conf, seed, True), 0.1)
    )
    tune.run(
        "PPO",
        config={
            "env": "deeptrade",
            "lambda": 0.95,
            #"kl_coeff": 0.5,
            "clip_rewards": True,
            "clip_param": 0.2,
            "vf_clip_param": 10.0,
            "vf_loss_coeff": 0.1,
            "entropy_coeff": 0.01,
            "train_batch_size": 5000,
            "sample_batch_size": 20,
            "sgd_minibatch_size": 500,
            "num_sgd_iter": 10,
            "num_workers": 10,
            "num_envs_per_worker": 4,
            "batch_mode": "truncate_episodes",
            "observation_filter": "MeanStdFilter",
            "vf_share_layers": True,
            "model": {"use_lstm": True},
            #"lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
            "num_workers": 11,
            "num_envs_per_worker": 4,
            "num_gpus": 1,
            "env_config": env_args,
            "observation_filter": "MeanStdFilter", #  https://ray.readthedocs.io/en/latest/rllib-training.html#common-parameters
            "callbacks": {
                "on_episode_end": on_episode_end,
            },
        },
    )
