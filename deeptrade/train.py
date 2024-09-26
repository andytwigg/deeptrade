from datetime import datetime
from baselines import logger
from deeptrade.agent.policies import get_policy
from deeptrade.agent import ppo2
from deeptrade.cmd_utils import make_vec_tradeenv
from mpi4py import MPI
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from deeptrade.agent import utils
from deeptrade.agent.utils import is_mpi_root, mpi_print
from deeptrade.args import common_arg_parser, load_config, save_config, dump_git_info, to_pathspec
import os

def main():
    args = common_arg_parser().parse_args()
    assert args.load_path or args.config, 'Must specify one of load_path or config'
    if args.load_path:
        config = load_config(os.path.join(args.load_path.split('checkpoints')[0], 'config.json'))
    else:
        config = load_config(args.config)
    env_args, model_args = config['env'], config['model']
    env_id = config['env_id']
    nenvs = config['nenvs']

    if is_mpi_root():
        logdir=os.getenv('OPENAI_LOGDIR')
        pathspec = to_pathspec(config)
        logpath=os.path.join(logdir, pathspec, datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")) if logdir else None
        logger.configure(logpath)
        logdir=logger.get_dir()
        # save config
        save_config(config, logdir)
        dump_git_info(logdir)
    else:
        logger.configure(format_strs=[])

    # mpi setup
    comm = MPI.COMM_WORLD
    ngpu = utils.setup_mpi_gpus()
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True  # pylint: disable=E1101
    mpi_print(f'using {ngpu} GPUs, {comm.size*nenvs} worker envs')

    with tf.Session(config=tfconfig):
        env = make_vec_tradeenv(env_id, nenvs, env_args, seed=config['seed'], is_training=True, gamma=model_args['gamma'])
        policy = get_policy(config['policy'], backprop_aux=config['backprop_aux'])
        mpi_print('env_action_space=',env.action_space)
        mpi_print('env_ob_space=',env.observation_space)

        # anneal lr
        lr_rate = model_args['lr']
        model_args['lr'] = lambda f: f * lr_rate

        model = ppo2.learn(
            policy=policy,
            env=env,
            log_interval=1,
            save_interval=100,
            total_timesteps=int(config['num_timesteps']),
            aux_targets=config['aux_targets'] if config['aux'] else [],
            load_path=args.load_path,
            verbose=1,
            **model_args
        )

        if is_mpi_root():
            checkdir = os.path.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            modelpath=os.path.join(checkdir, 'final')
            model.save(modelpath)
            env.save(logger.get_dir()) # pickle vecnormalize
            

        env.close()


if __name__ == '__main__':
    main()
