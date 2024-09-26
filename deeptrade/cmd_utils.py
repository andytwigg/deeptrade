from mpi4py import MPI
from deeptrade.agent.monitor import Monitor
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from deeptrade.agent.vec_env import VecNormalize, VecFrameStack
from baselines.common import set_global_seeds
from deeptrade.agent.utils import mpi_print
from deeptrade.envs.wrappers import RewardScaler, MetaEnvWrapper, NoopResetEnv
from deeptrade.envs import make_tradeenv

def make_vec_tradeenv(
        env_id,
        num_env,
        env_args,
        seed=None,
        start_index=0,
        is_training=True,
        gamma=0.99,
        verbose=0):

    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None

    def make_env(rank):
        def fn():
            seed_ = seed + 1024 * mpi_rank + rank if seed is not None else None
            env = make_tradeenv(env_id, env_args, seed_, is_training)
            env.seed(seed_)
            logdir = None  # logger.get_dir() and os.path.join(logger.get_dir(), modestr, str(mpi_rank)+'.'+str(rank))
            env = Monitor(
                env,
                logdir,
                allow_early_resets=True,
                info_keywords=('apv','fees_paid','lim_buy_qty','lim_sell_qty','mkt_buy_qty','mkt_sell_qty', 'map', 'inv', 'price_delta'),
            )

            if env_args['rew_scale'] != 1.0:
                mpi_print('adding reward scaler: {}'.format(env_args['rew_scale']))
                env = RewardScaler(env, scale=env_args['rew_scale'])
            if env_args['meta']:
                mpi_print('adding meta wrapper')
                env = MetaEnvWrapper(env)
            #env = NoopResetEnv(env, [0,0], 32)
            return env
        return fn

    set_global_seeds(seed)
    if num_env>1:
        mpi_print('creating {} rollout workers'.format(num_env))
        env = SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])
    else:
        env = DummyVecEnv([make_env(start_index)])

    if env_args['obnorm'] or env_args['rewnorm']:
        mpi_print('adding vec normalize with gamma={}'.format(gamma))
        env = VecNormalize(env, ob=env_args['obnorm'], ret=env_args['rewnorm'], gamma=gamma)

    if env_args['framestack']>1:
        mpi_print('using framestack {}'.format(env_args['framestack']))
        env = VecFrameStack(env, env_args['framestack'])

    return env

