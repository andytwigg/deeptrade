import ray
import ray.tune as tune
from ray.tune import run_experiments
from ray.tune.suggest import HyperOptSearch

from hyperopt import hp
import time
import argparse
from collections import deque
from baselines.common import explained_variance
from args import add_env_args, add_model_args, extract_args
import multiprocessing
from agent import ppo2
import os

parser = argparse.ArgumentParser()
parser.add_argument('--num-timesteps', help='number of timesteps to run', default=1e4, type=float)
parser.add_argument('--seed', help='seed', default=123, type=int)
parser.add_argument('--env', help='environment', required=True)
parser.add_argument('--policy', help='policy', default='mlp')
parser.add_argument('--nenvs', help='num envs', default=multiprocessing.cpu_count() // 4, type=int)
parser.add_argument('--verbose', help='verbose', action='store_true')
parser.add_argument('--data', help='data location (or live)', required=True)
parser.add_argument('--logdir', default='logs')
parser.add_argument('--load-path', help='load from a specific checkpoint', default=None)
parser.add_argument('--save-path', help='save to a specific checkpoint', default=None)
parser.add_argument('--play', help='play an episode', action='store_true')
parser.add_argument('--curses', action='store_true')
add_env_args(parser)
add_model_args(parser)
args = parser.parse_args()

env_args = extract_args('env', args)
env_args['verbose'] = args.verbose
model_args = extract_args('model', args)
model_args['load_path'] = args.load_path

print(env_args)
print(model_args)

# tie state_fn to policy
if args.policy == 'mlp':
    env_args['state_fn'] = 1
if args.policy == 'cnn':
    env_args['state_fn'] = 2


def train_ppo2(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
               vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
               nminibatches=4, noptepochs=4, cliprange=0.2, log_interval=1,
               load_path=None, reporter=None, **kwargs):

    if isinstance(lr, float): lr = ppo2.constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = ppo2.constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    make_model = lambda : ppo2.Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                                     nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                                     max_grad_norm=max_grad_norm)
    model = make_model()
    if load_path is not None:
        load_path=os.path.expanduser(load_path)
        print('loading model from {}'.format(load_path))
        model.load(load_path)
    runner = ppo2.Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=1000)
    tfirststart = time.time()

    nepisodes = 0
    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
        epinfobuf.extend(epinfos)
        nepisodes += len(epinfos)
        mblossvals = []
        if states is None: # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            # logger.log(logger.get_dir())
            ev = explained_variance(values, returns)
            report_dict = {}
            report_dict["nepisodes"]= nepisodes
            report_dict["nupdates"] = update
            report_dict["total_timesteps"] = update*nbatch
            report_dict["fps"] = fps
            report_dict["explained_variance"] = float(ev)
            eprewmean = ppo2.safemean([epinfo['r'] for epinfo in epinfobuf])
            report_dict['episode_reward_mean'] = eprewmean
            report_dict['eplenmean'] = ppo2.safemean([epinfo['l'] for epinfo in epinfobuf])
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                report_dict[lossname] = lossval
            for k in ['apv', 'adv_hold_apv']:
                report_dict[k] = ppo2.safemean([epinfo[k] for epinfo in epinfobuf])
            if reporter:
                reporter(timesteps=update, **report_dict)

    return model

def tune_ppo(config, reporter):
    print(config)
    env_id = args.env
    my_model_args = model_args.copy()
    my_env_args = env_args.copy()

    my_model_args.update(config)

    num_timesteps = args.num_timesteps
    seed = config['seed']
    policy_name = args.policy
    nenvs = args.nenvs
    data = args.data

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True  # pylint: disable=E1101
    tf.Session(config=config).__enter__()
    policy = {
        'lstm': LSTMPolicy,
        'mlp': MlpPolicy,
        'cnn': CnnPolicy,
    }[policy_name]

    # convert fixed args to annealed fns
    lr = my_model_args['lr']
    my_model_args['lr'] = lambda f: f * lr
    cliprange = my_model_args['cliprange']
    my_model_args['cliprange'] = lambda f: f * cliprange
    assert callable(my_model_args['cliprange']), 'cliprange not callable'
    env = make_env(env_id, nenvs, seed, data, my_env_args, gamma=my_model_args['gamma'])
    model = train_ppo2(
        policy=policy,
        env=env,
        total_timesteps=num_timesteps,
        reporter=reporter,
        log_interval=1,
        **my_model_args
    )
    env.close()


if __name__ == '__main__':
    from datetime import datetime

    tune.register_trainable("PPOTune", tune_ppo)
    ray.init()

    ncpus = multiprocessing.cpu_count()
    nenvs = args.nenvs
    print('ncpus={} nenvs={} max_concurrent={}'.format(ncpus, nenvs, 2*ncpus//nenvs))

    space = {
        'nminibatches': hp.choice('nminibatches', [4, 16]),
        'ent_coef': hp.choice('ent_coef', [0.0,0.001]),
        #'model_vf_coef': hp.choice('vgin 0.1 0.5; do
        'nsteps': hp.choice("nsteps", [128, 512, 1024, 2048]),
        'gamma': hp.choice("gamma", [0.999, 0.99]),
        'noptepochs': hp.choice('noptepochs', [4, 8]),
        'lr': hp.loguniform('lr', np.log(1e-5), np.log(1e-2)),
        'cliprange': hp.choice('cliprange', [0.1, 0.2, 0.3]),
        'seed': hp.randint('seed', 10),
    }

    algo = HyperOptSearch(space, max_concurrent=2*ncpus//nenvs, reward_attr="episode_reward_mean")
    #async_hb_scheduler = AsyncHyperBandScheduler(
    #    reward_attr='episode_reward_mean',
    #    max_t=args.num_timesteps,
    #    grace_period=10,
    #    reduction_factor=3,
    #    brackets=3)

    exp_name = 'deeptrade_{}'.format(datetime.now().isoformat(timespec='seconds'))
    run_experiments(
        {
            exp_name: {
                'run': 'PPOTune',
                'num_samples': 100,
                'trial_resources': {
                    'cpu': nenvs//2,
                    'gpu': 0
                }
            }
        }, search_alg=algo)

