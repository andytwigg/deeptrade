import os
import numpy as np
from datetime import datetime
from pprint import pprint
from collections import deque
from sys import argv

from deeptrade.cmd_utils import make_vec_tradeenv
from deeptrade.args import load_args, extract_args
from argparse import Namespace

# generate training data

config_file = argv[1]
lookahead = int(argv[2])
target_label=argv[3] # price, bid, ask
print('config_file={} lookahead={} target_label={}'.format(config_file, lookahead, target_label))

args = Namespace(**load_args(config_file))
env_args = extract_args('env', args)
print(env_args)
model_args = extract_args('model', args)
print('framestack={} state={}'.format(env_args['framestack'], env_args['state_fn']))

envid = args.env
nenvs = 24#args.nenvs
seed = args.seed
data_path = os.environ.get('DEEPTRADE_DATA')
data_level = args.level

train_env = make_vec_tradeenv(
    env_id=envid,
    num_env=nenvs,
    data=data_path,
    level=data_level,
    env_args=env_args,
    seed=seed,
    rew_scale=env_args['rew_scale'],
    obnorm=True,
    rewnorm=False, # otherwise needs to be inside tf.session
    is_training=True,
)

# collect episodes from vecenv
def collect_episodes(env):
    nenvs=env.num_envs
    obs = env.reset()
    obs_buf = {i: [ob] for i,ob in enumerate(obs)}
    info_buf = {i: [] for i in range(nenvs)}
    rew_buf = {i: [] for i in range(nenvs)}
    done_buf = {i: [] for i in range(nenvs)}
    while True:
        act = [None] * nenvs
        # act = [env.action_space.sample() for _ in range(nenvs)]
        obs, rews, dones, infos = env.step(act)
        for i, (ob, rew, done, info) in enumerate(zip(obs, rews, dones, infos)):
            rew_buf[i].append(rew)
            info_buf[i].append(info)
            done_buf[i].append(done)
            epinfo = info.get('episode')
            if epinfo:
                yield obs_buf[i], rew_buf[i], info_buf[i]
                # clear buf for next ep
                obs_buf[i] = []
                rew_buf[i] = []
                info_buf[i] = []
                done_buf[i] = []
            obs_buf[i].append(ob)

def ep2labels(ep_obs, infos):
    targets = np.asarray([float(info[target_label]) for info in infos])
    y = 1 + np.sign(targets[lookahead:]-targets[:-lookahead]).astype(int)  # [0,1,2] labels
    return np.asarray(ep_obs)[:-lookahead], y

# save X,Y to disk
prodidstr = env_args['product_id'].replace('-', '')
path = f'episodes/{envid}_{prodidstr}_t{env_args["mintime"]}_p{env_args["minprice"]}_m{env_args["minmatches"]}'
print('writing to {}'.format(path))
os.makedirs(path, exist_ok=True)
epnum=0
total_ep_duration=0

def pprinttime(t): return datetime.fromtimestamp(t).isoformat()

for ep_obs, ep_rews, ep_infos in collect_episodes(train_env):
    if len(ep_obs) < 100: continue # ignore short episodes
    total_ep_duration+=ep_infos[-1]['ep_duration']
    x, y = ep2labels(ep_obs, ep_infos)
    # save episode
    f=os.path.join(path, '{:06d}'.format(epnum))
    np.savez(f, x=x, y=y)

    epdur = ep_infos[-1]['time'] - ep_infos[0]['time']
    print('{} {}->{} len={} dur={:.1f}s total_duration={:.2f} days'.format(epnum, pprinttime(ep_infos[0]['time']), pprinttime(ep_infos[-1]['time']), len(ep_obs), epdur, total_ep_duration/86400))
    epnum+=1

