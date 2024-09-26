import streamlit as st

import numpy as np
from deeptrade.agent.policies import get_policy
from deeptrade.agent import ppo2
from deeptrade.cmd_utils import make_vec_tradeenv
from deeptrade.args import parse_all_the_args, extract_args
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
import os
from collections import defaultdict

# parse+load args
args = parse_all_the_args(arglist=['--config', 'market-conf.json', '--env', 'loladder'])
env_args = extract_args('env', args)
model_args = extract_args('model', args)
env_args['verbose'] = args.verbose
env_args['report_detail'] = True # to report detailed info with each step instead of at end
model_args['load_path'] = args.load_path
aux_targets = ['pdelta0'] if args.auxtargets else []
nenvs = 1

st.write(env_args)

def convert(y):
    return y.item() if isinstance(y, np.int32) or isinstance(y, np.float32) else y

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # pylint: disable=E1101
with tf.Session(config=config):
    env = make_vec_tradeenv(
        env_id=args.env,
        num_env=nenvs,
        data=args.data,
        level=args.level,
        env_args=env_args,
        seed=args.seed,
        rew_scale=env_args['rew_scale'],
        obnorm=env_args['obnorm'],
        rewnorm=env_args['rewnorm'],
        is_training=False,
    )
    env_ = env.unwrapped.envs[0].unwrapped

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    policy = get_policy(args.policy)
    model = ppo2.get_model(policy=policy, env=env, aux_targets=aux_targets, **model_args)

    obs = env.reset()
    def initialize_placeholders(nlstm=256, **kwargs):
        return np.zeros((nenvs or 1, 2 * nlstm)), np.zeros((1))
    state, dones = initialize_placeholders()#**extra_args)

    logkeys = ['apv', 'realized_pnl', 'unrealized_pnl', 'inv', 'price_delta', 'ep_duration', 'spread', 'fees_paid']
    os.makedirs('out', exist_ok=True)
    epinfo = []
    ep_bidfills = []
    ep_askfills = []
    ep_bidorders = []
    ep_askorders = []
    done = False
    nfills = 0
    step = 0
    actbuf = []
    rewbuf = []
    obbuf=[]
    fills=[]
    while not done:
        actions, vf, state, neglogp = model.step(obs, state=state, mask=dones)
        obs, rews, dones, infos = env.step(actions)
        act, value, ob, rew, done, info = actions[0], vf[0], obs[0], rews[0], dones[0], infos[0]

        rewbuf.append(rew)
        actbuf.append(act)
        # fills since last step
        st.write('step {} time={} seq={} act={} rew={:.3f} {}'.format(step, info['time'], info['seq'], act, rew, {k:round(info[k], 4) for k in logkeys}))
        time=info['time']
        for fill in env_.fills[nfills:]:
            if fill['side']=='buy':
                ep_bidfills.append((fill['time'], fill['price'], fill['size']))
            else:
                ep_askfills.append((fill['time'], fill['price'], fill['size']))
        nfills=len(env_.fills)
        # current orders
        bid_orders=defaultdict(float)
        ask_orders=defaultdict(float)
        for order in env_.orders.values():
            if order['side']=='buy':
                bid_orders[float(order['price'])]+=float(order['size'])
            else:
                ask_orders[float(order['price'])] += float(order['size'])
        for fp,fs in bid_orders.items():
            ep_bidorders.append((time, fp, fs))
        for fp,fs in ask_orders.items():
            ep_askorders.append((time, fp, fs))

        epinfo.append({k:convert(v) for k,v in info.items()})
        obbuf.append(np.ravel(obs))
        step+=1

    # plot episode
    st.write({k: info['episode'][k] for k in logkeys})
    ep_bidfills = np.asarray(ep_bidfills)
    ep_askfills = np.asarray(ep_askfills)
    ep_bidorders = np.asarray(ep_bidorders)
    ep_askorders = np.asarray(ep_askorders)
    df_info = pd.DataFrame.from_records(epinfo)
    t_start=df_info['time'].ix[0]
    df_info['time'] = df_info['time']-t_start
    fig, axs = plt.subplots(5, 1, sharex=True)
    axs[0].plot(df_info['time'], df_info['price'], 'b', alpha=0.5)
    if len(ep_bidfills)>0:
        axs[0].plot(ep_bidfills[:,0]-t_start, ep_bidfills[:,1], 'gX', markersize=10, alpha=0.5)
    if len(ep_askfills)>0:
        axs[0].plot(ep_askfills[:,0]-t_start, ep_askfills[:,1], 'rX', markersize=10, alpha=0.5)
    if len(ep_bidorders)>0:
        axs[0].plot(ep_bidorders[:,0]-t_start, ep_bidorders[:,1], 'g_', alpha=0.5)
    if len(ep_askorders)>0:
        axs[0].plot(ep_askorders[:,0]-t_start, ep_askorders[:,1], 'r_', alpha=0.5)

    axs[1].plot(df_info['time'], df_info['spread'], label='inv')
    axs[1].axhline(0, ls='--')
    plt.legend()
    axs[2].plot(df_info['time'], df_info['realized_pnl'], label='rpnl')
    axs[2].plot(df_info['time'], df_info['unrealized_pnl'], label='upnl')
    axs[2].axhline(0, ls='--')
    axs[2].legend()
    axs[3].plot(df_info['time'], rewbuf, label='rew')
    axs[3].axhline(0, ls='--')
    axs[3].legend()
    axs[4].plot(df_info['time'], df_info['book_imbalance'], label='b_imbalance')
    axs[4].plot(df_info['time'], df_info['book_imbalance'].rolling(10).mean())
    axs[4].legend()

    st.pyplot()

    env.close()
