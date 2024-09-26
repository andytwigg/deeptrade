from deeptrade.cmd_utils import make_vec_tradeenv
from deeptrade.args import parse_all_the_args, extract_args
import numpy as np
from keras.models import load_model
from collections import defaultdict
from supervised_train import ep2labels, confusion_report, reshape_obs, policy, eval_policy
import pandas as pd

nstack = 10
nahead = 1
tau_p = 1e-3
p_sample = 0.01

def plot_episode(env, policy):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    def convert(y):
        return y.item() if (isinstance(y, np.int32) or isinstance(y, np.float32)) else y
    logkeys = ['apv', 'fees_paid',  'price_delta', 'inv', 'ep_len', 'ep_duration']

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
    taubuf = []
    obs = env.reset()
    env_ = env.unwrapped.envs[0].unwrapped # TODO check if we can jsut do env.unwrapped
    last_price=1
    while not done:
        actions = policy(obs)
        obs, rews, dones, infos = env.step(actions)
        act, ob, rew, done, info = actions[0], obs[0], rews[0], dones[0], infos[0]
        tau = (info['price']-last_price)/last_price
        taubuf.append(tau)
        last_price=info['price']
        rewbuf.append(rew)
        actbuf.append(act)
        print('step {} seq={} act={} rew={:.3f} tau={:.4f} price={:.2f} {}'.format(step, info['seq'], act, rew, tau, info['price'], {k: round(info[k], 4) for k in logkeys}))
        # fills
        bid_fills = defaultdict(float)
        ask_fills = defaultdict(float)
        for fill in env_.fills[nfills:]:
            if fill['side'] == 'buy':
                bid_fills[float(fill['price'])] += float(fill['size'])
            else:
                ask_fills[float(fill['price'])] += float(fill['size'])
        nfills = len(env_.fills)
        for fp, fs in bid_fills.items():
            ep_bidfills.append((step - 1, fp, fs))
        for fp, fs in ask_fills.items():
            ep_askfills.append((step - 1, fp, fs))
        # current orders
        bid_orders = defaultdict(float)
        ask_orders = defaultdict(float)
        for order in env_.orders.values():
            if order['side'] == 'buy':
                bid_orders[float(order['price'])] += float(order['size'])
            else:
                ask_orders[float(order['price'])] += float(order['size'])
        for fp, fs in bid_orders.items():
            ep_bidorders.append((step - 1, fp, fs))
        for fp, fs in ask_orders.items():
            ep_askorders.append((step - 1, fp, fs))

        info = {k: convert(v) for k, v in info.items()}
        epinfo.append(info)
        step += 1

    ep_bidfills = np.asarray(ep_bidfills)
    ep_askfills = np.asarray(ep_askfills)
    ep_bidorders = np.asarray(ep_bidorders)
    ep_askorders = np.asarray(ep_askorders)
    price = np.asarray([info['price'] for info in epinfo])
    # bid = np.asarray([info['book_bid'] for info in epinfo])
    # ask = np.asarray([info['book_ask'] for info in epinfo])
    #taubuf=taubuf[1:]
    y = ep2labels(None, None, epinfo, tau_p)
    #for i,x in enumerate(zip(y,taubuf,actbuf)):
    #    print(i,x)
    print('UNSAMPLED:')
    actbuf=np.asarray(actbuf)
    confusion_report(y, actbuf[1:])
    sample = (y > 0) | (np.random.random(len(y)) < p_sample)
    print('SAMPLED')
    confusion_report(y[sample], actbuf[1:][sample])
    #return

    fig, axs = plt.subplots(6, 1, sharex=True)
    axs[0].plot(price, 'b', alpha=0.5)
    if len(ep_bidfills) > 0:
        axs[0].plot(ep_bidfills[:, 0], ep_bidfills[:, 1], 'go', markersize=10, alpha=0.5)
    if len(ep_askfills) > 0:
        axs[0].plot(ep_askfills[:, 0], ep_askfills[:, 1], 'ro', markersize=10, alpha=0.5)
    if len(ep_bidorders) > 0:
        axs[0].plot(ep_bidorders[:, 0], ep_bidorders[:, 1], 'g.', alpha=0.5)
    if len(ep_askorders) > 0:
        axs[0].plot(ep_askorders[:, 0], ep_askorders[:, 1], 'r.', alpha=0.5)
    invs = [info['inv'] for info in epinfo]
    #spreads = [info['spread'] for info in epinfo]
    rpnl = [info['realized_pnl'] for info in epinfo]
    upnl = [info['unrealized_pnl'] for info in epinfo]
    b_imbalances = [info['book_imbalance'] for info in epinfo]
    axs[1].plot(invs, label='inv')
    #axs[1].plot(y, 'rx', label='y')
    axs[2].plot(pd.Series(price).pct_change(1))
    plt.legend()
    axs[3].plot(rpnl, label='rpnl')
    axs[3].plot(upnl, label='upnl')
    axs[3].legend()
    axs[4].plot(rewbuf, label='rew')
    axs[4].plot(actbuf, 'ro')
    axs[4].legend()
    axs[5].plot(b_imbalances, label='b_imbalance')
    axs[5].legend()
    plt.show()


if __name__=='__main__':
    args = parse_all_the_args()
    env_args = extract_args('env', args)
    model_args = extract_args('model', args)
    env_args['verbose'] = args.verbose
    model_args['load_path'] = args.load_path
    print(f'env_args={env_args}')

    model_path = 'model.h5'
    print('model path={}'.format(model_path))

    envid = args.env
    nenvs = 1  # for episode replay
    seed = args.seed
    data_path = args.data
    data_level = args.level

    env_args['report_detail'] = True
    eval_env = make_vec_tradeenv(
        env_id=envid,
        num_env=nenvs,
        data=data_path,
        level=data_level,
        env_args=env_args,
        seed=seed,
        rew_scale=env_args['rew_scale'],
        obnorm=False,
        rewnorm=False,  # otherwise needs to be inside tf.session
        is_training=True,
    )

    model = load_model(model_path)

    #print('eval policy in test_env')
    #for thresh in np.arange(0.5,1.0,0.05):
    #    print(f'thresh={thresh}')
    #    eval_policy(eval_env, lambda ob: policy(model.predict(reshape_obs(ob, nstack)), thresh), neps=10)
    thresh = 0.9


    while True:
        plot_episode(eval_env, lambda ob: policy(model.predict(reshape_obs(ob, nstack))))#, thresh))
        #eval_policy(eval_env, lambda ob: policy(model.predict(reshape_obs(ob, nstack)), thresh), neps=10)
        input('press a key')
