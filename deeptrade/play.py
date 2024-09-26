import numpy as np
from collections import deque
from deeptrade.agent.policies import get_policy
from deeptrade.agent import ppo2
from deeptrade.cmd_utils import make_vec_tradeenv
from deeptrade.curses_interface import CursesDisplay
from deeptrade.args import common_arg_parser, load_config
from deeptrade.agent.utils import is_mpi_root, mpi_print

import os
import tensorflow as tf

"""
plays episodes in parallel using a saved model and the vectorized environment
use --curses to use the curses interface
otherwise print average rewards and apv to console
"""

def main():
    logkeys = ['seq', 'r', 'apv', 'fees_paid', 'price_delta', 'ep_len', 'ep_duration', 'lim_buy_qty', 'lim_sell_qty', 'mkt_buy_qty', 'mkt_sell_qty', 'map', 'seq']
    args = common_arg_parser().parse_args()
    if args.load_path:
        modelpath = args.load_path.split('checkpoints')[0]
        config = load_config(os.path.join(modelpath, 'config.json'))
    if args.config:
        # can overide load_path config
        config = load_config(args.config)
    env_args, model_args = config['env'], config['model']
    env_id = config['env_id']
    nenvs = config['nenvs']
    if args.curses or args.slow:
        nenvs=1
    mpi_print('Playing rollouts, using nenvs={}, deterministic_step={}'.format(nenvs, args.deterministic_step))

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True  # pylint: disable=E1101
    with tf.Session(config=tfconfig):
        env = make_vec_tradeenv(env_id, nenvs, env_args, seed=config['seed'], is_training=True, gamma=model_args['gamma'])
        if not args.random:
            mpi_print('creating PPO policy, load_path={}'.format(args.load_path))
            policy = get_policy(config['policy'], backprop_aux=config['backprop_aux'])
            model = ppo2.get_model(
                policy=policy,
                env=env,
                aux_targets=config['aux_targets'] if config['aux'] else [],
                load_path=args.load_path,
                **model_args
            )
            if args.load_path:
                env.load(modelpath) # load vecnormalize pkl

        else:
            mpi_print('using random policy')

        try:
            if args.curses:
                display = CursesDisplay(env_id)
            info_buf=deque(maxlen=1)
            def initialize_placeholders(nlstm=256, **kwargs):
                return np.zeros((nenvs or 1, 2 * nlstm)), np.zeros(nenvs)
            state, dones = initialize_placeholders()#**extra_args)
            epnum = 0
            obs = env.reset()

            while True:
                if args.random:
                    actions = [env.action_space.sample() for _ in range(nenvs)]
                    vf = [None]*nenvs
                else:
                    if args.deterministic_step:
                        actions, vf, state, neglogp = model.step_mode(obs, state=state, mask=dones)
                    else:
                        actions, vf, state, neglogp = model.step(obs, state=state, mask=dones)

                obs, rews, dones, infos = env.step(actions)
                for act, value, ob, rew, done, info in zip(actions,vf,obs,rews,dones,infos):
                    #print(ob.shape)
                    #print(ob)
                    epinfo = info.get('episode')
                    if epinfo: # done
                        epnum+=1
                        info_buf.append(epinfo)
                        if args.curses:
                            display.pause()
                        else: #elif epnum%nenvs==0:
                            print('{} {}'.format(epnum, {k: round(np.mean([info.get(k,0) for info in info_buf]), 3) for k in logkeys}))

                if args.curses:
                    display.update(env.unwrapped.envs[0].unwrapped)  # HACK to get env from wrapped dummyvecenv+monitor
                    if args.slow:#  and infos[0]['seq']>=11928300000:
                        display.pause()
                elif args.slow:
                    env.render()
                    input()


        except KeyboardInterrupt:
            pass
        finally:
            if args.curses:
                display.close()
            env.close()

if __name__=='__main__':
    main()
