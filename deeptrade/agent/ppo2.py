import os
import time
import functools
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from baselines.common.tf_util import save_variables, load_variables
from baselines.common.runners import AbstractEnvRunner
from baselines.common.tf_util import initialize
from baselines.common.mpi_util import sync_from_root
tf.logging.set_verbosity(tf.logging.ERROR)

from mpi4py import MPI
import deeptrade.agent.utils as utils
mpi_print = utils.mpi_print

def display_var_info(vars):
    from baselines import logger
    count_params = 0
    for v in vars:
        name = v.name
        if "/Adam" in name or "beta1_power" in name or "beta2_power" in name: continue
        v_params = np.prod(v.shape.as_list())
        count_params += v_params
        if "/b:" in name or "/biases" in name: continue  # Wx+b, bias is not interesting to look at => count params, but not print
        logger.info("\t%s%s %s \t%i params" % (name, " " * (30 - len(name)), str(v.shape), v_params))

    logger.info("Total model parameters: %0.2f m" % (count_params * 1e-6))

# based on https://github.com/openai/coinrun/blob/master/coinrun/ppo2.py

class MpiAdamOptimizer(tf.train.AdamOptimizer):
    """Adam optimizer that averages gradients across mpi processes."""
    def __init__(self, comm, **kwargs):
        self.comm = comm
        tf.train.AdamOptimizer.__init__(self, **kwargs)

    def compute_gradients(self, loss, var_list, **kwargs):
        grads_and_vars = tf.train.AdamOptimizer.compute_gradients(self, loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = tf.concat([tf.reshape(g, (-1,)) for g, v in grads_and_vars], axis=0)
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(sum(sizes), np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars


class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                 nsteps, ent_coef, vf_coef, l2_coef, aux_coef, max_grad_norm, aux_targets):
        sess = tf.get_default_session()

        nauxtargets=len(aux_targets)
        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
            train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, nauxtargets=nauxtargets, training=True)
            norm_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, nauxtargets=nauxtargets, training=False)

            A = train_model.pdtype.sample_placeholder([None])
            ADV = tf.placeholder(tf.float32, [None])
            R = tf.placeholder(tf.float32, [None])
            OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
            OLDVPRED = tf.placeholder(tf.float32, [None])
            LR = tf.placeholder(tf.float32, [])
            CLIPRANGE = tf.placeholder(tf.float32, [])
            neglogpac = train_model.pd.neglogp(A)
            entropy = ent_coef * tf.reduce_mean(train_model.pd.entropy())
            vpred = train_model.vf
            vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
            vf_losses1 = tf.square(vpred - R)
            vf_losses2 = tf.square(vpredclipped - R)
            vf_loss = vf_coef * .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
            ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
            pg_losses = -ADV * ratio
            pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
            approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
            clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

            params = tf.trainable_variables()
            weight_params = [v for v in params if '/b' not in v.name]
            if utils.is_mpi_root():
                display_var_info(weight_params)

            l2_loss = l2_coef * tf.reduce_sum([tf.nn.l2_loss(v) for v in weight_params])
            # coefficients have been pushed into the variables
            loss = pg_loss - entropy + vf_loss + l2_loss

            # aux losses
            T = tf.placeholder(tf.int32, [None, nauxtargets], name='auxtargets')
            aux_loss = tf.constant(0.0)
            aux_accs = []
            if nauxtargets>0:
                mpi_print('adding losses for auxtargets {}'.format(aux_targets))
                aux_coef /= nauxtargets
                for i,aux_name in enumerate(aux_targets):
                    auxlabel = T[:,i]
                    auxlogit = train_model.aux_targets[i]
                    #print(auxlabel.shape, auxlogit.shape)
                    aux_loss += aux_coef * tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=auxlabel, logits=auxlogit))
                    #aux_baseline = tf.reduce_mean(tf.cast(tf.equal(auxlabel, 1), tf.float32))
                    aux_acc = tf.reduce_mean(tf.cast(tf.equal(auxlabel, train_model.aux_targets0[i]), tf.float32))#/aux_baseline
                    aux_accs.append(aux_acc)
                    #aux_loss = aux_coef * tf.reduce_mean(tf.losses.huber_loss(T, train_model.aux_targets))
                loss += aux_loss

            comm_train_size = MPI.COMM_WORLD.Get_size()
            if comm_train_size>1:
                #mpi_print("PPO2: using MpiAdamOptimizer, nmpi={}".format(comm_train_size))
                trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
            else:
                #mpi_print("PPO2: Using Non-mpi optimizer")
                trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)

            grads_and_var = trainer.compute_gradients(loss, params)
            grads, var = zip(*grads_and_var)
            if max_grad_norm is not None:
                grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            grads_and_var = list(zip(grads, var))
            with tf.control_dependencies(norm_update_ops):
                _train = trainer.apply_gradients(grads_and_var)

            def train(lr, cliprange, obs, returns, masks, actions, values, neglogpacs, auxtargets, states=None):
                advs = returns - values
                adv_mean = np.mean(advs, axis=0, keepdims=True)
                adv_std = np.std(advs, axis=0, keepdims=True)
                advs = (advs - adv_mean) / (adv_std + 1e-8)
                td_map = {train_model.X: obs, A: actions, ADV: advs, R: returns, LR: lr,
                          CLIPRANGE: cliprange, OLDNEGLOGPAC: neglogpacs, OLDVPRED: values, T:auxtargets}
                if states is not None:
                    td_map[train_model.S] = states
                    td_map[train_model.M] = masks
                #_auxloss, _auxacc, _auxpred= sess.run([aux_loss, aux_acc, train_model.aux_targets0], {train_model.X:obs, T:auxtargets})
                #print(f'auxtargets={auxtargets}\naux_pred=  {_auxpred} aux_loss={_auxloss:.4f} aux_acc={_auxacc:.4f}')#.format(sess.run([aux_loss]), td_map))#{train_model.X:obs, T:auxtargets}))
                runtargets =  [loss, aux_loss, pg_loss, vf_loss, l2_loss, entropy, approxkl, clipfrac]+aux_accs
                results = sess.run(runtargets+[_train], td_map)
                return results[:-1]

            self.loss_names = ['loss/total', 'loss/aux', 'loss/policy', 'loss/value', 'loss/l2', 'loss/policy_entropy', 'loss/approxkl', 'loss/clipfrac'] + ['aux/'+x for x in aux_targets]

            #ratio_summary = tf.summary.histogram('train/pac_ratio', ratio)
            #ret_summary = tf.summary.histogram('train/returns', self.R)
            #self.summ_op = tf.summary.merge([ratio_summary, ret_summary])

            self.train = train
            self.train_model = train_model
            self.act_model = act_model
            self.step = act_model.step  # sampled step
            self.step_mode = act_model.step_mode  # deterministic step
            self.value = act_model.value
            self.initial_state = act_model.initial_state

            self.save = functools.partial(save_variables, sess=sess)
            self.load = functools.partial(load_variables, sess=sess)

            if comm_train_size>1:
                if MPI.COMM_WORLD.Get_rank() == 0:
                    initialize()
                global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
                sync_from_root(sess, global_variables)  # pylint: disable=E1101
            else:
                initialize()

class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, nsteps, gamma, lam, aux_targets=[]):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma
        self.aux_targets = aux_targets

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        mb_auxtargets = []
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, state=self.states, mask=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
            # 1-step aux targets
            auxinfos = [info.get('aux') for info in infos]
            auxs = np.stack(np.asarray([np.asarray([aux[k] for k in self.aux_targets]) for aux in auxinfos]))
            #auxs = np.asarray([info.get('aux')['pdelta0'] for info in infos])
            mb_auxtargets.append(auxs)

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_auxtargets = np.asarray(mb_auxtargets, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, state=self.states, mask=self.dones)

        # discount/bootstrap off value fn
        #mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_auxtargets)),
            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f


def get_model(*, policy, env, nsteps=2048, nminibatches=4,
              ent_coef=0.0, vf_coef=0.5, l2_coef=1e-4, aux_coef=0.1, max_grad_norm=0.5, load_path=None, aux_targets=[], **kwargs):
    nenvs = env.num_envs
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    model = Model(policy=policy, ob_space=env.observation_space, ac_space=env.action_space, nbatch_act=nenvs, nbatch_train=nbatch_train, nsteps=nsteps,
                  ent_coef=ent_coef, vf_coef=vf_coef, l2_coef=l2_coef, aux_coef=aux_coef, max_grad_norm=max_grad_norm, aux_targets=aux_targets)
    if load_path is not None:
        print(f'[ppo2.get_model] loading model from {load_path}')
        model.load(load_path)
        # HACK
        env.load(os.path.join(load_path.split('checkpoints')[0]))
    return model

def learn(*, policy, env, total_timesteps, eval_env = None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
          vf_coef=0.5, l2_coef=1e-4, aux_coef=0.1, max_grad_norm=0.5, gamma=0.99, lam=0.95,
          eval_interval=10, log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
          save_interval=0, load_path=None, aux_targets=[], callback=None, verbose=0):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    set_global_seeds(seed)
    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    if verbose > 0:
        print('nenvs={} nbatch={} nbatch_train={} nsteps={} nminibatches={}'.format(nenvs, nbatch, nbatch_train, nsteps, nminibatches))
    model = get_model(policy=policy, env=env, nsteps=nsteps, nminibatches=nminibatches, ent_coef=ent_coef,
                      vf_coef=vf_coef, l2_coef=l2_coef, aux_coef=aux_coef, max_grad_norm=max_grad_norm, load_path=load_path, aux_targets=aux_targets)
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, aux_targets=aux_targets)
    if eval_env is not None:
        eval_runner = Runner(env=eval_env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, aux_targets=aux_targets)

    EVALBUFLEN=100
    epinfobuf = deque(maxlen=EVALBUFLEN)
    eval_epinfobuf = deque(maxlen=EVALBUFLEN)
    epnum = 0
    tfirststart = time.time()
    nupdates = total_timesteps//nbatch

    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)

        # any env-specific annealing
        #if hasattr(env, "anneal_params"):
        #env.anneal_params(frac)

        t=time.time()
        obs, returns, masks, actions, values, neglogpacs, auxtargets, states, epinfos = runner.run() #pylint: disable=E0632
        epinfobuf.extend(epinfos)
        epnum += len(epinfos)
        if eval_env is not None and update%eval_interval==0:
            eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_auxtargets, eval_states, eval_epinfos = eval_runner.run() #pylint: disable=E0632
            eval_epinfobuf.extend(eval_epinfos)

        mblossvals = []
        t = time.time()
        if states is None: # nonrecurrent version
            inds = np.arange(nbatch)
            for epoch in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs, auxtargets))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs, auxtargets))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if rank==0 and (update%log_interval==0 or update==1):
            mpi_print(logger.get_dir())
            ev = explained_variance(values, returns)
            logger.logkv("timesteps_m", update*nsteps/1e6)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps_m", update*nbatch/1e6)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprew_min', safemin([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eprew_max', safemax([epinfo['r'] for epinfo in epinfobuf]))
            eprew_mean = safemean([epinfo['r'] for epinfo in epinfobuf])
            eprew_std = safestd([epinfo['r'] for epinfo in epinfobuf])
            logger.logkv('eprew_mean', eprew_mean)
            logger.logkv('eprew_std', eprew_std)
            logger.logkv('eplen_mean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('epnum', epnum)
            if eval_env is not None:
                logger.logkv('eval/eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]))
                logger.logkv('eval/eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]))
                eval_eprew_mean = safemean([epinfo['r'] for epinfo in eval_epinfobuf])
                logger.logkv('eval/eprew_mean', eval_eprew_mean)

            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            if len(epinfobuf) > 0:
                for k in epinfobuf[0].keys():
                    if k!='aux' and k!='episode':
                        logger.logkv(f'env/{k}', '{:.5f}'.format(safemean([epinfo[k] for epinfo in epinfobuf])))
            logger.dumpkvs()
        if callback:
            eprew_mean = safemean([epinfo['r'] for epinfo in epinfobuf])
            callback.report(eprew_mean)

        if rank==0:
            if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
                checkdir = osp.join(logger.get_dir(), 'checkpoints')
                os.makedirs(checkdir, exist_ok=True)
                savepath = osp.join(checkdir, '%.5i'%update)
                mpi_print('checkpoint: {} to {}'.format(update, savepath))
                model.save(savepath)
                env.save(logger.get_dir())

    return model

def safestd(xs):
    return 0.0 if len(xs) == 0 else np.nanstd(xs, ddof=1)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.nanmean(xs)
def safemax(xs):
    return np.nan if len(xs) == 0 else np.nanmax(xs)
def safemin(xs):
    return np.nan if len(xs) == 0 else np.nanmin(xs)
