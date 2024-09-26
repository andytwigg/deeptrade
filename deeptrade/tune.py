import numpy as np
import optuna
from optuna.pruners import SuccessiveHalvingPruner, MedianPruner
from optuna.samplers import RandomSampler, TPESampler
from optuna.integration.skopt import SkoptSampler
from deeptrade.agent.policies import get_policy
from deeptrade.agent import ppo2
from deeptrade.cmd_utils import make_vec_tradeenv
import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from deeptrade.args import parse_all_the_args, extract_args

class SimpleTrialCallback:
    def __init__(self, trial):
        self.results = []
        self.is_pruned = False
        self.last_result = None
        self.trial = trial
        self.step = 0
        self.report_num = 0
        self.report_freq = 10

    def report(self, result):
        self.step += 1
        self.results.append(result)
        self.last_mean_result = np.nanmean(self.results[-10:])
        if (self.step % self.report_freq) == 0:
            self.report_num+=1
            print('step {}\t result {:.3f}'.format(self.step, result))
            self.trial.report(-1.*self.last_mean_result, self.step)
            if self.trial.should_prune(self.step):
                self.is_pruned=True
                raise optuna.exceptions.TrialPruned()


def tune(make_env, policy_name, n_trials=10, n_timesteps=5000, hyperparams=None,
        n_jobs=1, sampler_method='random', pruner_method='halving', seed=0, verbose=1):

    if hyperparams is None:
        hyperparams = {}

    n_startup_trials = 100

    if sampler_method == 'random':
        sampler = RandomSampler(seed=seed)
    elif sampler_method == 'tpe':
        sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed)
    elif sampler_method == 'skopt':
        sampler = SkoptSampler(skopt_kwargs={'base_estimator': "GP", 'acq_func': 'gp_hedge'})
    else:
        raise ValueError('Unknown sampler: {}'.format(sampler_method))

    n_evaluations = 10
    if pruner_method == 'halving':
        pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
    elif pruner_method == 'median':
        pruner = MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_evaluations // 3)
    elif pruner_method == 'none':
        pruner = MedianPruner(n_startup_trials=n_trials, n_warmup_steps=n_evaluations)
    else:
        raise ValueError('Unknown pruner: {}'.format(pruner_method))

    if verbose > 0:
        print("Sampler: {} - Pruner: {}".format(sampler_method, pruner_method))

    study = optuna.create_study(sampler=sampler, pruner=pruner)
    algo_sampler = sample_ppo2_params

    def objective(trial):
        trial.model_class = None
        # TODO add env hyperparams
        kwargs = hyperparams.copy()
        kwargs.update(algo_sampler(trial))
        eval_callback = SimpleTrialCallback(trial)
        try:
            print('trial: ', kwargs)
            env = make_env()
            policy = get_policy(policy_name)
            ppo2.learn(
                policy=policy,
                env=env,
                log_interval=1,
                total_timesteps=int(n_timesteps),
                callback=eval_callback,
                verbose=0,
                **kwargs
            )
        except AssertionError as e:
            print('ASSERTION ERROR', e)
            raise optuna.exceptions.TrialPruned()
        finally:
            env.close()
            del env

        is_pruned = eval_callback.is_pruned
        cost = -1. * eval_callback.last_mean_result
        if is_pruned:
            raise optuna.exceptions.TrialPruned()
        return cost

    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    except KeyboardInterrupt:
        pass

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print('Value:', trial.value)
    print('Params:')
    for key, value in trial.params.items():
        print('\t{}:\t{}'.format(key, value))
    return study.trials_dataframe()


def sample_ppo2_params(trial):

    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    n_steps = trial.suggest_categorical('n_steps', [16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical('gamma', [0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1)
    l2_coef = trial.suggest_loguniform('l2_coef', 1e-5, 1e-3)
    ent_coef = trial.suggest_loguniform('ent_coef', 0.00000001, 0.1)
    vf_coef = trial.suggest_loguniform('vf_coef', 0.5, 1)
    aux_coef = trial.suggest_loguniform('aux_coef', 0.000001, 1)
    cliprange = trial.suggest_categorical('cliprange', [0.1, 0.2, 0.3, 0.4])
    noptepochs = trial.suggest_categorical('noptepochs', [1, 5, 10, 20, 30, 50])
    lam = trial.suggest_categorical('lambda', [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])

    if n_steps < batch_size:
        nminibatches = 1
    else:
        nminibatches = int(n_steps / batch_size)

    return {
        'nsteps': n_steps,
        'nminibatches': nminibatches,
        'gamma': gamma,
        'lr': learning_rate,
        'l2_coef': l2_coef,
        'ent_coef': ent_coef,
        'vf_coef': vf_coef,
        'aux_coef': aux_coef,
        'cliprange': cliprange,
        'noptepochs': noptepochs,
        'lam': lam
    }


def main():
    # parse+load args
    args = parse_all_the_args()
    env_args = extract_args('env', args)
    model_args = extract_args('model', args)
    make_env = lambda: make_vec_tradeenv(args.env, args.nenvs, args.data, args.level, env_args,
        seed=args.seed,
        rew_scale=env_args['rew_scale'],
        obnorm=False,#env_args['obnorm'],
        rewnorm=False,#env_args['rewnorm'],
        use_meta=args.meta,
        is_training=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # pylint: disable=E1101
    with tf.Session(config=config):
        tune(make_env, args.policy, n_trials=100, n_timesteps=1e6,
            n_jobs=1, sampler_method='random', pruner_method='halving', seed=0, verbose=1)


if __name__ == '__main__':
    main()
