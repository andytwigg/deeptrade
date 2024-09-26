from baselines.common.vec_env import VecEnvWrapper
import numpy as np
import pickle as pkl
import os.path as osp
from .vec_env import VecEnvWrapper
import numpy as np
from gym import spaces


class VecFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack
        wos = venv.observation_space  # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs

    def save(self, path):
        if hasattr(self.venv, 'save'):
            self.venv.save(path)

    def load(self, path):
        if hasattr(self.venv, 'load'):
            self.venv.load(path)


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """
    def __init__(self, venv, training=True, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, use_tf=False):
        VecEnvWrapper.__init__(self, venv)
        if use_tf:
            from baselines.common.running_mean_std import TfRunningMeanStd
            self.ob_rms = TfRunningMeanStd(shape=self.observation_space.shape, scope='obrms') if ob else None
            self.ret_rms = TfRunningMeanStd(shape=(), scope='retrms') if ret else None
        else:
            from baselines.common.running_mean_std import RunningMeanStd
            self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
            self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        obs = self._obfilt(obs)
        self.ret = self.ret * self.gamma + rews
        if self.ret_rms:
            #if self.training:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            #if self.training:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)

    def save(self, path):
        with open(osp.join(path, 'vecnormalize.pkl'),'wb') as fh:
            pkl.dump((self.ob_rms, self.ret_rms), fh)
            print('saving',osp.join(path, 'vecnormalize.pkl'))

    def load(self, path):
        with open(osp.join(path, 'vecnormalize.pkl'),'rb') as fh:
            self.ob_rms, self.ret_rms = pkl.load(fh)
            print('loading',osp.join(path, 'vecnormalize.pkl'))

