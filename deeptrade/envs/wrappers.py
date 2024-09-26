import numpy as np
import gym

class TradingEnvWrapper(gym.Env):
    env = None
    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    # these properties are used by curses_interface.py
    @property
    def books(self):
        return self.env.books
    #@property
    #def client(self):
    #    return self.env.client
    @property
    def target_portfolio(self):
        return self.env.target_portfolio
    @property
    def start_portfolio(self):
        return self.env.start_portfolio
    @property
    def orders(self):
        return self.env.orders
    def close(self):
        self.env.close()
    @property
    def fills(self):
        return self.env.fills
    def step(self, action):
        return self.env.step(action)
    def reset(self):
        return self.env.reset()
    def _state(self):
        return self.env._state()
    def get_value(self):
        return self.env.get_value()
    def get_portfolio(self):
        return self.env.get_portfolio()
    def anneal_params(self, param):
        return self.env.anneal_params(param)
    @property
    def unwrapped(self):
        return self.env.unwrapped
    def render(self, mode='human'):
        self.env.render(mode)
    def seed(self, seed=None):
        return self.env.seed(seed)


class StopIterationWrapper(TradingEnvWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        if not self.done:
            try:
                self.obs, self.rew, self.done, self.info = self.env.step(action)
            except StopIteration:
                print('stop_iteration in step(), rew={}'.format(self.rew))
                self.done = True
        return self.obs, self.rew, self.done, self.info

    def reset(self, t=None):
        self.done = False
        self.obs = np.zeros(self.env.observation_space.shape)
        self.rew = 0
        self.info = {}
        try:
            self.obs = self.env.reset(t)
        except StopIteration:
            print('stop_iteration in reset()')
            self.done = True
        return self.obs

class RewardScaler(TradingEnvWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def __init__(self, env, scale=0.01):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward*self.scale, done, info


class StickyActionEnv(TradingEnvWrapper):

    def __init__(self, env, p=0.25):
        super().__init__(env)
        self.p = p
        self.last_action = None

    def reset(self):
        self.last_action = None
        return self.env.reset()

    def step(self, action):
        if (self.last_action is not None) and np.random.random() < self.p:  #np_random.uniform() < self.p:
            action = self.last_action
        self.last_action = action
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info


class MetaEnvWrapper(TradingEnvWrapper):
    # writes the action + reward into last state elements
    # need to leave room for it

    def __init__(self, env):
        super().__init__(env)
        if isinstance(env.action_space, gym.spaces.MultiDiscrete):
            nvec = env.action_space.nvec
            self.coffsets = np.cumsum(nvec)
            self.coffsets[1:] = self.coffsets[:-1]
            self.coffsets[0] = 0
        elif isinstance(env.action_space, gym.spaces.Discrete):
            nvec = [env.action_space.n]
        else:
            raise NotImplementedError('unsupported action_space: {}'.format(env.action_space))

        self.NUM_ACTIONS = np.sum(nvec)
        # 1hot actions + rew + aux info (5)
        self.OFFSET=self.NUM_ACTIONS+1

    def reset(self):
        obs = self.env.reset()
        assert np.all(obs[-self.OFFSET:]==0), 'MetaWrapper ERROR: need at least {} space for meta variables in state ob, obshape={}'.format(self.OFFSET, str(obs.shape))
        obs[-self.OFFSET:] = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        #assert np.all(obs[-self.OFFSET:]==0), 'need to leave at least {} space for meta variables in state ob, obs.shape={}'.format(self.OFFSET, str(obs.shape))
        if isinstance(action, np.ndarray):
            for i, a in enumerate(action):
                obs[-self.OFFSET+self.coffsets[i]+a] = 1.0
        else:
            obs[-self.OFFSET+action] = 1.0

        obs[-1] = float(reward)
        #obs[-6:-1] = info['aux'].values()
        return obs, reward, done, info

class NoopResetEnv(TradingEnvWrapper):
    # execute action 0 for the first k steps
    # we return after each noop action, so that the vecenv can do stacking
    # this is different to the baselines noopresetenv wrapper since
    def __init__(self, env, noop_action=0, noop_max=10):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = noop_action

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        noops = np.random.randint(1,self.noop_max)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)
