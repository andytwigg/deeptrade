import os
import numpy as np
from datetime import datetime
from pprint import pprint
from collections import deque
from sys import argv

from deeptrade.cmd_utils import make_vec_tradeenv
from deeptrade.args import load_config
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

config_file = argv[1]
model_type = argv[2]
lookahead = int(argv[3])
target_label=argv[4] # price, bid, ask
threshold=float(argv[5])
print('config_file={} model_type={} lookahead={} target_label={} threshold={}'.format(config_file, model_type, lookahead, target_label, threshold))

config = load_config(config_file)
env_args, model_args = config['env'], config['model']
env_id = config['env_id']
nenvs = config['nenvs']
print('framestack={} state={}'.format(env_args['framestack'], env_args['state_fn']))

model_path = 'model_{}_{}_{}.h5'.format(model_type, lookahead, target_label)
nenvs = 12#args.nenvs
seed = config['seed']
data_path = os.environ.get('DEEPTRADE_DATA')
train_env = make_vec_tradeenv(env_id, nenvs, env_args, seed=config['seed'], is_training=True)

from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, name='Attention', **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

    def build(self, input_shape):
        assert len(input_shape) == 3, input_shape
        self.W = self.add_weight((input_shape[-1],),
                                 initializer='glorot_uniform',
                                 trainable=True)

        self.features_dim = input_shape[-1]
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     trainable=True)
        else:
            self.b = None
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


# build model
ob = train_env.reset()
input_shape = ob.shape[1:] # ignore nenvs
print("input_shape={}".format(input_shape))
model = Sequential()
if model_type == 'lstm_attention':
    model.add(Permute((2,1), input_shape=input_shape))  # output= (batch, time_steps, features)
    model.add(BatchNormalization())
    model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
    model.add(Attention(64))

elif model_type=='lstm':
    model.add(Permute((2,1), input_shape=input_shape))  # output= (batch, time_steps, features)
    model.add(BatchNormalization())
    model.add(Bidirectional(CuDNNLSTM(128)))

elif model_type=='lstm_dp':
    model.add(Permute((2,1), input_shape=input_shape))  # output= (batch, time_steps, features)
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(128, dropout=0.5)))

elif model_type == 'mlp':
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='relu'))

elif model_type == 'mlp_big':
    dropout=0.5
    model.add(Flatten(input_shape=input_shape))
    for _ in range(3):
        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropout))

elif model_type == 'cnn':
    model.add(Reshape(input_shape+(1,), input_shape=input_shape)) # np.expand_dims(ob,-1)
    model.add(Conv2D(32, 5, strides=3, padding='same', activation='relu'))
    model.add(Conv2D(32, 3, strides=2, padding='same', activation='relu'))
    model.add(Conv2D(32, 1, strides=1, padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

elif model_type=='cnn1d':
    print('input_shape=', input_shape)
    dp=0.5
    model.add(Permute((2,1), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv1D(32, 5, activation='relu', padding='same'))
    model.add(Conv1D(32, 5, activation='relu', padding='same'))
    model.add(Dropout(dp))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(64, 3, activation='relu', padding='same'))
    model.add(Conv1D(64, 3, activation='relu', padding='same'))
    model.add(Dropout(dp))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(Dropout(dp))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(256, 3, activation='relu', padding='same'))
    model.add(Conv1D(256, 3, activation='relu', padding='same'))
    model.add(Dropout(dp))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    
    model.add(GlobalMaxPooling1D())
    #model.add(Dropout(0.2))
    #model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dp))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))

elif model_type == 'cnn_big':
    dropout=0.2
    model.add(Reshape(input_shape+(1,), input_shape=input_shape))
    for _ in range(4):
        model.add(Conv2D(32, (3,3), strides=1, padding='same'))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    model.add(Conv2D(2, (1,1), strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    #model.add(Dropout(dropout))

model.add(Dense(3, activation='softmax'))
adam = Adam(lr=1e-3)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=adam,
    metrics=['sparse_categorical_accuracy']
)
model.summary()

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
        #act = [env.action_space.sample() for _ in range(nenvs)]
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
    deltas = targets[lookahead:]-targets[:-lookahead]
    y = np.where(deltas<-threshold, 0, np.where(deltas>threshold, 2, 1))
    #deltas = targets[lookahead:]/targets[:-lookahead]
    #y = np.where(deltas<(1-threshold), 0, np.where(deltas>(1+threshold), 2, 1))
    #y = 1 + np.sign(targets[lookahead:]-targets[:-lookahead]).astype(int)  # [0,1,2] labels
    #y = 1+np.sign(targets[lookahead:]).astype(int)
    return np.asarray(ep_obs)[:-lookahead], y

def pprinttime(t): return datetime.fromtimestamp(t).isoformat()

def generate_episodes(env, neps):
    global total_ep_duration
    print('rolling out {} episodes..'.format(neps))
    X,Y=[],[]
    epnum=0
    from tqdm import tqdm
    for ep_obs, ep_rews, ep_infos in tqdm(collect_episodes(env), total=neps):
        if len(ep_obs) < 100: continue  # ignore short episodes
        total_ep_duration+=ep_infos[-1]['ep_duration']
        #print('ep {} {} -> {}\t total_ep_duration={:.2f} days'.format(epnum, pprinttime(ep_infos[0]['time']), pprinttime(ep_infos[-1]['time']), total_ep_duration/86400))
        x, y = ep2labels(ep_obs, ep_infos)
        x, y = x[20:-1], y[20:-1] # trim missing framestacks and last one
        X.extend(x)
        Y.extend(y)
        epnum+=1
        if epnum>=neps: break
    print('total_ep_duration={:.2f} days'.format(total_ep_duration/86400))
    return np.asarray(X), np.asarray(Y).astype(int)

def normalize_eps(X): # x.shape = (-1,nfeats,nstack)
    return (X-np.mean(X,axis=0))/np.std(X,axis=0) # technically we should push framestack into dimension 0 but hopefully this will be fine

rollout_eps=10*nenvs
print('train on {} episodes, lookahead={}'.format(rollout_eps, lookahead))
from itertools import count
losses=deque(maxlen=100)
acc=deque(maxlen=100)
total_ep_duration=0
for iter in range(1):#count():
    X,Y = generate_episodes(train_env, rollout_eps)
    print('target distribution=', np.bincount(Y) / len(Y))
    hist=model.fit(X,Y, batch_size=64, epochs=5, validation_split=0.2, verbose=1, shuffle=True)
    metrics={k:v[-1] for k,v in hist.history.items()}
    losses.append(metrics['loss'])
    acc.append(metrics['val_sparse_categorical_accuracy'])
    print('iter {} mean_acc={:.4f}'.format(iter, np.mean(acc)))
    print('writing to', model_path)
    model.save(model_path)
    
"""
# save X,Y to disk
prodidstr = env_args['product_id'].replace('-', '')
path = f'episodes/{envid}_{prodidstr}_t{env_args["mintime"]}_p{env_args["minprice"]}_m{env_args["minmatches"]}'
os.makedirs(path, exist_ok=True)
neps=100
epnum=0
min_eplen=100+LOOKAHEAD
total_ep_duration=0
losses=deque(maxlen=100)
acc=deque(maxlen=100)
mode='train'

def pprinttime(t): return datetime.fromtimestamp(t).isoformat()

for ep_obs, ep_rews, ep_infos in collect_episodes(train_env):
    if len(ep_obs) < min_eplen: continue # ignore short episodes
    total_ep_duration+=ep_infos[-1]['ep_duration']
    x, y = ep2labels(ep_obs, ep_infos)
    if mode=='save':
        # save episode
        f=os.path.join(path, '{:06d}'.format(epnum))
        np.savez(f, x=x, y=y)
    elif mode=='train':
        # train on episode as batch
        #print(x.shape, y.shape)
        hist=model.fit(x,y)
        loss,accuracy=hist.history['loss'][-1],hist.history['sparse_categorical_accuracy'][-1]
        losses.append(loss)
        acc.append(accuracy)

        #loss = model.train_on_batch(x,y)
        #losses.append(loss[0])
        #acc.append(loss[1])
        print('mean_loss={:.4f} mean_acc={:.4f}'.format(np.mean(losses), np.mean(acc)))
        print(np.bincount(y) / len(y))

    epdur = ep_infos[-1]['time'] - ep_infos[0]['time']
    print('{} {}->{} len={} dur={:.1f}s total_duration={:.2f} days'.format(epnum, pprinttime(ep_infos[0]['time']), pprinttime(ep_infos[-1]['time']), len(ep_obs), epdur, total_ep_duration/86400))
    epnum+=1
"""
