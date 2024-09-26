import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_input

sqrt2 = np.sqrt(2)

def mlp_block(layers=[64, 64], activ=tf.nn.tanh, dropout=0, batchnorm=False):
    def network_fn(x, training):
        h = tf.layers.flatten(x)
        for i, n in enumerate(layers):
            h = fc(h, f'fc{i}', n, init_scale=np.sqrt(2))
            if dropout > 0:
                h = tf.layers.dropout(h, dropout, training=training)
            if batchnorm:
                h = tf.contrib.layers.batch_norm(h, center=True, scale=True, is_training=training)
            h = activ(h)
        return h, []

    return network_fn


def cnn1d():
    def network_fn(x, training):
        print('input_shape=', x.shape)
        x = tf.transpose(x, (0, 2, 1))
        activ = tf.nn.relu
        h = activ(tf.layers.conv1d(x, 32, 3, 1, padding='same'))
        h = activ(tf.layers.conv1d(h, 64, 3, 2, padding='same'))
        h = activ(tf.layers.conv1d(h, 64, 3, 4, padding='same'))
        out = tf.layers.flatten(h)
        # h = tf.layers.batch_normalization(h, center=True, scale=True, training=training)
        out = tf.nn.relu(tf.layers.dense(out, 256, activation=tf.nn.relu))
        return out, []
    return network_fn

from tensorflow.keras.layers import GlobalMaxPooling1D as gmp1d

def cnn1d_large():
    def network_fn(x, training):
        h=tf.transpose(x, (0,2,1))
        h = tf.layers.conv1d(h, 32, 5, 1, padding='same', activation='relu')
        h = tf.layers.conv1d(h, 32, 5, 1, padding='same', activation='relu')
        h = tf.layers.max_pooling1d(h, 2, 2)

        h = tf.layers.conv1d(h, 64, 3, 1, padding='same', activation='relu')
        h = tf.layers.conv1d(h, 64, 3, 1, padding='same', activation='relu')
        h = tf.layers.max_pooling1d(h, 2, 2)        

        h = tf.layers.conv1d(h, 64, 5, 1, padding='same', activation='relu')
        h = tf.layers.conv1d(h, 64, 5, 1, padding='same', activation='relu')
        h = tf.layers.max_pooling1d(h, 2, 2)
        
        h = tf.layers.conv1d(h, 64, 5, 1, padding='same', activation='relu')
        h = tf.layers.conv1d(h, 64, 5, 1, padding='same', activation='relu')
        h = tf.layers.max_pooling1d(h, 2, 2)

        #h = gmp1d()
        h = tf.layers.flatten(h)
        h = tf.layers.dense(h, 128, activation='relu')
        return h, []
    return network_fn


def nature_cnn(**conv_kwargs):
    def network_fn(x, training):
        activ = tf.nn.relu
        x = tf.transpose(x, (0, 2, 1))
        x = tf.expand_dims(x, -1) # rank 4 for conv2d
        h = activ(conv(x, 'c1', nf=32, rf=8, stride=4, init_scale=sqrt2, **conv_kwargs))
        h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=sqrt2, **conv_kwargs))
        h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=sqrt2, **conv_kwargs))
        h3 = conv_to_fc(h3)
        return activ(fc(h3, 'fc1', nh=512, init_scale=sqrt2)), []
    return network_fn


def impala_cnn(depths=[16, 32, 32], use_batch_norm=False, dropout=0):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """
    def network_fn(x, training):
        x = tf.transpose(x, (0, 2, 1))
        x = tf.expand_dims(x, -1) # rank 4 for conv2d

        #dropout_layer_num = [0]
        dropout_assign_ops = []
        def conv_layer(out, depth):
            out = tf.layers.conv2d(out, depth, 3, padding='same')
            out = tf.layers.dropout(out, rate=dropout)
            if use_batch_norm:
                out = tf.contrib.layers.batch_norm(out, center=True, scale=True, is_training=training)

            return out

        def residual_block(inputs):
            depth = inputs.get_shape()[-1].value
            out = tf.nn.relu(inputs)
            out = conv_layer(out, depth)
            out = tf.nn.relu(out)
            out = conv_layer(out, depth)
            return out + inputs

        def conv_sequence(inputs, depth):
            out = conv_layer(inputs, depth)
            out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
            out = residual_block(out)
            out = residual_block(out)
            return out

        out = x
        for depth in depths:
            out = conv_sequence(out, depth)

        out = tf.layers.flatten(out)
        out = tf.nn.relu(out)
        out = tf.layers.dense(out, 256, activation=tf.nn.relu)

        return out, dropout_assign_ops
    return network_fn

"""
def inception2d_block(**kwargs):
    def network_fn(x, training):
        activ = tf.nn.relu
        x = tf.transpose(x, (0, 2, 1))
        x = tf.expand_dims(x, -1)  # (batch, timesteps, feats, 1)
        print('after reshaping=', x.shape)
        # conv2d=tf.keras.layers.Conv2D

        h1 = activ(conv2d(x, 32, (1, 2), (1, 2)))
        h1 = activ(conv2d(h1, 32, (4, 1), 1, pad='SAME'))
        h1 = activ(conv2d(h1, 32, (4, 1), 1, pad='SAME'))

        h2 = activ(conv2d(h1, 32, (1, 2), (1, 2)))
        h2 = activ(conv2d(h2, 32, (4, 1), 1, pad='SAME'))
        h2 = activ(conv2d(h2, 32, (4, 1), 1, pad='SAME'))

        h3 = activ(conv2d(h2, 32, (1, 2), (1, 2)))
        h3 = activ(conv2d(h3, 32, (4, 1), 1, pad='SAME'))
        h3 = activ(conv2d(h3, 32, (4, 1), 1, pad='SAME'))

        # inception
        h4 = activ(conv2d(h3, 64, (1, 1), pad='SAME'))
        h4 = activ(conv2d(h4, 64, (3, 1), pad='SAME'))

        h5 = activ(conv2d(h3, 64, (1, 1), pad='SAME'))
        h5 = activ(conv2d(h5, 64, (5, 1), pad='SAME'))

        h6 = tf.keras.layers.MaxPooling2D((3, 1), (1, 1), padding='same')(h3)

        h7 = tf.keras.layers.concatenate([h4, h5, h6], axis=-1)
        h7 = tf.keras.layers.Reshape((100, -1))(h7)  # int(h7.shape[1]), int(h7.shape[3])) ## ?

        out = tf.keras.layers.CuDNNLSTM(128)(h7)

        # layer_1 = activ(conv(x, 'c1', nf=32, rf=3, stride=1, init_scale=np.sqrt(2), **kwargs))
        # layer_2 = activ(conv(layer_1, 'c2', nf=32, rf=3, stride=2, init_scale=np.sqrt(2), **kwargs))
        # layer_3 = activ(conv(layer_2, 'c3', nf=64, rf=3, stride=2, init_scale=np.sqrt(2), **kwargs))
        # layer_3 = conv_to_fc(layer_3)
        # out = activ(fc(layer_3, 'fc1', 512, init_scale=np.sqrt(2)))
        return out, []
    return network_fn
"""


def lstm_policy(block, backprop_aux):
    class LstmPolicy:
        def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, nauxtargets=0, training=False):
            nenv = nbatch // nsteps
            assert nenv>0, 'nenv too small: nbatch={} nsteps={} nbatch//nsteps={}'.format(nbatch, nsteps, nenv)
            #print(f'lstm_policy: ob_space={ob_space} ac_space={ac_space} batch={nbatch} nsteps={nsteps} nlstm={nlstm} nauxtargets={nauxtargets}')
            self.pdtype = make_pdtype(ac_space)
            X, processed_x = observation_input(ob_space, nbatch)

            M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
            S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
            with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
                h, self.dropout_assign_ops = block(processed_x, training=training)
                xs = batch_to_seq(h, nenv, nsteps)
                ms = batch_to_seq(M, nenv, nsteps)
                h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
                h5 = seq_to_batch(h5)
                latent = h5
                vf = fc(latent, 'v', 1)[:,0]
                self.pd, self.pi = self.pdtype.pdfromlatent(latent)
                # aux heads
                if nauxtargets>0:
                    print('building aux head with {} targets, backprop_aux={}'.format(nauxtargets, backprop_aux))
                    def sample(logits):
                        u = tf.random_uniform(tf.shape(logits), dtype=logits.dtype)
                        return tf.cast(tf.argmax(logits - tf.log(-tf.log(u)), axis=-1), tf.int32)

                    self.aux_targets, self.aux_targets0 = [], []
                    for i in range(nauxtargets):
                        if backprop_aux:
                            self.aux_targets.append(fc(latent, 'aux', 3))
                        else:
                            self.aux_targets.append(fc(tf.stop_gradient(latent), 'aux', 3))
                        self.aux_targets0.append(sample(self.aux_targets[-1]))

            a0 = self.pd.sample()
            neglogp0 = self.pd.neglogp(a0)
            a0_mode = self.pd.mode()
            neglogp0_mode = self.pd.neglogp(a0_mode)
            self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

            def step(ob, state, mask):
                return sess.run([a0, vf, snew, neglogp0], {X:ob, S:state, M:mask})

            def step_mode(ob, state, mask):
                return sess.run([a0_mode, vf, snew, neglogp0_mode], {X:ob, S:state, M:mask})

            def value(ob, state, mask):
                return sess.run(vf, {X:ob, S:state, M:mask})

            self.X = X
            self.M = M
            self.S = S
            self.vf = vf
            self.step = step
            self.step_mode = step_mode
            self.value = value
    return LstmPolicy


def ff_policy(block, backprop_aux):
    class FFPolicy:
        def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nauxtargets=0, training=False, **conv_kwargs): #pylint: disable=W0613
            print(f'ff_policy: ob_space={ob_space} ac_space={ac_space} batch={nbatch} nsteps={nsteps} nauxtargets={nauxtargets}')
            self.pdtype = make_pdtype(ac_space)
            X, processed_x = observation_input(ob_space, nbatch)

            with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
                latent, self.dropout_assign_ops = block(processed_x, training=training)
                #print(latent.shape)
                vf = fc(latent, 'v', 1)[:,0]
                self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01)
                # aux heads
                if nauxtargets>0:
                    print('building aux head with {} targets, backprop_aux={}'.format(nauxtargets, backprop_aux))
                    def sample(logits):
                        u = tf.random_uniform(tf.shape(logits), dtype=logits.dtype)
                        return tf.cast(tf.argmax(logits - tf.log(-tf.log(u)), axis=-1), tf.int32)

                    self.aux_targets, self.aux_targets0 = [], []
                    for i in range(nauxtargets):
                        if backprop_aux:
                            self.aux_targets.append(fc(latent, 'aux', 3))
                        else:
                            self.aux_targets.append(fc(tf.stop_gradient(latent), 'aux', 3))
                        self.aux_targets0.append(sample(self.aux_targets[-1]))
                #self.aux_heads = {}
                #for aux_name, aux_action_space in aux_targets.items():
                #    aux_pdtype = make_pdtype(aux_action_space)
                #    aux_pd, aux_pi  = aux_pdtype.pdfromlatent(vf_latent)
                #    self.aux_heads[aux_name] = {'pd': aux_pd}

            a0 = self.pd.sample()
            neglogp0 = self.pd.neglogp(a0)
            a0_mode = self.pd.mode()
            neglogp0_mode = self.pd.neglogp(a0_mode)
            self.initial_state = None

            def step(ob, *_args, **_kwargs):
                a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
                return a, v, self.initial_state, neglogp

            def step_mode(ob, *_args, **_kwargs):
                a, v, neglogp = sess.run([a0_mode, vf, neglogp0_mode], {X:ob})
                return a, v, self.initial_state, neglogp

            def value(ob, *_args, **_kwargs):
                return sess.run(vf, {X:ob})

            def eval(ob, op, *args, **kwargs):
                return sess.run(op, {X:ob})

            self.X = X
            self.vf = vf
            self.step = step
            self.step_mode = step_mode
            self.value = value
            self.eval = eval

    return FFPolicy


def get_policy(name, backprop_aux=False):
    # name is of the form archtype_blocktype eg ff_mlp, ff_conv1d, lstm_tcn1d
    archname,blockname=name.split('_', maxsplit=1)
    blocks = {
        'mlp': mlp_block([512]*2, batchnorm=False),
        'mlp_bn': mlp_block([512]*3, batchnorm=True),
        'mlplarge': mlp_block([512]*4, batchnorm=True),
        'impala': impala_cnn(),
        'impala_large': impala_cnn(depths=[32, 64, 64, 64, 64]),
        'cnn1d': cnn1d(),
        'cnn1dlarge': cnn1d_large(),
        'nature': nature_cnn(),
        #inception2d': inception2d_block(),
    }
    if blockname not in blocks:
        raise NotImplementedError(f'unknown blocktype={blockname}, options are {list(blocks.keys())}')
    block = blocks[blockname]
    if archname=='ff':
        return ff_policy(block, backprop_aux=backprop_aux)
    elif archname=='lstm':
        return lstm_policy(block, backprop_aux=backprop_aux)
    else:
        raise NotImplementedError(f'unknown archtype={archname}')
