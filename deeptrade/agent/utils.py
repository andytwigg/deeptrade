import tensorflow as tf
import os
import numpy as np
from mpi4py import MPI
import platform


def setup_mpi_gpus():
    if 'RCALL_NUM_GPU' not in os.environ:
        print('RCALL_NUM_GPU env variable not found, using 0 gpus')
        return 0
    num_gpus = int(os.environ['RCALL_NUM_GPU'])
    node_id = platform.node()
    nodes = MPI.COMM_WORLD.allgather(node_id)
    local_rank = len([n for n in nodes[:MPI.COMM_WORLD.Get_rank()] if n == node_id])
    gpuid = local_rank % num_gpus
    print('[mpi-rank {}] num_gpus={} => gpuid={}'.format(local_rank, num_gpus, gpuid))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)
    return num_gpus

def is_mpi_root():
    return MPI.COMM_WORLD.Get_rank() == 0

def tf_initialize(sess):
    sess.run(tf.initialize_all_variables())
    sync_from_root(sess)


def sync_from_root(sess, variables, comm=None):
    """
    Send the root node's parameters to every worker.
    Arguments:
      sess: the TensorFlow session.
      variables: all parameter variables including optimizer's
    """
    if comm is None: comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    for var in variables:
        if rank == 0:
            comm.bcast(sess.run(var))
        else:
            sess.run(tf.assign(var, comm.bcast(None)))


def mpi_average(values):
    return mpi_average_comm(values, MPI.COMM_WORLD)


def mpi_average_comm(values, comm):
    size = comm.size

    x = np.array(values)
    buf = np.zeros_like(x)
    comm.Allreduce(x, buf, op=MPI.SUM)
    buf = buf / size

    return buf

def mpi_print(*args):
    if is_mpi_root():
        print(*args)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class TfRunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    '''
    TensorFlow variables-based implmentation of computing running mean and std
    Benefit of this implementation is that it can be saved / loaded together with the tensorflow model
    '''
    def __init__(self, epsilon=1e-4, shape=(), scope=''):
        sess = tf.get_default_session()

        self._new_mean = tf.placeholder(shape=shape, dtype=tf.float32)
        self._new_var = tf.placeholder(shape=shape, dtype=tf.float32)
        self._new_count = tf.placeholder(shape=(), dtype=tf.float32)

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self._mean  = tf.get_variable('mean/b', initializer=np.zeros(shape, 'float32'), dtype=tf.float32)
            self._var   = tf.get_variable('std/b', initializer=np.ones(shape, 'float32'), dtype=tf.float32)
            self._count = tf.get_variable('count/b', initializer=np.full((), epsilon, 'float32'), dtype=tf.float32)

        self.update_ops = tf.group([
            self._var.assign(self._new_var),
            self._mean.assign(self._new_mean),
            self._count.assign(self._new_count)
        ])

        sess.run(tf.variables_initializer([self._mean, self._var, self._count]))
        self.sess = sess
        self._set_mean_var_count()

    def _set_mean_var_count(self):
        self.mean, self.var, self.count = self.sess.run([self._mean, self._var, self._count])

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        new_mean, new_var, new_count = update_mean_var_count_from_moments(self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

        self.sess.run(self.update_ops, feed_dict={
            self._new_mean: new_mean,
            self._new_var: new_var,
            self._new_count: new_count
        })

        self._set_mean_var_count()

