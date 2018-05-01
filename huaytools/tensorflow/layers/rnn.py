"""

References:
    tf.nn.rnn
    tl.layers
"""

import tensorflow as tf
import tensorlayer as tl
import keras as K

K.layers.LSTM

tf.nn.rnn
tl.layers.RNNLayer


def rnn_basic(inputs, n_units, cell_fn,
              cell_init_args=None,
              initializer=tf.random_uniform_initializer(-0.1, 0.1),
              initial_state=None,
              n_steps=5,
              name="rnn",
              reuse=None):
    """
    最基础的 rnn 用法，相当于 `tf.nn.static_rnn` 等封装函数

    Args:
        inputs: A 3D tensor with shape `[batch_size, max_steps, n_features]`
        n_units:
        cell_fn: A RNNCell, such as `LSTMCell`, `GRUCell`, etc.
        cell_init_args:
        initializer:
        initial_state:
        n_steps(int): default 5 for basic rnn, 20~30 for lstm or gru
        name:
        reuse:

    Returns:
        outputs, final_state
        对 outputs 的处理方法一般有以下几种：
            1. outputs = outputs[-1]
            2. outputs = tf.reshape(tf.concat(outputs, axis=1), [-1, config.hidden_size])
            2. outputs = tf.reshape(tf.concat(outputs, axis=1), [-1, n_steps, n_hidden])
        这里返回原始的 outputs，具体的处理方法，放在外部完成
    """
    cell_init_args = {} if cell_init_args is None else cell_init_args

    inputs = tf.convert_to_tensor(inputs)
    batch_size = inputs.get_shape()[0].value
    max_steps = inputs.get_shape()[1].value

    n_steps = max_steps if n_steps > max_steps else n_steps

    cell = cell_fn(n_units, reuse=reuse, **cell_init_args)

    if initial_state is None:
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)
    state = initial_state

    outputs = []
    with tf.variable_scope(name, initializer=initializer) as vs:
        for time_step in range(n_steps):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            cell_output, state = cell(inputs[:, time_step, :], state)
            outputs.append(cell_output)

    outputs = tf.convert_to_tensor(outputs)
    final_state = state
    return outputs, final_state


def lstm(inputs, n_units):
    """"""

    inputs = tf.convert_to_tensor(inputs)
