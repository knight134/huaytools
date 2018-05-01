"""

说明：
    `tf.nn.static_rnn` 中是没有 time_major 这个参数的，
        它接收的是一个 `[batch_size, n_features]` 的 2D tensor
        所以当 inputs shape 为 `[batch_size, max_steps, n_features]` 时，
        需要使用 `tf.unstack(inputs, max_steps, axis=1)` 调整 shape
        ref: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dynamic_rnn.py

    `tf.nn.dynamic_rnn` 完全可以代替 `tf.nn.static_rnn`
        它默认接收的是一个 `[batch_size, max_steps, n_features]` 的 3D tensor
        因为 dynamic_rnn 出的比较晚，好多比较旧的教程都使用的是 static_rnn，

    `tf.nn.rnn_cell.GRUCell`
        注意 GRUCell 和 LSTMCell 的参数不同，所以一般需要将 cell 作为参数放到声明里，然后在外部创建 cell，
        如果想构建通用的函数，可以参考 `tl.layers.RNNLayer` - 它传入的是一个 cell_fn

References:
    tf.nn.rnn
    tl.layers
"""

import tensorflow as tf


def rnn_basic(inputs, n_units, cell_fn,
              cell_init_args=None,
              initializer=tf.random_uniform_initializer(-0.1, 0.1),
              initial_state=None,
              n_steps=5,
              name="rnn",
              reuse=None):
    """
    最基础的 rnn 用法，相当于 `tf.nn.static_rnn` 等封装函数
    实际使用中建议直接使用 `tf.nn.dynamic_rnn`

    Args:
        inputs: A 3D tensor with shape `[batch_size, max_steps, n_features]`
        n_units:
        cell_fn: A RNNCell, such as `LSTMCell`, `GRUCell`, etc.
        cell_init_args:
        initializer:
        initial_state:
        n_steps(int): default 5 for basic rnn
            如果使用 lstm 或 gru，那么 n_steps 可以设置到 20-30
        name(str):
        reuse(bool):

    Returns:
        outputs, final_state

        对 outputs 的处理方法一般有以下几种：
            1. outputs = outputs[-1]
            2. outputs = tf.reshape(tf.concat(outputs, axis=1), [-1, config.hidden_size])
            2. outputs = tf.reshape(tf.concat(outputs, axis=1), [-1, n_steps, n_hidden])
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

    # outputs = tf.convert_to_tensor(outputs)
    outputs = tf.stack(outputs, axis=0)
    final_state = state
    return outputs, final_state


def lstm(inputs, n_units,
         sequence_length=None,
         cell_init_args=None,
         rnn_init_args=None):
    """

    Args:
        inputs: A 3D tensor with shape `[batch_size, max_steps, n_features]`
        n_units(int):
        sequence_length:
        cell_init_args:
        rnn_init_args:

    Returns:
        outputs, state
            outputs shape `[batch_size, max_time, n_units]`
            更详细的说明 ref: `tf.nn.dynamic_rnn`

    """
    if cell_init_args is None:
        cell_init_args = {}
    if rnn_init_args is None:
        rnn_init_args = {}

    cell = tf.nn.rnn_cell.LSTMCell(n_units, **cell_init_args)

    outputs, state = tf.nn.dynamic_rnn(cell, inputs,
                                       sequence_length=sequence_length,
                                       dtype=tf.float32,
                                       **rnn_init_args)

    return outputs, state


def bi_lstm(inputs, n_units,
            sequence_length=None,
            cell_init_args=None,
            rnn_init_args=None):
    """

    Args:
        inputs: A 3D tensor with shape `[batch_size, max_steps, n_features]`
        n_units(int):
        sequence_length:
        cell_init_args:
        rnn_init_args:

    Returns:
        outputs, output_states
            == (output_fw, output_bw), (output_state_fw, output_state_bw)
            更详细的说明 ref: `tf.nn.bidirectional_dynamic_rnn`

        一般对 outputs 的处理方式是拼接双向的输出
            `outputs = tf.concat(outputs, axis=2)` which shape `[batch_size, max_steps, n_units*2]`

    """
    if cell_init_args is None:
        cell_init_args = {}
    if rnn_init_args is None:
        rnn_init_args = {}

    cell_fw = tf.nn.rnn_cell.LSTMCell(n_units, **cell_init_args)
    cell_bw = tf.nn.rnn_cell.LSTMCell(n_units, **cell_init_args)

    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
                                                             sequence_length=sequence_length,
                                                             **rnn_init_args)

    return outputs, output_states