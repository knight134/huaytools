"""

"""

import tensorflow as tf
import tensorlayer as tl

logging = tf.logging


# def conv(rank, n_filter, kernel_size,
#          padding='valid',
#          data_format=None,
#          kernel_init=None):
#     """
#
#     Args:
#         rank(int): The rank of the convolution, e.g. "2" for 2D convolution.
#         n_filter(int): The dimensionality of the output space / The number of filters in the convolution.
#         kernel_size: A tuple/list of n integers (n >= 1), specifying the size of the convolution window.
#         padding(str): One of "valid" or "same", default is "valid"
#         data_format(str):
#
#     Returns:
#
#     """

def conv1d(inputs, kernel_size, out_channels,
           strides=1,
           activation=tf.nn.relu,
           use_bias=True,
           padding="VALID",
           W_init=tf.truncated_normal_initializer(stddev=0.02),
           W_init_args=None,
           b_init=tf.constant_initializer(value=0.0),
           b_init_args=None,
           data_format="NWC",
           name="conv1d",
           reuse=None):
    """
    1D 卷积

    Args:
        inputs: A 3D `Tensor` with [batch_size, max_length, in_channels]
        kernel_size:
        out_channels:
        strides(int):
        activation: default to use `tf.nn.relu`
        use_bias(bool):
        padding(str):
        W_init:
        W_init_args(dict):
        b_init:
        b_init_args(dict):
        data_format(str):
        name(str):
        reuse(bool):

    Returns:
        A 2D `Tensor` with [batch_size, max_length/strides, out_channels]

    """
    in_channels = tf.convert_to_tensor(inputs).get_shape()[-1].value

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)
    kernel_shape = kernel_size + (in_channels, out_channels)

    logging.info("Conv1dLayer: %s - kernel_shape: %s strides: %s padding: %s activation: %s" % (
        name, str(kernel_shape), str(strides), padding, activation.__name__))

    with tf.variable_scope(name, reuse=reuse):
        W = tf.get_variable(name='W', shape=kernel_shape, initializer=W_init, dtype=tf.float32,
                            **W_init_args)
        if use_bias:
            b = tf.get_variable(name='b', shape=kernel_shape[-1:], initializer=b_init, dtype=tf.float32,
                                **b_init_args)
            outputs = activation(
                tf.nn.bias_add(
                    tf.nn.conv1d(inputs, W, strides=strides, padding=padding, data_format=data_format), b))
        else:
            outputs = activation(
                tf.nn.conv2d(inputs, W, strides=strides, padding=padding, data_format=data_format))

    return outputs


def conv2d(inputs, kernel_size, out_channels,
           strides=(1, 1, 1, 1),
           activation=tf.nn.relu,
           use_bias=True,
           padding="VALID",
           W_init=tf.truncated_normal_initializer(stddev=0.02),
           W_init_args=None,
           b_init=tf.constant_initializer(value=0.0),
           b_init_args=None,
           data_format="NHWC",
           name="conv2d",
           reuse=None):
    """
    2D 卷积

    Args:
        inputs: A 4D `Tensor` with [batch_size, height, width, in_channels]
        kernel_size(int or tuple): A integer or a tuple/list with (k_height, k_width)
            特别的，当应用在 NLP 中时，一般置 kernel_size=(n_gram, embedding_size)
        out_channels(int): The number of the out channels
        strides(int or tuple): A integer or a tuple/list with length 4
        activation: default to use `tf.nn.relu`
        use_bias(bool):
        padding(str):
        W_init:
        W_init_args(dict):
        b_init:
        b_init_args(dict):
        data_format(str):
        name(str):
        reuse(bool):

    Returns:
        A 4D `Tensor` with [batch_size, height/strides[1], width/strides[2], out_channels]
    """
    in_channels = tf.convert_to_tensor(inputs).get_shape()[-1].value

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 2
    kernel_shape = kernel_size + (in_channels, out_channels)  # [kernel_h, kernel_w, in_channels, out_channels]

    if isinstance(strides, int):
        strides = (strides,) * 4

    logging.info("Conv2dLayer: %s - kernel_shape: %s strides: %s padding: %s activation: %s" % (
        name, str(kernel_shape), str(strides), padding, activation.__name__))

    with tf.variable_scope(name, reuse=reuse):
        W = tf.get_variable(name='W', shape=kernel_shape, initializer=W_init, dtype=tf.float32,
                            **W_init_args)
        if use_bias:
            b = tf.get_variable(name='b', shape=kernel_shape[-1:], initializer=b_init, dtype=tf.float32,
                                **b_init_args)
            outputs = activation(
                tf.nn.bias_add(
                    tf.nn.conv2d(inputs, W, strides=strides, padding=padding, data_format=data_format), b))
        else:
            outputs = activation(
                tf.nn.conv2d(inputs, W, strides=strides, padding=padding, data_format=data_format))

    return outputs
