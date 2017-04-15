
import tensorflow as tf
from tensorflow.contrib.layers import convolution2d, convolution2d_transpose, xavier_initializer

import math

def ema_ae(input_image):
    """
    Autoencoder with shared weights.
    :param input_image: Original image.
    :return: (output_image, embedding_tensor)
    """
    with tf.variable_scope('autoencoder'):
        out = input

        pad = 'SAME'

        #####################
        ###    ENCODER    ###
        #####################

        with tf.variable_scope('conv1'):
            out = convolution2d(inputs=input_image, num_outputs=1, kernel_size=3, stride=1, padding=pad, rate=1,
                                activation_fn=tf.nn.relu, weights_initializer=xavier_initializer())

        with tf.variable_scope('conv2'):
            out = convolution2d(inputs=out, num_outputs=1, kernel_size=3, stride=2, padding=pad, rate=1,
                                activation_fn=tf.nn.relu, weights_initializer=xavier_initializer())

        with tf.variable_scope('conv3'):
            out = convolution2d(inputs=out, num_outputs=16, kernel_size=3, stride=1, padding=pad, rate=1,
                                activation_fn=tf.nn.relu, weights_initializer=xavier_initializer())

        with tf.variable_scope('conv4'):
            out = convolution2d(inputs=out, num_outputs=16, kernel_size=3, stride=2, padding=pad, rate=1,
                                activation_fn=tf.nn.relu, weights_initializer=xavier_initializer())

        with tf.variable_scope('conv5'):
            out = convolution2d(inputs=out, num_outputs=32, kernel_size=3, stride=1, padding=pad, rate=1,
                                activation_fn=tf.nn.relu, weights_initializer=xavier_initializer())

        with tf.variable_scope('conv6'):
            embedding_tensor = convolution2d(inputs=out, num_outputs=32, kernel_size=3, stride=2, padding=pad, rate=1,
                                activation_fn=tf.nn.relu, weights_initializer=xavier_initializer())

        #####################
        ###    DECODER    ###
        #####################

        with tf.variable_scope('conv6'):
            out = convolution2d_transpose(inputs=embedding_tensor, num_outputs=32, kernel_size=3, stride=2, padding=pad,
                                activation_fn=tf.nn.relu, weights_initializer=xavier_initializer())

        with tf.variable_scope('conv5'):
            out = convolution2d_transpose(inputs=out, num_outputs=32, kernel_size=3, stride=1, padding=pad,
                                activation_fn=tf.nn.relu, weights_initializer=xavier_initializer())

        with tf.variable_scope('conv4'):
            out = convolution2d_transpose(inputs=out, num_outputs=16, kernel_size=3, stride=2, padding=pad,
                                activation_fn=tf.nn.relu, weights_initializer=xavier_initializer())

        with tf.variable_scope('conv3'):
            out = convolution2d_transpose(inputs=out, num_outputs=16, kernel_size=3, stride=1, padding=pad,
                                activation_fn=tf.nn.relu, weights_initializer=xavier_initializer())

        with tf.variable_scope('conv2'):
            out = convolution2d_transpose(inputs=out, num_outputs=1, kernel_size=3, stride=2, padding=pad,
                                activation_fn=tf.nn.relu, weights_initializer=xavier_initializer())

        with tf.variable_scope('conv1'):
            output_image = convolution2d_transpose(inputs=out, num_outputs=1, kernel_size=3, stride=1, padding=pad,
                                activation_fn=tf.nn.relu, weights_initializer=xavier_initializer())

        return output_image, embedding_tensor