import tensorflow as tf
import numpy as np








def batch_norm(input, momentum=0.9, epsilon=1e-5, train=True):
    output = tf.keras.layers.BatchNormalization(axis = -1, 
                                                        momentum = momentum,
                                                        epsilon = epsilon,
                                                        scale = True,
                                                        )(input)
    return output

def relu(input):
    output = tf.keras.activations.relu(input)
    return output

def conv_2d(input, kernel_dim, kernel_size, downsample):

    if downsample:
        input = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    output = tf.keras.layers.Conv2D(kernel_dim, kernel_size, strides, padding = padding)(input)
    return output


def conv_1x1(input, kernel_dim , kernel_size = 1, strides = 1):
    output = tf.keras.layers.Conv2D(kernel_dim, kernel_size, strides, padding = 'valid')(input)
    return output


def pointwise_block(input, kernel_dim):
    output = conv_1x1(input, kernel_dim)
    output = batch_norm(output)
    return output


def depthwise_block(input, kernel_size, depth_multiplier ):
    output = tf.keras.layers.DepthwiseConv2D(kernel_size = kernel_size, depth_multiplier = depth_multiplier, padding = 'same')(input)
    output = batch_norm(output)
    return output


def depthwise_separable_conv(input, kernel_dim, kernel_size, depth_multiplier):
    output = depthwise_block(input, kernel_size=kernel_size, depth_multiplier = depth_multiplier)
    output = relu(output)

    output = pointwise_block(output, kernel_dim= kernel_dim)
    output = relu(output)

    return output







def module_test():
    input_shape = (4, 28, 28, 64)
    sample = tf.random.normal(input_shape)
    print("Input shape : ",input_shape)
    print("conv_2d: {}".format(conv_2d(sample, 128, 3 , downsample= False).shape))
    print("conv_1x1: {}".format(conv_1x1(sample, 128).shape))
    print("depthwise_block: {}".format(depthwise_block(sample, 3, depth_multiplier = 1).shape))
    print("depthwise_separable_conv: {}".format(depthwise_separable_conv(sample,kernel_dim = 128, kernel_size= 3, depth_multiplier= 1).shape))

# module_test()