import tensorflow as tf
from common import *


class Depthwise_separable_conv(tf.keras.layers.Layer):
    def __init__(self, kernel_dim, kernel_size, downsample, depth_multiplier ):
        super(Depthwise_separable_conv, self).__init__()

        if downsample:
            padding = 'same'
            strides = 2
        else:
            strides = 1
            padding = 'same'

        self.dw_conv1 = tf.keras.layers.DepthwiseConv2D(kernel_size = kernel_size,
                                                        depth_multiplier = depth_multiplier,
                                                        strides = strides,
                                                        padding = padding)

        self.bn1 = tf.keras.layers.BatchNormalization(axis = -1, 
                                                        momentum = 0.9,
                                                        epsilon = 1e-5,
                                                        scale = True,
                                                        )
        self.ac_relu1 = tf.keras.activations.relu
        
        self.pw_conv1 = tf.keras.layers.Conv2D( filters = kernel_dim,
                                                kernel_size = 1,
                                                strides = 1,
                                                padding = 'valid')

        self.bn2 = tf.keras.layers.BatchNormalization(axis = -1, 
                                                        momentum = 0.9,
                                                        epsilon = 1e-5,
                                                        scale = True,
                                                        )      
        self.ac_relu2 = tf.keras.activations.relu  
                                                

    def call(self, input):
        output = self.dw_conv1(input)
        output = self.bn1(output)
        output = self.ac_relu1(output)
        output = self.pw_conv1(output)
        output = self.bn2(output)
        output = self.ac_relu2(output)
        return output



class MobileNet_v1(tf.keras.models.Model):
    def __init__(self):
        super(MobileNet_v1, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')
        self.dws_conv1 = Depthwise_separable_conv(kernel_dim=64, kernel_size=3, downsample = False, depth_multiplier=1)
        self.dws_conv2 = Depthwise_separable_conv(kernel_dim=128, kernel_size=3, downsample = True, depth_multiplier=1)
        self.dws_conv3 = Depthwise_separable_conv(kernel_dim=128, kernel_size=3, downsample = False, depth_multiplier=1)
        self.dws_conv4 = Depthwise_separable_conv(kernel_dim=256, kernel_size=3, downsample = True, depth_multiplier=1)
        self.dws_conv5 = Depthwise_separable_conv(kernel_dim=256, kernel_size=3, downsample = False, depth_multiplier=1)
        self.dws_conv6 = Depthwise_separable_conv(kernel_dim=512, kernel_size=3, downsample = True, depth_multiplier=1)

        self.dws_conv7 = Depthwise_separable_conv(kernel_dim=512, kernel_size=3, downsample = False, depth_multiplier=1)
        self.dws_conv8 = Depthwise_separable_conv(kernel_dim=512, kernel_size=3, downsample = False, depth_multiplier=1)
        self.dws_conv9 = Depthwise_separable_conv(kernel_dim=512, kernel_size=3, downsample = False, depth_multiplier=1)
        self.dws_conv10 = Depthwise_separable_conv(kernel_dim=512, kernel_size=3, downsample = False, depth_multiplier=1)
        self.dws_conv11 = Depthwise_separable_conv(kernel_dim=512, kernel_size=3, downsample = False, depth_multiplier=1)

        self.dws_conv12 = Depthwise_separable_conv(kernel_dim=1024, kernel_size=3, downsample = True, depth_multiplier=1)
        self.dws_conv13 = Depthwise_separable_conv(kernel_dim=1024, kernel_size=3, downsample = False, depth_multiplier=1)

        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size = 7, strides = 1)

        self.fc_layer = tf.keras.Sequential(name = "FC_Layer")
        self.fc_layer.add(tf.keras.layers.Dense(units = 1000, activation = 'softmax'))
        self.fc_layer.add(tf.keras.layers.Dense(units = 80, activation = 'softmax'))

    def call(self, input):
        output = self.conv1(input)
        output = self.dws_conv1(output)
        output = self.dws_conv2(output)
        output = self.dws_conv3(output)
        output = self.dws_conv4(output)
        output = self.dws_conv5(output)
        output = self.dws_conv6(output)

        output = self.dws_conv7(output)
        output = self.dws_conv8(output)
        output = self.dws_conv9(output) 
        output = self.dws_conv10(output)
        output = self.dws_conv11(output)

        output = self.dws_conv12(output)
        output = self.dws_conv13(output)
        
        output = self.avg_pool(output)
        output = self.fc_layer(output)
        return output
    
    def model(self, input_shape):
        input = tf.keras.Input(shape=input_shape)
        return tf.keras.models.Model(inputs=[input], outputs=self.call(input))


input_shape = (224, 224, 3)
sample = tf.random.normal(input_shape)


mobile = MobileNet_v1()
mobile.model(input_shape).summary()

from tensorflow.python.client import device_lib
device_lib.list_local_devices()
