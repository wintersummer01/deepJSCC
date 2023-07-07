import math
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
from keras import layers, Model
import sionna as sn

# Model
class jsccEncoder_tfc(layers.Layer):
    def __init__(self, filter, k):
        super().__init__()
        self.k = k
        self.conv1 = tfc.layers.SignalConv2D(filters=16, kernel_support=5, strides_down=2, padding='same_zeros')
        self.prelu1 = layers.PReLU()
        self.conv2 = tfc.layers.SignalConv2D(filters=32, kernel_support=5, strides_down=2, padding='same_zeros')
        self.prelu2 = layers.PReLU()
        self.conv3 = tfc.layers.SignalConv2D(filters=32, kernel_support=5, padding='same_zeros')
        self.prelu3 = layers.PReLU()
        self.conv4 = tfc.layers.SignalConv2D(filters=32, kernel_support=5, padding='same_zeros')
        self.prelu4 = layers.PReLU()
        self.conv5 = tfc.layers.SignalConv2D(filters=filter, kernel_support=5, padding='same_zeros')
        self.prelu5 = layers.PReLU()
        
    def __call__(self, inputs):
        x = inputs/256 # Normalizing range [0, 256] into [0, 1]
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = self.conv5(x)
        x = self.prelu5(x)
        x = tf.reshape(x, (2, -1)) # Flattening
        z = tf.complex(x[0], x[1]) # Modulization
        z = self.encoderNormalizer(z) # Normalizing due to power constraint
        return z
    
    # Normalization function
    def encoderNormalizer(self, z):
        norm = tf.math.sqrt(tf.reduce_sum(tf.math.conj(z) * z))
        return z * math.sqrt(self.k) / norm # Assume power constraint is 1
    
    
    
class jsccDecoder_tfc(layers.Layer):
    def __init__(self, filter):
        super().__init__()
        self.filter = int(filter)
        self.trans_conv5 = tfc.layers.SignalConv2D(filters=32, kernel_support=5, padding='same_zeros')
        self.prelu5 = layers.PReLU()
        self.trans_conv4 = tfc.layers.SignalConv2D(filters=32, kernel_support=5, padding='same_zeros')
        self.prelu4 = layers.PReLU()
        self.trans_conv3 = tfc.layers.SignalConv2D(filters=32, kernel_support=5, padding='same_zeros')
        self.prelu3 = layers.PReLU()
        self.trans_conv2 = tfc.layers.SignalConv2D(filters=16, kernel_support=5, strides_up=2, padding='same_zeros')
        self.prelu2 = layers.PReLU()
        self.trans_conv1 = tfc.layers.SignalConv2D(filters=3, kernel_support=5, strides_up=2, padding='same_zeros')
        self.sigmoid = tf.keras.activations.sigmoid
        
    def __call__(self, outputs):
        x = tf.concat([tf.math.real(outputs), tf.math.imag(outputs)], 0) # Demodulization
        x = tf.reshape(x, [1, self.filter, self.filter, -1]) # Shaping
        x = self.trans_conv5(x)
        x = self.prelu5(x)
        x = self.trans_conv4(x)
        x = self.prelu4(x)
        x = self.trans_conv3(x)
        x = self.prelu3(x)
        x = self.trans_conv2(x)
        x = self.prelu2(x)
        x = self.trans_conv1(x)
        x = self.sigmoid(x)
        x = x*256 # Normalizing range [0, 256] into [0, 1]
        return x
    
    
    
class jsccEnd2End_tfc(Model):
    def __init__(self, shape, bw, SNR):
        super().__init__()
        k = np.prod(shape)*bw
        self.noise = sn.utils.ebnodb2no(SNR, num_bits_per_symbol=1, coderate=1)
        self.encoder = jsccEncoder_tfc(k*2/((shape[0]/4)**2), k)
        self.channel = sn.channel.AWGN()
        self.decoder = jsccDecoder_tfc(shape[0]/4)
        
    def call(self, x):
        z = self.encoder(x)
        z_hat = self.channel([z, self.noise])
        x_hat = self.decoder(z_hat)
        return x_hat