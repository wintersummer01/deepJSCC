import math
import numpy as np
import tensorflow as tf
from keras import layers, Model
import sionna as sn

# Model
class jsccEncoder(layers.Layer):
    def __init__(self, filter, k):
        super().__init__()
        self.k = k
        self.conv1 = layers.Conv2D(filters=16, kernel_size=5, strides=2, padding='same')
        self.prelu1 = layers.PReLU()
        self.conv2 = layers.Conv2D(filters=32, kernel_size=5, strides=2, padding='same')
        self.prelu2 = layers.PReLU()
        self.conv3 = layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same')
        self.prelu3 = layers.PReLU()
        self.conv4 = layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same')
        self.prelu4 = layers.PReLU()
        self.conv5 = layers.Conv2D(filters=filter, kernel_size=5, strides=1, padding='same')
        self.prelu5 = layers.PReLU()
        
    def __call__(self, inputs):
        x = tf.cast(inputs, tf.float32)/256 # Normalizing range [0, 256] into [0, 1]
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
    
    
    
class jsccDecoder(layers.Layer):
    def __init__(self, filter):
        super().__init__()
        self.filter = int(filter)
        self.trans_conv5 = layers.Conv2DTranspose(filters=32, kernel_size=5, strides=1, padding='same')
        self.prelu5 = layers.PReLU()
        self.trans_conv4 = layers.Conv2DTranspose(filters=32, kernel_size=5, strides=1, padding='same')
        self.prelu4 = layers.PReLU()
        self.trans_conv3 = layers.Conv2DTranspose(filters=32, kernel_size=5, strides=1, padding='same')
        self.prelu3 = layers.PReLU()
        self.trans_conv2 = layers.Conv2DTranspose(filters=16, kernel_size=5, strides=2, padding='same')
        self.prelu2 = layers.PReLU()
        self.trans_conv1 = layers.Conv2DTranspose(filters=3, kernel_size=5, strides=2, padding='same')
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
    
    
    
class jsccEnd2EndModel(Model):
    def __init__(self, shape, bw, SNR):
        super().__init__()
        k = np.prod(shape)*bw
        self.noise = sn.utils.ebnodb2no(SNR, num_bits_per_symbol=1, coderate=1)
        self.encoder = jsccEncoder(k*2/((shape[0]/4)**2), k)
        self.channel = sn.channel.AWGN()
        self.decoder = jsccDecoder(shape[0]/4)
        
    def call(self, x):
        z = self.encoder(x)
        z_hat = self.channel([z, self.noise])
        x_hat = self.decoder(z_hat)
        return x_hat