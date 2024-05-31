from keras.layers import ELU,ReLU,LeakyReLU
from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv2D,BatchNormalization
from keras.layers import Input
from keras import regularizers
import tensorflow as tf

class BDC_tf(tf.keras.layers.Layer):
    def __init__(self, is_vec=True, input_dim=[256, 20], dimension_reduction=None, activate='relu', **kwargs):
        super(BDC_tf, self).__init__(**kwargs)
        self.is_vec = is_vec
        self.dr = dimension_reduction
        self.activate = activate
        self.input_dim = input_dim[0]
        if self.dr is not None and self.dr != self.input_dim:
            if activate == 'relu':
                self.act = ReLU()
            elif activate == 'leaky_relu':
                self.act = LeakyReLU(0.1)
            else:
                self.act = ReLU()
            self.conv_dr_block = Sequential([
                Conv2D(self.dr, kernel_size=1, strides=1, use_bias=False),
                BatchNormalization(),
                self.act
            ])
        output_dim = self.dr if self.dr else self.input_dim
        if self.is_vec:
            self.output_dim = int(output_dim*(output_dim+1)/2)
        else:
            self.output_dim = int(output_dim*output_dim)
        self.temperature = tf.compat.v1.Variable(tf.math.log((1. / (2 * input_dim[1])) * tf.ones(1, 1)), trainable=True)

    def call(self, x):
        if self.dr is not None and self.dr != self.input_dim:
            x = self.conv_dr_block(x)
        x = BDCovpool(x, self.temperature)
        if self.is_vec:
            x = Triuvec(x)
        else:
            x = tf.reshape(x, [x.shape[0], -1])
        return x


def BDCovpool(x, t):
    x = tf.transpose(x, [0, 2, 1])
    batchSize, dim, M = x.shape
    I = tf.eye(num_rows=dim, num_columns=dim)
    I_M = tf.ones([dim, dim], dtype=x.dtype)
    x_pow2 = tf.matmul(x, tf.transpose(x, [0, 2, 1]))
    dcov = tf.matmul(I_M, x_pow2 * I) + tf.matmul(x_pow2 * I, I_M) - 2 * x_pow2
    dcov = tf.clip_by_value(dcov, clip_value_min=0.0, clip_value_max=tf.float32.max)
    dcov = tf.exp(t) * dcov
    dcov = tf.sqrt(dcov + 1e-5)
    t = dcov - 1. / dim * tf.matmul(dcov, I_M) - 1. / dim * tf.matmul(I_M, dcov) + 1. / (dim * dim) * tf.matmul(tf.matmul(I_M, dcov), I_M)
    return t



def Triuvec(x):
    batchSize, dim, dim = x.shape
    r = tf.reshape(x, (-1, dim * dim))
    I = tf.linalg.band_part(tf.ones([dim, dim]), 0, -1)
    I = tf.reshape(I, [-1])
    index = tf.where(tf.equal(I, 1))
    index = tf.convert_to_tensor(index)
    y = tf.gather(r, index,axis=1, batch_dims=0)
    y = tf.reduce_mean(y, 2)
    return y

