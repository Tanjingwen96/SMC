# The model is the DF model by Sirinam et al
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input

from keras.layers import Activation
from keras.layers import ELU
from keras.layers import Conv1D,Conv2D
from keras.layers import MaxPooling1D, MaxPooling2D, LayerNormalization, BatchNormalization
from keras.layers import Dropout, Flatten
from keras.layers.core import Flatten
from bdc_module import BDC_tf


def DF(input_shape=None, emb_size=None):
    # -----------------Entry flow -----------------
    input_data = Input(shape=input_shape)
    filter_num = ['None', 32, 64, 128, 256]
    kernel_size = ['None', 8, 8, 8, 8]
    conv_stride_size = ['None', 1, 1, 1, 1]
    pool_stride_size = ['None', 4, 4, 4, 4]
    pool_size = ['None', 8, 8, 8, 8]

    model = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1], input_shape=input_shape,
                     strides=conv_stride_size[1], padding='same',
                     name='block1_conv1')(input_data)
    # model = BatchNormalization(axis=-1)(model)
    model = ELU(alpha=1.0, name='block1_adv_act1')(model)
    model = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                     strides=conv_stride_size[1], padding='same',
                     name='block1_conv2')(model)
    # model = BatchNormalization(axis=-1)(model)
    model = ELU(alpha=1.0, name='block1_adv_act2')(model)
    model = MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
                           padding='same', name='block1_pool')(model)
    model = Dropout(0.1, name='block1_dropout')(model)

    model = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                     strides=conv_stride_size[2], padding='same',
                     name='block2_conv1')(model)
    # model = BatchNormalization()(model)
    model = Activation('relu', name='block2_act1')(model)

    model = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                     strides=conv_stride_size[2], padding='same',
                     name='block2_conv2')(model)
    # model = BatchNormalization()(model)
    model = Activation('relu', name='block2_act2')(model)
    model = MaxPooling1D(pool_size=pool_size[2], strides=pool_stride_size[3],
                           padding='same', name='block2_pool')(model)
    model = Dropout(0.1, name='block2_dropout')(model)

    model = Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                     strides=conv_stride_size[3], padding='same',
                     name='block3_conv1')(model)
    # model = BatchNormalization()(model)
    model = Activation('relu', name='block3_act1')(model)
    model = Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                     strides=conv_stride_size[3], padding='same',
                     name='block3_conv2')(model)
    # model = BatchNormalization()(model)
    model = Activation('relu', name='block3_act2')(model)
    model = MaxPooling1D(pool_size=pool_size[3], strides=pool_stride_size[3],
                           padding='same', name='block3_pool')(model)
    model = Dropout(0.1, name='block3_dropout')(model)

    model = Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                     strides=conv_stride_size[4], padding='same',
                     name='block4_conv1')(model)
    # model = BatchNormalization()(model)
    model = Activation('relu', name='block4_act1')(model)
    model = Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                     strides=conv_stride_size[4], padding='same',
                     name='block4_conv2')(model)
    # model = BatchNormalization()(model)
    model = Activation('relu', name='block4_act2')(model)
    model = MaxPooling1D(pool_size=pool_size[4], strides=pool_stride_size[4],
                           padding='same', name='block4_pool')(model)
    model = Dropout(0.1, name='block4_dropout')(model)

    flat = Flatten()(model)
    out = Dense(emb_size, name='FeaturesVec')(flat)
    BDC_layer = BDC_tf(is_vec=True, input_dim=[256, 20], dimension_reduction=256, activate='leaky_relu')(model)

    # shared_conv2 = Model(inputs=input_data, outputs=dense_layer)
    shared_conv2 = Model(inputs=input_data, outputs=[model, BDC_layer])

    return shared_conv2


def DF_global(input_shape=None, emb_size=None):
    # -----------------Entry flow -----------------
    input_data = Input(shape=input_shape)
    filter_num = ['None', 32, 64, 128, 256]
    kernel_size = ['None', 8, 8, 8, 8]
    conv_stride_size = ['None', 1, 1, 1, 1]
    pool_stride_size = ['None', 4, 4, 4, 4]
    pool_size = ['None', 8, 8, 8, 8]

    model = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1], input_shape=input_shape,
                     strides=conv_stride_size[1], padding='same',
                     name='block1_conv1')(input_data)
    model = BatchNormalization(axis=-1)(model)
    model = ELU(alpha=1.0, name='block1_adv_act1')(model)
    model = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                     strides=conv_stride_size[1], padding='same',
                     name='block1_conv2')(model)
    model = BatchNormalization(axis=-1)(model)
    model = ELU(alpha=1.0, name='block1_adv_act2')(model)
    model = MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
                           padding='same', name='block1_pool')(model)
    model = Dropout(0.5, name='block1_dropout')(model)

    model = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                     strides=conv_stride_size[2], padding='same',
                     name='block2_conv1')(model)
    model = BatchNormalization()(model)
    model = Activation('relu', name='block2_act1')(model)

    model = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                     strides=conv_stride_size[2], padding='same',
                     name='block2_conv2')(model)
    model = BatchNormalization()(model)
    model = Activation('relu', name='block2_act2')(model)
    model = MaxPooling1D(pool_size=pool_size[2], strides=pool_stride_size[3],
                           padding='same', name='block2_pool')(model)
    model = Dropout(0.3, name='block2_dropout')(model)

    model = Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                     strides=conv_stride_size[3], padding='same',
                     name='block3_conv1')(model)
    model = BatchNormalization()(model)
    model = Activation('relu', name='block3_act1')(model)
    model = Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                     strides=conv_stride_size[3], padding='same',
                     name='block3_conv2')(model)
    model = BatchNormalization()(model)
    model = Activation('relu', name='block3_act2')(model)
    model = MaxPooling1D(pool_size=pool_size[3], strides=pool_stride_size[3],
                           padding='same', name='block3_pool')(model)
    model = Dropout(0.2, name='block3_dropout')(model)

    model = Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                     strides=conv_stride_size[4], padding='same',
                     name='block4_conv1')(model)
    model = BatchNormalization()(model)
    model = Activation('relu', name='block4_act1')(model)
    model = Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                     strides=conv_stride_size[4], padding='same',
                     name='block4_conv2')(model)
    model = BatchNormalization()(model)
    model = Activation('relu', name='block4_act2')(model)
    model = MaxPooling1D(pool_size=pool_size[4], strides=pool_stride_size[4],
                           padding='same', name='block4_pool')(model)
    model = Dropout(0.1, name='block4_dropout')(model)

    flat = Flatten()(model)
    out = Dense(emb_size, name='FeaturesVec')(flat)
    BDC_layer = BDC_tf(is_vec=True, input_dim=[256, 20], dimension_reduction=256, activate='leaky_relu')(model)

    # shared_conv2 = Model(inputs=input_data, outputs=dense_layer)
    shared_conv2 = Model(inputs=input_data, outputs=[out, BDC_layer])

    return shared_conv2


# 生成器，线性映射，最小化WD
def generator_model(input_shape, emb_size=None):
    input_data = Input(shape=input_shape)
    data = Dense(emb_size, activation="linear", name='generate1')(input_data)
    data = Dense(emb_size, activation="linear", name='generate2')(data)
    generator = Model(inputs=input_data, outputs=data)
    return generator

# 鉴别器，评估映射后的WD
def discriminator_model(input_shape,class_num):
    input_data = Input(shape=input_shape)
    hidden_1 = Dense(64, activation="leaky_relu", name='dense_1')(input_data)
    hidden_2 = Dense(32, activation="relu", name='dense_2')(hidden_1)
    out = Dense(class_num, activation="sigmoid", name='dense_3')(hidden_2)
    discriminator = Model(inputs=input_data, outputs=out)

    return discriminator
