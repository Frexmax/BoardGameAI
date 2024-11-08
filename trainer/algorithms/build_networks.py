import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, Flatten, Dense


def build_shared_model_simple(input_shape, output_shape, num_kernels, creg, dreg):
    input_layer = Input(shape=input_shape)
    conv0 = Conv2D(num_kernels, (3, 3), padding='same', activation='relu',
                   use_bias=True, data_format='channels_last',
                   kernel_regularizer=creg, bias_regularizer=creg)(input_layer)

    bn0 = BatchNormalization(axis=-1)(conv0)
    conv1 = Conv2D(num_kernels, (3, 3), padding='same', activation='relu',
                   use_bias=True, data_format='channels_last',
                   kernel_regularizer=creg, bias_regularizer=creg)(bn0)

    bn1 = BatchNormalization(axis=-1)(conv1)
    conv2 = Conv2D(num_kernels, (3, 3), padding='same', activation='relu',
                   use_bias=True, data_format='channels_last',
                   kernel_regularizer=creg, bias_regularizer=creg)(bn1)
    bn2 = BatchNormalization(axis=-1)(conv2)

    # CREATE POLICY HEAD
    policy_conv0 = Conv2D(8, (1, 1), padding='same', activation='relu',
                          use_bias=True, data_format='channels_last',
                          kernel_regularizer=creg, bias_regularizer=creg)(bn2)
    bn_pol0 = BatchNormalization(axis=-1)(policy_conv0)
    policy_flat1 = Flatten()(bn_pol0)
    policy_output = Dense(output_shape, activation='softmax', use_bias=True,
                          kernel_regularizer=dreg, bias_regularizer=dreg)(policy_flat1)

    # CREATE VALUE HEAD
    value_flat1 = Flatten()(bn2)
    value_dense1 = Dense(64, activation='relu', use_bias=True,
                         kernel_regularizer=dreg, bias_regularizer=dreg)(value_flat1)
    bn_val2 = BatchNormalization(axis=-1)(value_dense1)
    value_output = Dense(1, activation='tanh', use_bias=True, kernel_regularizer=dreg,
                         bias_regularizer=dreg)(bn_val2)

    model = tf.keras.Model(inputs=input_layer, outputs=[policy_output, value_output])
    return model


def build_shared_model_checkers(input_shape, output_shape, num_kernels, creg, dreg):
    input_layer = Input(shape=input_shape)
    conv0 = Conv2D(num_kernels, (3, 3), padding='same', activation='relu',
                   use_bias=True, data_format='channels_last',
                   kernel_regularizer=creg, bias_regularizer=creg)(input_layer)

    bn0 = BatchNormalization(axis=-1)(conv0)
    conv1 = Conv2D(num_kernels, (3, 3), padding='same', activation='relu',
                   use_bias=True, data_format='channels_last',
                   kernel_regularizer=creg, bias_regularizer=creg)(bn0)

    bn1 = BatchNormalization(axis=-1)(conv1)
    conv2 = Conv2D(num_kernels, (3, 3), padding='same', activation='relu',
                   use_bias=True, data_format='channels_last',
                   kernel_regularizer=creg, bias_regularizer=creg)(bn1)

    bn2 = BatchNormalization(axis=-1)(conv2)

    conv3 = Conv2D(num_kernels, (3, 3), padding='same', activation='relu',
                   use_bias=True, data_format='channels_last',
                   kernel_regularizer=creg, bias_regularizer=creg)(bn2)
    bn3 = BatchNormalization(axis=-1)(conv3)

    # CREATE POLICY HEAD
    policy_conv1 = Conv2D(num_kernels, (3, 3), padding='same', activation='relu',
                          use_bias=True, data_format='channels_last',
                          kernel_regularizer=creg, bias_regularizer=creg)(bn3)
    bn_pol1 = BatchNormalization(axis=-1)(policy_conv1)
    policy_conv2 = Conv2D(8, (1, 1), padding='same', activation='relu',
                          use_bias=True, data_format='channels_last',
                          kernel_regularizer=creg, bias_regularizer=creg)(bn_pol1)
    bn_pol2 = BatchNormalization(axis=-1)(policy_conv2)
    policy_flat1 = Flatten()(bn_pol2)
    policy_output = Dense(output_shape, activation='softmax', use_bias=True,
                          kernel_regularizer=dreg, bias_regularizer=dreg)(policy_flat1)

    # CREATE VALUE HEAD
    value_conv1 = Conv2D(1, (1, 1), padding='same', activation='relu',
                         use_bias=True, data_format='channels_last',
                         kernel_regularizer=creg, bias_regularizer=creg)(bn3)
    bn_val1 = BatchNormalization(axis=-1)(value_conv1)
    value_flat1 = Flatten()(bn_val1)
    value_dense1 = Dense(64, activation='relu', use_bias=True,
                         kernel_regularizer=dreg, bias_regularizer=dreg)(value_flat1)
    bn_val2 = BatchNormalization(axis=-1)(value_dense1)
    value_output = Dense(1, activation='tanh', use_bias=True, kernel_regularizer=dreg,
                         bias_regularizer=dreg)(bn_val2)

    model = tf.keras.Model(inputs=input_layer, outputs=[policy_output, value_output])
    return model
