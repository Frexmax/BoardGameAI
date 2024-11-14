import tensorflow as tf
from keras.src.layers import Input, Conv2D, BatchNormalization, Flatten, Dense


def build_shared_model_simple(input_shape, output_shape, num_kernels, creg, dreg):
    """
    Create a two-headed model, with one action probability output head (policy head)
    and the other state value head (value head).
    The number of filters in each convolutional layer is parametrized, just like the kernel and bias
    regularizes. The output of these layers is also later batch normalized.

    This model type is used for simple games, such as Tic-Tac-Toe and Connect4

    :param input_shape: shape of the model input
    :param output_shape: shape of model output - number of output neurons
    :param num_kernels: number of filters in each convolutional layer
    :param creg: type of kernel regularization e.g. L2 regularization
    :param dreg: type of bias regularization e.g. L2 regularization
    :return: tensorflow model
    """

    # ================================= BACKBONE ================================= #

    input_layer = Input(shape=input_shape)

    # Layer 1 - convolutional
    conv0 = Conv2D(num_kernels, (3, 3), padding='same', activation='relu',
                   use_bias=True, data_format='channels_last',
                   kernel_regularizer=creg, bias_regularizer=creg)(input_layer)
    bn0 = BatchNormalization(axis=-1)(conv0)

    # Layer 2 - convolutional
    conv1 = Conv2D(num_kernels, (3, 3), padding='same', activation='relu',
                   use_bias=True, data_format='channels_last',
                   kernel_regularizer=creg, bias_regularizer=creg)(bn0)
    bn1 = BatchNormalization(axis=-1)(conv1)

    # Layer 3 - convolutional
    conv2 = Conv2D(num_kernels, (3, 3), padding='same', activation='relu',
                   use_bias=True, data_format='channels_last',
                   kernel_regularizer=creg, bias_regularizer=creg)(bn1)
    bn2 = BatchNormalization(axis=-1)(conv2)

    # ================================= HEADS ================================= #

    # Create policy head
    policy_conv0 = Conv2D(8, (1, 1), padding='same', activation='relu',
                          use_bias=True, data_format='channels_last',
                          kernel_regularizer=creg, bias_regularizer=creg)(bn2)
    bn_pol0 = BatchNormalization(axis=-1)(policy_conv0)
    policy_flat1 = Flatten()(bn_pol0)
    policy_output = Dense(output_shape, activation='softmax', use_bias=True,
                          kernel_regularizer=dreg, bias_regularizer=dreg)(policy_flat1)

    # Create value head
    value_flat1 = Flatten()(bn2)
    value_dense1 = Dense(64, activation='relu', use_bias=True,
                         kernel_regularizer=dreg, bias_regularizer=dreg)(value_flat1)
    bn_val2 = BatchNormalization(axis=-1)(value_dense1)
    value_output = Dense(1, activation='tanh', use_bias=True, kernel_regularizer=dreg,
                         bias_regularizer=dreg)(bn_val2)

    model = tf.keras.Model(inputs=input_layer, outputs=[policy_output, value_output])
    return model


def build_shared_model_checkers(input_shape, output_shape, num_kernels, creg, dreg):
    """
    Create a two-headed model, with one action probability output head (policy head)
    and the other state value head (value head).
    The number of filters in each convolutional layer is parametrized, just like the kernel and bias
    regularizes. The output of these layers is also later batch normalized.

    This model type is used for checkers,
     with an extra 3 convolutional layers compared to the model for simple games:
        - one in the model backbone
        - one in the policy head
        - one in the value head

    :param input_shape: shape of the model input
    :param output_shape: shape of model output - number of output neurons
    :param num_kernels: number of filters in each convolutional layer
    :param creg: type of kernel regularization e.g. L2 regularization
    :param dreg: type of bias regularization e.g. L2 regularization
    :return: tensorflow model
    """

    # ================================= BACKBONE ================================= #

    input_layer = Input(shape=input_shape)

    # Layer 1 - convolutional
    conv0 = Conv2D(num_kernels, (3, 3), padding='same', activation='relu',
                   use_bias=True, data_format='channels_last',
                   kernel_regularizer=creg, bias_regularizer=creg)(input_layer)

    bn0 = BatchNormalization(axis=-1)(conv0)

    # Layer 2 - convolutional
    conv1 = Conv2D(num_kernels, (3, 3), padding='same', activation='relu',
                   use_bias=True, data_format='channels_last',
                   kernel_regularizer=creg, bias_regularizer=creg)(bn0)

    bn1 = BatchNormalization(axis=-1)(conv1)

    # Layer 3 - convolutional
    conv2 = Conv2D(num_kernels, (3, 3), padding='same', activation='relu',
                   use_bias=True, data_format='channels_last',
                   kernel_regularizer=creg, bias_regularizer=creg)(bn1)

    bn2 = BatchNormalization(axis=-1)(conv2)

    # Layer 4 - convolutional
    conv3 = Conv2D(num_kernels, (3, 3), padding='same', activation='relu',
                   use_bias=True, data_format='channels_last',
                   kernel_regularizer=creg, bias_regularizer=creg)(bn2)
    bn3 = BatchNormalization(axis=-1)(conv3)

    # ================================= HEADS ================================= #

    # Create policy head
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

    # Create value head
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
