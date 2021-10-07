import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization

import rules
from board import Board
import nn_cfg


gpus = tf.config.experimental.list_physical_devices('GPU')
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)


class NeuralNetwork(Model):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self._conv_block_init()
        self._residual_tower_init()
        self._policy_head_init()
        self._value_head_init()

    def call(self, x, training=False):
        batch_size = x.shape[0]

        # initial convolutional block
        x = self.bn1(self.conv1(x), training=training)
        x = tf.nn.relu(x)

        # residual tower
        for conv1, bn1, conv2, bn2 in self.residual_blocks:
            skip = x
            x = bn1(conv1(x), training=training)
            x = tf.nn.relu(x)
            x = bn2(conv2(x), training=training)
            x = x + skip
            x = tf.nn.relu(x)

        # policy head
        x_policy = self.bn_policy(self.conv_policy(x), training=training)
        x_policy = tf.nn.relu(x_policy)
        x_policy = tf.reshape(x_policy, (batch_size, -1))
        x_policy = self.fc_policy(x_policy)

        # value head
        x_value = self.bn_value(self.conv_value(x), training=training)
        x_value = tf.nn.relu(x_value)
        x_value = tf.reshape(x_value, (batch_size, -1))
        x_value = self.fc1_value(x_value)
        x_value = self.fc2_value(x_value)

        return x_policy, x_value

    def predict_board(self, board):
        return self(np.array([board.get_input_features()], dtype=np.float32), training=False)

    def _conv_block_init(self):
        self.conv1 = Conv2D(
            filters=nn_cfg.NUM_FILTERS,
            kernel_size=nn_cfg.KERNEL_SIZE,
            strides=nn_cfg.STRIDE,
            data_format="channels_first",
            padding="same",
        )
        self.bn1 = BatchNormalization(axis=1)

    def _residual_tower_init(self):
        self.residual_blocks = []
        for _ in range(nn_cfg.NUM_RES_BLOCKS):
            conv1 = Conv2D(
                filters=nn_cfg.NUM_FILTERS,
                kernel_size=nn_cfg.KERNEL_SIZE,
                strides=nn_cfg.STRIDE,
                data_format="channels_first",
                padding="same",
            )
            bn1 = BatchNormalization(axis=1)
            conv2 = Conv2D(
                filters=nn_cfg.NUM_FILTERS,
                kernel_size=nn_cfg.KERNEL_SIZE,
                strides=nn_cfg.STRIDE,
                data_format="channels_first",
                padding="same"
            )
            bn2 = BatchNormalization(axis=1)
            self.residual_blocks.append((conv1, bn1, conv2, bn2))

    def _policy_head_init(self):
        self.conv_policy = Conv2D(
            filters=nn_cfg.NUM_FILTERS_POLICY,
            kernel_size=nn_cfg.KERNEL_SIZE_POLICY,
            strides=nn_cfg.STRIDE,
            data_format="channels_first",
            padding="same",
        )
        self.bn_policy = BatchNormalization(axis=1)
        self.fc_policy = Dense(
            nn_cfg.NUM_POLICY_DENSE_LAYER_OUTPUT,
            name="policy_head"
        )

    def _value_head_init(self):
        self.conv_value = Conv2D(
            filters=nn_cfg.NUM_FILTERS_HEAD,
            kernel_size=nn_cfg.KERNEL_SIZE_HEAD,
            strides=nn_cfg.STRIDE,
            data_format="channels_first",
            padding="same",
        )
        self.bn_value = BatchNormalization(axis=1)
        self.fc1_value = Dense(
            nn_cfg.NUM_HEAD_DENSE_LAYER_OUTPUT_1,
            activation="relu"
        )
        self.fc2_value = Dense(
            nn_cfg.NUM_HEAD_DENSE_LAYER_OUTPUT_2,
            activation="tanh",
            name="value_head"
        )


if __name__ == "__main__":
    nn = NeuralNetwork()
    board = Board()
    print(nn.predict_board(board))
