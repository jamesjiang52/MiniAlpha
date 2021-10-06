import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, ReLU, Add
from keras import regularizers
from keras import callbacks

import rules
import nn_cfg


class NeuralNetwork():
    def __init__(self):
        self.inputs = Input(
            shape=(nn_cfg.NUM_INPUT_PLANES, rules.BOARD_SIDE_LENGTH, rules.BOARD_SIDE_LENGTH),
            name="inputs"
        )

        layers = self._conv_block(self.inputs)
        for i in range(nn_cfg.NUM_RES_BLOCKS):
            layers = self._residual_block(layers)

        policy_out = self._policy_head(layers)
        value_out = self._value_head(layers)

        self.model = Model(
            inputs=[self.inputs],
            outputs=[policy_out, value_out]
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=nn_cfg.INITIAL_LEARNING_RATE,
                momentum=nn_cfg.MOMENTUM
            ),
            loss={
                "value_head": "mean_squared_error",
                "policy_head": "categorical_crossentropy"
            },
            loss_weights={
                "policy_head": 1,
                "value_head": 1
            }
        )

        #self.learning_rate_scheduler = callbacks.LearningRateScheduler(self._adapt_learning_rate)


    def _conv_block(self, layers):
        layers = Conv2D(
            filters=nn_cfg.NUM_FILTERS,
            kernel_size=nn_cfg.KERNEL_SIZE,
            strides=nn_cfg.STRIDE,
            data_format="channels_first",
            padding="same",
            kernel_regularizer=regularizers.l2(nn_cfg.L2_REG_CONST)
        )(layers)

        layers = BatchNormalization(axis=1)(layers)
        layers = ReLU()(layers)

        return layers


    def _residual_block(self, inputs):
        layers = Conv2D(
            filters=nn_cfg.NUM_FILTERS,
            kernel_size=nn_cfg.KERNEL_SIZE,
            strides=nn_cfg.STRIDE,
            data_format="channels_first",
            padding="same",
            kernel_regularizer=regularizers.l2(nn_cfg.L2_REG_CONST)
        )(inputs)

        layers = BatchNormalization(axis=1)(layers)
        layers = ReLU()(layers)

        layers = Conv2D(
            filters=nn_cfg.NUM_FILTERS,
            kernel_size=nn_cfg.KERNEL_SIZE,
            strides=nn_cfg.STRIDE,
            data_format="channels_first",
            padding="same",
            kernel_regularizer=regularizers.l2(nn_cfg.L2_REG_CONST)
        )(layers)

        layers = BatchNormalization(axis=1)(layers)
        layers = Add()([inputs, layers])
        layers = ReLU()(layers)

        return layers


    def _policy_head(self, layers):
        layers = Conv2D(
            filters=nn_cfg.NUM_FILTERS_POLICY,
            kernel_size=nn_cfg.KERNEL_SIZE_POLICY,
            strides=nn_cfg.STRIDE,
            data_format="channels_first",
            padding="same",
            kernel_regularizer=regularizers.l2(nn_cfg.L2_REG_CONST)
        )(layers)

        layers = BatchNormalization(axis=1)(layers)
        layers = ReLU()(layers)
        layers = Flatten()(layers)
        layers = Dense(
            nn_cfg.NUM_POLICY_DENSE_LAYER_OUTPUT,
            kernel_regularizer=regularizers.l2(nn_cfg.L2_REG_CONST),
            name="policy_head"
        )(layers)

        return layers


    def _value_head(self, layers):
        layers = Conv2D(
            filters=nn_cfg.NUM_FILTERS_HEAD,
            kernel_size=nn_cfg.KERNEL_SIZE_HEAD,
            strides=nn_cfg.STRIDE,
            data_format="channels_first",
            padding="same",
        )(layers)

        layers = BatchNormalization(axis=1)(layers)
        layers = ReLU()(layers)
        layers = Flatten()(layers)

        layers = Dense(
            nn_cfg.NUM_HEAD_DENSE_LAYER_OUTPUT_1,
            kernel_regularizer=regularizers.l2(nn_cfg.L2_REG_CONST)
        )(layers)

        layers = ReLU()(layers)

        layers = Dense(
            nn_cfg.NUM_HEAD_DENSE_LAYER_OUTPUT_2,
            activation="tanh",
            kernel_regularizer=regularizers.l2(nn_cfg.L2_REG_CONST),
            name="value_head"
        )(layers)

        return layers


        
if __name__ == "__main__":
    nn = NeuralNetwork()
