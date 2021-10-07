import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, ReLU, add
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers

import rules
from board import Board
import nn_cfg


class NeuralNetwork(Model):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.layer_list = []
        self.policy_head_layers = []
        self.value_head_layers = []

        self._conv_block_init()
        for i in range(nn_cfg.NUM_RES_BLOCKS):
            self._residual_block_init()
        self._policy_head_init()
        self._value_head_init()


    def call(self, x, training=False):
        for layer in self.layer_list:
                x = layer(x)

        policy_out = x
        value_out = x

        for layer in self.policy_head_layers:
            policy_out = layer(policy_out)

        for layer in self.value_head_layers:
            value_out = layer(value_out)

        return policy_out, value_out
        
        
    def predict_board(self, board):
        return super(NeuralNetwork, self).predict(np.array([board.get_input_features()], dtype="float32"))


    def _conv_block_init(self):
        self.layer_list.append(
            Conv2D(
                filters=nn_cfg.NUM_FILTERS,
                kernel_size=nn_cfg.KERNEL_SIZE,
                strides=nn_cfg.STRIDE,
                data_format="channels_first",
                padding="same",
                kernel_regularizer=regularizers.l2(nn_cfg.L2_REG_CONST)
            )
        )

        self.layer_list.append(BatchNormalization(axis=1))
        self.layer_list.append(ReLU())


    def _residual_block_init(self):
        inputs = self.layer_list[-1]
    
        self.layer_list.append(
            Conv2D(
                filters=nn_cfg.NUM_FILTERS,
                kernel_size=nn_cfg.KERNEL_SIZE,
                strides=nn_cfg.STRIDE,
                data_format="channels_first",
                padding="same",
                kernel_regularizer=regularizers.l2(nn_cfg.L2_REG_CONST)
            )
        )

        self.layer_list.append(BatchNormalization(axis=1))
        self.layer_list.append(ReLU())

        self.layer_list.append(
            Conv2D(
                filters=nn_cfg.NUM_FILTERS,
                kernel_size=nn_cfg.KERNEL_SIZE,
                strides=nn_cfg.STRIDE,
                data_format="channels_first",
                padding="same",
                kernel_regularizer=regularizers.l2(nn_cfg.L2_REG_CONST)
            )
        )

        self.layer_list.append(BatchNormalization(axis=1))
        self.layer_list.append(add([self.layer_list[-1], inputs]))
        self.layer_list.append(ReLU())


    def _policy_head_init(self):
        self.policy_head_layers.append(
            Conv2D(
                filters=nn_cfg.NUM_FILTERS_POLICY,
                kernel_size=nn_cfg.KERNEL_SIZE_POLICY,
                strides=nn_cfg.STRIDE,
                data_format="channels_first",
                padding="same",
                kernel_regularizer=regularizers.l2(nn_cfg.L2_REG_CONST)
            )
        )

        self.policy_head_layers.append(BatchNormalization(axis=1))
        self.policy_head_layers.append(ReLU())
        self.policy_head_layers.append(Flatten())
        self.policy_head_layers.append(
            Dense(
                nn_cfg.NUM_POLICY_DENSE_LAYER_OUTPUT,
                kernel_regularizer=regularizers.l2(nn_cfg.L2_REG_CONST),
                name="policy_head"
            )
        )


    def _value_head_init(self):
        self.value_head_layers.append(
            Conv2D(
                filters=nn_cfg.NUM_FILTERS_HEAD,
                kernel_size=nn_cfg.KERNEL_SIZE_HEAD,
                strides=nn_cfg.STRIDE,
                data_format="channels_first",
                padding="same",
            )
        )

        self.value_head_layers.append(BatchNormalization(axis=1))
        self.value_head_layers.append(ReLU())
        self.value_head_layers.append(Flatten())

        self.value_head_layers.append(
            Dense(
                nn_cfg.NUM_HEAD_DENSE_LAYER_OUTPUT_1,
                kernel_regularizer=regularizers.l2(nn_cfg.L2_REG_CONST)
            )
        )

        self.value_head_layers.append(ReLU())

        self.value_head_layers.append(
            Dense(
                nn_cfg.NUM_HEAD_DENSE_LAYER_OUTPUT_2,
                activation="tanh",
                kernel_regularizer=regularizers.l2(nn_cfg.L2_REG_CONST),
                name="value_head"
            )
        )



if __name__ == "__main__":
    nn = NeuralNetwork()
    nn.compile(
        loss={
            "policy_head": "categorical_crossentropy",
            "value_head": "mean_squared_error"
        },
        optimizer=optimizers.SGD(),
        loss_weights={
            "policy_head": 1,
            "value_head": 1
        }	
	)
    board = Board()
    nn.predict_board(board)
