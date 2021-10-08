import os
import copy

from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers

from board import Board
from nn import NeuralNetwork
from mcts import Node


def generate_game(model, num_playouts=1600, cpuct=1):

    # one copy of the board to keep track of the game, and another
    # copy of the board for the MCTS to freely mutilate
    board = Board()
    tree = Node(Board(), model)

    # stores (state, policy) pairs
    data = []

    while True:
        for i in range(num_playouts):
            print(i)
            tree.playout(cpuct)

        move, policy, tree = tree.select_move(temp=1)
        data.append((copy.deepcopy(board).get_input_features(), policy))
        return_code = board.move(move)

        if return_code != 2:
            # mark the current player's previous positions as winning
            # and the other player's previous positions as losing (or
            # vice versa)
            data = data[::-1]
            for i, (state, policy) in enumerate(data):
                if i % 2 == 0:
                    data[i] = (state, policy, return_code)
                else:
                    data[i] = (state, policy, -1 * return_code)

            break

    return data[::-1]


def main():
    cfg_path = os.path.join(os.path.dirname(__file__), 'configs', 'default.yaml')
    model = NeuralNetwork(cfg_path)

    data = generate_game(model)
    print(data)


if __name__ == '__main__':
    main()
