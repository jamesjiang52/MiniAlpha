import os
import copy
import time
import multiprocessing as mp
import queue

from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers

from board import Board
from cfg import load_cfg
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
            tree.playout(cpuct)

        move, policy, tree = tree.select_move(temp=1)
        data.append((copy.deepcopy(board).get_input_features().transpose(1, 2, 0), policy))
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

'''
def generate_worker(in_queue, out_queue, model_in_queue, model_out_queue, num_playouts, cpuct):
    """
    The worker function to generate games. The items in in_queue
        are unused, and the items in out_queue are the lists
        returned by generate_game.
    """
    try:
        while True:
            in_queue.get(block=False)
            data = generate_game(model_in_queue, model_out_queue, num_playouts, cpuct)
            out_queue.put(data)
    except queue.Empty:
        return


def worker_manager(done_queue, out_queue, num_games):
    data = []
    gen_loop = tqdm(range(num_games))
    for i in gen_loop:
        gen_loop.set_description(f'Generating game {i + 1}/{num_games}')
        item = out_queue.get()
        data.extend(item)

    done_queue.put(data)


def generate_training_data(model, num_playouts=1600, num_games=100, cpuct=1, num_workers=6):
    in_queue = mp.Queue()
    out_queue = mp.Queue()
    done_queue = mp.Queue()

    model_in_queues = [mp.Queue() for _ in range(num_workers)]
    model_out_queues = [mp.Queue() for _ in range(num_workers)]

    workers = [mp.Process(
        target=generate_worker,
        args=(in_queue, out_queue, model_in_queues[i], model_out_queues[i], num_playouts, cpuct)
    ) for i in range(num_workers)]
    manager = mp.Process(target=worker_manager, args=(done_queue, out_queue, num_games))

    for _ in range(num_games):
        in_queue.put(None)

    for worker in workers:
        worker.start()
    manager.start()

    # neural network evaluation
    while True:
        for i, (q1, q2) in enumerate(zip(model_in_queues, model_out_queues)):
            try:
                item = q1.get(block=False)
                policy, value = model(np.expand_dims(item, axis=0))
                policy, value = tf.squeeze(policy).numpy(), tf.squeeze(value).numpy()
                q2.put((policy, value))
            except queue.Empty:
                pass
        try:
            data = done_queue.get(block=False)
            return data
        except queue.Empty:
            pass
'''


def generate_training_data(model, num_playouts=1600, num_games=100, cpuct=1):
    data = []
    gen_loop = tqdm(range(num_games))
    for i in gen_loop:
        gen_loop.set_description(f'Generating game {i + 1}/{num_games}')
        data.extend(generate_game(model, num_playouts, cpuct))
    return data


def train():
    cfg_path = os.path.join(os.path.dirname(__file__), 'configs', 'default.yaml')
    cfg = load_cfg(cfg_path)

    model = NeuralNetwork(cfg)
    optimizer = optimizers.Adam(learning_rate=1e-3)

    @tf.function(jit_compile=True)
    def train_step(x, true_policy, true_value, step):
        with tf.GradientTape() as tape:
            pred_policy, pred_value = model(x, training=True)
            ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=true_policy, logits=pred_policy)
            mse_loss = tf.math.square(pred_value - true_value)
            loss = mse_loss + ce_loss

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return tf.reduce_mean(mse_loss), tf.reduce_mean(ce_loss)

    agent_num = 1
    while True:
        data = generate_training_data(
            model,
            num_playouts=cfg.NUM_PLAYOUTS_PER_GAME,
            num_games=cfg.NUM_GAMES_PER_ITERATION,
            cpuct=cfg.CPUCT
        )

        states = np.array([x[0] for x in data])
        policies = np.array([x[1] for x in data])
        values = np.array([x[2] for x in data]).astype(np.float32)

        num_steps = cfg.NUM_STEPS_PER_ITERATION
        train_loop = tqdm(range(num_steps))
        moving_loss1, moving_loss2 = 0, 0
        for s in train_loop:
            ind = np.random.randint(states.shape[0], size=cfg.BATCH_SIZE)
            loss1, loss2 = train_step(states[ind], policies[ind], values[ind], tf.constant(s))

            if s == 0:
                moving_loss1 = loss1
                moving_loss2 = loss2
            else:
                moving_loss1 = 0.97 * moving_loss1 + 0.03 * loss1
                moving_loss2 = 0.97 * moving_loss2 + 0.03 * loss2

            train_loop.set_description(f'Train step {s + 1}/{num_steps}, Loss1 {moving_loss1:.4f}, Loss2 {moving_loss2:.4f}')

        if agent_num % 3 == 0:
            model.save_weights(os.path.join(os.path.dirname(__file__), 'checkpoints', f'weights-agent{agent_num}.hdf5'))

        agent_num += 1


def main():
    train()


if __name__ == '__main__':
    main()
