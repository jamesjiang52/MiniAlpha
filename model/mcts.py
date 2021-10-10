import copy
from typing import Tuple
from dataclasses import dataclass

import numpy as np

from board import Board
from nn import NeuralNetwork


# declaration for typing
class Node:
    pass


@dataclass
class Edge:
    """
    Edge class for the MCTS
    :param p: prior probability of selecting this edge
    :param action: the index of the action this edge represents
    :param parent_node: the (unique) node with this edge as its child
    """
    p: float
    action: int
    parent: Node
    n: int = 0
    w: float = 0
    q: float = 0
    child: Node = None
    is_terminal: bool = False


class Node:

    def __init__(self, board: Board, model: NeuralNetwork):
        """
        Node class for the MCTS
        :param board: the board with the current position at this node
        :param model: the model for position evaluation
        """
        self.board = board
        self.model = model

        self.parent_edge = None
        self.children = []
        self.value = None
        self.n_visits = 0

    @staticmethod
    def softmax(x: np.ndarray):
        """
        Compute softmax activation of x
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def init_children(self) -> None:
        """
        Initialize the children using the model's predictions on
            the board, which includes the value of the current
            position and the probabilities of all actions
            to take
        :return: None
        """
        # self.in_queue.put(self.board.get_input_features().transpose(1, 2, 0))
        # action_probs, value = self.out_queue.get()

        action_probs, value = self.model.predict_board(self.board)

        # convert logit activations to probabilities
        action_probs = self.softmax(action_probs)

        self.children = [Edge(prob, i, self) for i, prob in enumerate(action_probs)]
        self.value = value

    def choose_child_to_explore(self, cpuct: float = 0.1) -> Edge:
        """
        Choose a child edge to explore; the search strategy initially
            prefers actions with high prior probability and low visit
            count, but asymptotically prefers actions with high action
            value
        :param cpuct: constant for PUCT; higher value of cpuct trades off
                      exploitation for more exploration
        :return: the edge to explore next
        """
        qs = np.array([edge.q for edge in self.children])
        ps = np.array([edge.p for edge in self.children])
        ns = np.array([edge.n for edge in self.children])

        best_index = np.argmax(qs + cpuct * ps * np.sqrt(self.n_visits) / (1 + ns))
        return self.children[best_index]

    def playout(self, cpuct: float = 0.1) -> None:
        """
        Perform a single iteration of MCTS starting from
            this node as the root
        :param cpuct: constant for PUCT; higher value of cpuct trades off
                      exploitation for more exploration
        :return: None
        """
        if not self.children:
            self.init_children()

        # selection phase
        curr_node = self
        while True:
            edge = curr_node.choose_child_to_explore(cpuct)
            if edge.child is None:
                break
            else:
                curr_node = edge.child

        # expansion phase
        if curr_node.parent_edge is not None and curr_node.parent_edge.is_terminal:
            # don't keep exploring if the game has already ended
            edge = curr_node.parent_edge
            new_node = curr_node
        else:
            new_board = copy.deepcopy(curr_node.board)
            return_code = new_board.move(edge.action)
            new_node = Node(new_board, self.model)
            new_node.parent_edge = edge
            new_node.init_children()
            edge.child = new_node

            # if the new board is the end of a game, mark the edge as terminal
            if return_code != 2:
                edge.is_terminal = True

        # backup phase
        curr_edge = edge
        value = new_node.value
        while True:
            curr_edge.n += 1
            curr_edge.w += value
            curr_edge.q = curr_edge.w / curr_edge.n
            curr_edge.parent.n_visits += 1

            curr_edge = curr_edge.parent.parent_edge
            if curr_edge is None:
                break

    def select_move(self, temp: float = 1) -> Tuple[int, np.ndarray, Node]:
        """
        Select a move to play based on the previous playouts;
            the move is probabilistic and based on the number of
            node visits in the MCTS tree.
        :param temp: Temperature parameter; a higher temperature favours
                     more exploration, and a temperature of zero means
                     no exploration (select the best move)
        :return: a tuple containing the move to play, the move probabilities,
                 and the subtree rooted at the next board position
        """
        if temp == 0:
            # deterministically select the node with the highest visit count
            move = np.argmax([edge.n for edge in self.children])
            policy = np.zeros(len(self.children))
            policy[move] = 1
        else:
            # probabilistically select a move in rough proportion to its
            # visit count
            policy = np.array([np.power(edge.n, 1 / temp) for edge in self.children])
            policy = policy / np.sum(policy)
            move = np.random.choice(np.arange(policy.shape[0]), p=policy)

        return move, policy, self.children[move].child
