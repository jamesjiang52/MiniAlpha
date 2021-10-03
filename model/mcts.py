import numpy as np


# replace these two classes with the actual implementations
class Board:
    pass


class Model:
    pass


# declaration for typing
class Node:
    pass


class Edge:

    def __init__(self, p: float, action: int, parent_node: Node):
        """
        Edge class for the MCTS
        :param p: prior probability of selecting this edge
        :param action: the index of the action this edge represents
        :param parent_node: the (unique) node with this edge as its child
        """
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = p
        self.action = action
        self.parent = parent_node
        self.child = None


class Node:

    def __init__(self, board: Board, model: Model):
        """
        Node class for the MCTS
        :param board: the board with the current position at this node
        :param model: the model used for evaluation
        """
        self.board = board
        self.model = model
        self.parent_edge = None
        self.children = []
        self.value = None
        self.n_visits = 0

    def init_children(self) -> None:
        """
        Initialize the children using the model's predictions on
            the board, which includes the value of the current
            position and the probabilities of all actions
            to take
        :return: None
        """
        value, action_probs = self.model.evaluate(self.board)
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
        best_value = -1
        best_edge = None
        for edge in self.children:
            edge_value = edge.q + cpuct * edge.p * np.sqrt(self.n_visits) / (1 + edge.n)
            if edge_value > best_value:
                best_value = edge_value
                best_edge = edge
        return edge

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
            edge = choose_child_to_explore(cpuct)
            if edge.child is None:
                break
            else:
                curr_node = edge.child

        # expansion phase
        new_board = self.board.update(edge.action)
        new_node = Node(new_board, self.model)
        new_node.parent_edge = edge
        new_node.init_children()
        edge.child = new_node

        # backup phase
        curr_edge = edge
        value = new_node.value
        while True:
            edge.n += 1
            edge.w += value
            edge.q = edge.w / edge.n
            edge.parent.n_visits += 1

            curr_edge = edge.parent.parent_edge
            if curr_edge is None:
                break
