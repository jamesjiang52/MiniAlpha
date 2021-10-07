import rules


NUM_INPUT_PLANES = 83

NUM_FILTERS = 256
KERNEL_SIZE = 3
STRIDE = 1
NUM_RES_BLOCKS = 19

NUM_FILTERS_POLICY = 2
KERNEL_SIZE_POLICY = 1
NUM_POLICY_DENSE_LAYER_OUTPUT = rules.NUM_MOVES*rules.NUM_SQUARES

NUM_FILTERS_HEAD = 1
KERNEL_SIZE_HEAD = 1
NUM_HEAD_DENSE_LAYER_OUTPUT_1 = 256
NUM_HEAD_DENSE_LAYER_OUTPUT_2 = 1

MOMENTUM = 0.9
L2_REG_CONST = 0.0001
INITIAL_LEARNING_RATE = 0.01


# for learning rate annealing
# not sure how this is gonna be passed in
def _adapt_learning_rate(epoch):
    if epoch < 400000:
        return INITIAL_LEARNING_RATE
    if epoch < 600000:
        return 0.001
    return 0.0001