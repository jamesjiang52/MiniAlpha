import numpy as np


class Board:
    def __init__(self):
        # bitboards go from a1 (MSB) to e5 (LSB) in row major order
        
        # starting position
        self.pieces = np.array([
            0b0010000000000000000000000,  # white king
            0b0100100000000000000000000,  # white knights
            0b1001000000000000000000000,  # white bishops
            0b0000011111000000000000000,  # white pawns
            0b0000000000000000000000100,  # black king
            0b0000000000000000000001001,  # black knights
            0b0000000000000000000010010,  # black bishops
            0b0000000000000001111100000   # black pawns
        ])
        
        # current color to move
        # set LSB for white, second-LSB for black
        self.color = 0b0000000000000000000000001
        
        # count of total number of moves (just a regular int)
        self.total_move_count = 0b0000000000000000000000000
        
        # TODO: find a way to efficiently keep track of repetitions
        # self.repetition_count = 0
        
        # count of number of moves since last pawn move or capture (just a regular int)
        self.no_progress_count = 0b0000000000000000000000000


    def move(self, move):
        # TODO: write this function
        pass
        


board = Board()
print(board.get_position())
