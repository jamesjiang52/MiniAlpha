import sys
import numpy as np
import rules


class Board:
    def __init__(self):
        # starting position
        self.pieces = rules.start_position

        # current color to move
        # set 0 for white, 1 for black
        self.color = 0

        # count of total number of moves
        self.total_move_count = 0

        # track repetition of current move
        self.first_repetition = 0
        self.second_repetition = 0

        # count of number of moves since last pawn move or capture
        self.no_progress_move_count = 0

        # history of moves
        self.history = np.append(np.zeros((rules.HISTORY_LENGTH - 1, rules.NUM_PIECE_TYPES_TOTAL, rules.BOARD_SIDE_LENGTH, rules.BOARD_SIDE_LENGTH)), [self.pieces], axis=0)
        
        # repetition history
        self.first_repetition_history = [0]*rules.HISTORY_LENGTH
        self.second_repetition_history = [0]*rules.HISTORY_LENGTH
        
        
    def get_input_features(self):
        """
        Returns a numpy array with shape (83, 5, 5).

        The first 80 planes are the piece position and repetition planes for the last 8 moves (10 planes per move).
        The current color, the total move count, and the no-progress move count comprise the other 3 planes.
        """
        features = []
        for t in range(rules.HISTORY_LENGTH):
            features.extend(self.history[t])
            features.append(np.array([[self.first_repetition_history[t], 1 - self.first_repetition_history[t], 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))
            features.append(np.array([[self.second_repetition_history[t], 1 - self.second_repetition_history[t], 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))
            
        features.append(np.array([[self.color, 1 - self.color, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))
        features.append(np.reshape(np.fromiter(bin(self.total_move_count)[2:].zfill(rules.NUM_SQUARES), dtype=int), (rules.BOARD_SIDE_LENGTH, rules.BOARD_SIDE_LENGTH)))
        features.append(np.reshape(np.fromiter(bin(self.no_progress_move_count)[2:].zfill(rules.NUM_SQUARES), dtype=int), (rules.BOARD_SIDE_LENGTH, rules.BOARD_SIDE_LENGTH)))
        
        print(np.array(features).shape)
        return np.array(features)


    def _get_attacking_squares(self, piece_idx):
        # TODO: write this function
        pass


    def move(self, move):
        """
        Returns:
            -1 if move results in a loss (illegal move)
            0 if move results in a draw
            1 if move results in a win
            2 otherwise

        Moves are represented by a one-hot encoding of 34 5x5 planes.

        Planes:
            0. move 1 square up
            1. move 1 square left
            2. move 1 square down
            3. move 1 square right
            4. move 1 square diagonally up and left
            5. move 2 squares diagonally up and left
            6. move 3 squares diagonally up and left
            7. move 4 squares diagonally up and left
            8. move 1 square diagonally up and right
            9. move 2 squares diagonally up and right
            10. move 3 squares diagonally up and right
            11. move 4 squares diagonally up and right
            12. move 1 square diagonally down and right
            13. move 2 squares diagonally down and right
            14. move 3 squares diagonally down and right
            15. move 4 squares diagonally down and right
            16. move 1 square diagonally down and left
            17. move 2 squares diagonally down and left
            18. move 3 squares diagonally down and left
            19. move 4 squares diagonally down and left
            20. move 1 square up and 2 squares left
            21. move 2 squares up and 1 square left
            22. move 2 square up and 1 square right
            23. move 1 square up and 2 squares right
            24. move 1 square down and 2 squares right
            25. move 2 squares down and 1 square right
            26. move 2 squares down and 1 square left
            27. move 1 square down and 2 squares left
            28. move 1 square up and promote to knight
            29. move 1 square diagonally up and left and promote to knight
            30. move 1 square diagonally up and right and promote to knight
            31. move 1 square up and promote to bishop
            32. move 1 square diagonally up and left and promote to bishop
            33. move 1 square diagonally up and right and promote to bishop

        e.g. white plays b3:

        move = [
            [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        ]
        """

        move_one_hot = np.where(move)
        move_index, square_row, square_col = move_one_hot[0][0], move_one_hot[1][0], move_one_hot[2][0]

        pawn = 0
        capture = 0

        for i in range(rules.NUM_PIECE_TYPES_PER_COLOR):  # find which piece is on the source square
            if self.pieces[i][square_row][square_col]:
                if i == rules.P1_PAWN_IDX:
                    pawn = 1

                ########################################
                #### Check move legality (pre-move) ####
                ########################################

                # check if move is legal for the piece
                if not rules.piece_move_legality[i][move_index]:
                    return -1

                # check if piece would move off the board
                if not rules.square_move_legality[move_index][square_row]:
                    return -1

                # get move bitboard
                move_vec = rules.move_dests[move_index][square_row][square_col]
                move_vec_one_hot = np.where(move_vec)
                square_dest_row, square_dest_col = move_vec_one_hot[0][0], move_vec_one_hot[1][0]
                
                # check if same color piece occupies destination square (illegal move)
                for j in range(rules.NUM_PIECE_TYPES_PER_COLOR):
                    if i == j:
                        continue
                    if self.pieces[j][square_dest_row][square_dest_col]:
                        return -1
                
                # check if opposite color piece occupies destination square (capture)
                for j in range(rules.NUM_PIECE_TYPES_PER_COLOR, rules.NUM_PIECE_TYPES_TOTAL):
                    if self.pieces[j][square_dest_row][square_dest_col]:
                        capture = 1
                        capture_idx = j
                        break;
                        
                # check if move is blocked by another piece
                move_passthru_vec = rules.move_passthrus[move_index][square_row][square_col]
                for j in range(rules.NUM_PIECE_TYPES_TOTAL):
                    if (self.pieces[j][square_row][square_col] & move_passthru_vec).any():
                        return -1

                # check pawn move legality:
                #   - pawn forward movement is non-capture on 1st, 2nd, or 3rd ranks
                #   - pawn diagonal movement is capture on 1st, 2nd, or 3rd ranks
                #   - pawn forward promotion is non-capture on 4th rank (already covered in previous legality checks)
                #   - pawn diagonal promotion is capture on 4th rank
                if pawn:
                    if capture:
                        if not rules.pawn_capture_move_legality[move_index][square_row][square_col]:
                            return -1
                    elif not rules.pawn_move_legality[move_index][square_row][square_col]:
                        return -1


                ######################
                #### Execute move ####
                ######################

                if move_index < 28:  # not pawn promotion
                    self.pieces[i][square_row][square_col] = 0
                    self.pieces[i][square_dest_row][square_dest_col] = 1
                elif move_index < 31:  # promotion to knight
                    self.pieces[i][square_row][square_col] = 0
                    self.pieces[P1_KNIGHT_IDX][square_dest_row][square_dest_col] = 1
                else:  # promotion to bishop
                    self.pieces[i][square_row][square_col] = 0
                    self.pieces[P1_BISHOP_IDX][square_dest_row][square_dest_col] = 1
                
                if capture:
                    self.pieces[capture_idx][square_dest_row][square_dest_col] = 0


                #########################################
                #### Check move legality (post-move) ####
                #########################################

                # check if king is in check after move


                ####################################
                #### Successfully executed move ####
                ####################################

                # update counters
                if self.color == 1:
                    self.total_move_count += 1
                    if pawn or capture:
                        self.no_progress_move_count = 0
                    else:
                        self.no_progress_move_count += 1


                ###############################
                #### Check if game is over ####
                ###############################

                # check if checkmate

                # check if stalemate

                # check repetitions
                for t in range(rules.HISTORY_LENGTH):
                    if np.equal(self.history[t], self.pieces).all():
                        if self.second_repetition:  # draw
                            return 0
                        elif self.first_repetition:
                            self.first_repetition = 0
                            self.second_repetition = 1
                        else:
                            self.first_repetition = 1
                        break

                # check if 50-move rule
                if self.no_progress_move_count >= rules.NO_PROGRESS_MOVES_LIM:
                    return 0


                ##############################
                #### Advance to next move ####
                ##############################

                # flip board for next color
                self.pieces = np.array([
                    np.flip(self.pieces[rules.P2_KING_IDX]),
                    np.flip(self.pieces[rules.P2_KNIGHT_IDX]),
                    np.flip(self.pieces[rules.P2_BISHOP_IDX]),
                    np.flip(self.pieces[rules.P2_PAWN_IDX]),
                    np.flip(self.pieces[rules.P1_KING_IDX]),
                    np.flip(self.pieces[rules.P1_KNIGHT_IDX]),
                    np.flip(self.pieces[rules.P1_BISHOP_IDX]),
                    np.flip(self.pieces[rules.P1_PAWN_IDX])
                ])
                
                # update history
                self.history = np.append(self.history[1:], [self.pieces], axis=0)
                self.first_repetition_history = self.first_repetition_history[1:] + [self.first_repetition]
                self.second_repetition_history = self.second_repetition_history[1:] + [self.second_repetition]
                    
                # update color
                self.color = 1 if (self.color == 0) else 0

                return 2

        # source square has no piece or piece of opposite color
        return -1



if __name__ == "__main__":
    board = Board()
    
    move = np.array([
        [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    ])

    np.set_printoptions(threshold=sys.maxsize)
    #print(board.move(move))
    #print(board.pieces)
    print(board.get_input_features())
