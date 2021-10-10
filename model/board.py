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
        self.history = np.append(np.zeros((rules.HISTORY_LENGTH - 1, rules.NUM_PIECE_TYPES_TOTAL, rules.BOARD_SIDE_LENGTH, rules.BOARD_SIDE_LENGTH), int), [self.pieces], axis=0)

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
        features.append(np.full((5, 5), self.total_move_count))
        features.append(np.full((5, 5), self.no_progress_move_count))

        return np.array(features)


    def pprint(self):
        """
        Pretty prints the board position. Always prints from white's perspective.
        """
        position = [["  "]*rules.BOARD_SIDE_LENGTH for i in range(rules.BOARD_SIDE_LENGTH)]

        p1_color = "W" if (self.color == 0) else "B"
        p2_color = "B" if (self.color == 0) else "W"

        for row in range(rules.BOARD_SIDE_LENGTH):
            for col in range(rules.BOARD_SIDE_LENGTH):
                if self.pieces[rules.P1_KING_IDX][row][col]:
                    position[row][col] = p1_color + "K"
                elif self.pieces[rules.P1_KNIGHT_IDX][row][col]:
                    position[row][col] = p1_color + "N"
                elif self.pieces[rules.P1_BISHOP_IDX][row][col]:
                    position[row][col] = p1_color + "B"
                elif self.pieces[rules.P1_PAWN_IDX][row][col]:
                    position[row][col] = p1_color + "p"
                elif self.pieces[rules.P2_KING_IDX][row][col]:
                    position[row][col] = p2_color + "K"
                elif self.pieces[rules.P2_KNIGHT_IDX][row][col]:
                    position[row][col] = p2_color + "N"
                elif self.pieces[rules.P2_BISHOP_IDX][row][col]:
                    position[row][col] = p2_color + "B"
                elif self.pieces[rules.P2_PAWN_IDX][row][col]:
                    position[row][col] = p2_color + "p"

        if self.color == 0:
            print(" ________________________ ")
            print("|    |    |    |    |    |")
            print("| {} | {} | {} | {} | {} |".format(*position[4]))
            print("|____|____|____|____|____|")
            print("|    |    |    |    |    |")
            print("| {} | {} | {} | {} | {} |".format(*position[3]))
            print("|____|____|____|____|____|")
            print("|    |    |    |    |    |")
            print("| {} | {} | {} | {} | {} |".format(*position[2]))
            print("|____|____|____|____|____|")
            print("|    |    |    |    |    |")
            print("| {} | {} | {} | {} | {} |".format(*position[1]))
            print("|____|____|____|____|____|")
            print("|    |    |    |    |    |")
            print("| {} | {} | {} | {} | {} |".format(*position[0]))
            print("|____|____|____|____|____|")
        else:
            print(" ________________________ ")
            print("|    |    |    |    |    |")
            print("| {} | {} | {} | {} | {} |".format(*position[0][::-1]))
            print("|____|____|____|____|____|")
            print("|    |    |    |    |    |")
            print("| {} | {} | {} | {} | {} |".format(*position[1][::-1]))
            print("|____|____|____|____|____|")
            print("|    |    |    |    |    |")
            print("| {} | {} | {} | {} | {} |".format(*position[2][::-1]))
            print("|____|____|____|____|____|")
            print("|    |    |    |    |    |")
            print("| {} | {} | {} | {} | {} |".format(*position[3][::-1]))
            print("|____|____|____|____|____|")
            print("|    |    |    |    |    |")
            print("| {} | {} | {} | {} | {} |".format(*position[4][::-1]))
            print("|____|____|____|____|____|")


    def _get_attacking_squares(self, piece_idx, opposite=0):
        attacking_squares = np.zeros((rules.BOARD_SIDE_LENGTH, rules.BOARD_SIDE_LENGTH), int)

        # get all squares of piece
        square = np.nonzero(self.pieces[piece_idx])
        square_rows, square_cols = square[0], square[1]

        for i in range(len(square_rows)):
            if not opposite:  # squares being attacked by color
                attacking_squares = attacking_squares | rules.attack_squares_p1[piece_idx][square_rows[i]][square_cols[i]]
            else:  # squares being attacked by opposite color
                attacking_squares = attacking_squares | rules.attack_squares_p2[piece_idx - rules.NUM_PIECE_TYPES_PER_COLOR][square_rows[i]][square_cols[i]]

            # account for pieces blocking bishops
            # TODO: find a way to avoid doing this since this is slow
            if piece_idx == rules.P1_BISHOP_IDX or piece_idx == rules.P2_BISHOP_IDX:
                diagonal_move_indices = [5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
                for move_index in diagonal_move_indices:
                    dest = np.nonzero(rules.move_dests[move_index][square_rows[i]][square_cols[i]])
                    if not dest[0].size:
                        continue

                    dest_row, dest_col = dest[0][0], dest[1][0]
                    if attacking_squares[dest_row][dest_col]:  # move results in piece landing on the square being attacked
                        # check if piece is blocking
                        move_passthru_vec = rules.move_passthrus[move_index][square_rows[i]][square_cols[i]]
                        for j in range(rules.NUM_PIECE_TYPES_TOTAL):
                            # king doesn't actually "block" a bishop's attack
                            if opposite and j == rules.P1_KING_IDX:
                                continue
                            elif j == rules.P2_KING_IDX:
                                continue

                            if (self.pieces[j] & move_passthru_vec).any():  # piece is blocking
                                attacking_squares[dest_row][dest_col] = 0
                                break

        return attacking_squares


    def _get_pseudolegal_squares(self, piece_idx, opposite=0):
        pseudolegal_squares = self._get_attacking_squares(piece_idx, opposite=opposite)
        if opposite:
            for i in range(rules.NUM_PIECE_TYPES_PER_COLOR, rules.NUM_PIECE_TYPES_TOTAL):
                pseudolegal_squares = (pseudolegal_squares ^ self.pieces[i]) & pseudolegal_squares
        else:
            for i in range(rules.NUM_PIECE_TYPES_PER_COLOR):
                pseudolegal_squares = (pseudolegal_squares ^ self.pieces[i]) & pseudolegal_squares

        return pseudolegal_squares


    def move(self, move):
        """
        Returns:
            -1 if move results in a loss (illegal move)
            0 if move results in a draw
            1 if move results in a win
            2 otherwise

        Moves are represented by an index into 34 5x5 planes.

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

        #move_one_hot = np.nonzero(move)

        #move_index, square_row, square_col = move_one_hot[0][0], move_one_hot[1][0], move_one_hot[2][0]
        move_index = move//rules.NUM_SQUARES
        square = move % rules.NUM_SQUARES;
        square_row, square_col = square//rules.BOARD_SIDE_LENGTH, square % rules.BOARD_SIDE_LENGTH

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
                move_vec_one_hot = np.nonzero(move_vec)
                square_dest_row, square_dest_col = move_vec_one_hot[0][0], move_vec_one_hot[1][0]

                # check if opposite color piece occupies destination square (capture)
                for j in range(rules.NUM_PIECE_TYPES_PER_COLOR, rules.NUM_PIECE_TYPES_TOTAL):
                    if self.pieces[j][square_dest_row][square_dest_col]:
                        capture = 1
                        capture_idx = j
                        break

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

                # check if same color piece occupies destination square (illegal move)
                # check if bishop move is blocked by another piece
                else:
                    if not self._get_pseudolegal_squares(i)[square_dest_row][square_dest_col]:
                        return -1

                # for j in range(rules.NUM_PIECE_TYPES_PER_COLOR):
                #     if i == j:
                #         continue
                #     if self.pieces[j][square_dest_row][square_dest_col]:
                #         return -1


                # if i == rules.P1_BISHOP_IDX:
                #     move_passthru_vec = rules.move_passthrus[move_index][square_row][square_col]
                #     for j in range(rules.NUM_PIECE_TYPES_TOTAL):
                #         if (self.pieces[j] & move_passthru_vec).any():
                #             return -1

                ######################
                #### Execute move ####
                ######################

                if move_index < 28:  # not pawn promotion
                    self.pieces[i][square_row][square_col] = 0
                    self.pieces[i][square_dest_row][square_dest_col] = 1
                elif move_index < 31:  # promotion to knight
                    self.pieces[i][square_row][square_col] = 0
                    self.pieces[rules.P1_KNIGHT_IDX][square_dest_row][square_dest_col] = 1
                else:  # promotion to bishop
                    self.pieces[i][square_row][square_col] = 0
                    self.pieces[rules.P1_BISHOP_IDX][square_dest_row][square_dest_col] = 1

                if capture:
                    self.pieces[capture_idx][square_dest_row][square_dest_col] = 0


                #########################################
                #### Check move legality (post-move) ####
                #########################################

                # check if king is in check after move
                if i == rules.P1_KING_IDX:
                    for j in range(rules.NUM_PIECE_TYPES_PER_COLOR, rules.NUM_PIECE_TYPES_TOTAL):
                        if self._get_attacking_squares(j, opposite=1)[square_dest_row][square_dest_col]:
                            return -1


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
                attacking_squares = np.zeros((rules.BOARD_SIDE_LENGTH, rules.BOARD_SIDE_LENGTH), int)
                for j in range(rules.NUM_PIECE_TYPES_PER_COLOR):
                    attacking_squares = attacking_squares | self._get_attacking_squares(j)

                opposite_king_squares = self._get_pseudolegal_squares(rules.P2_KING_IDX, opposite=1)
                free_squares = (opposite_king_squares ^ attacking_squares) & opposite_king_squares

                if (attacking_squares & self.pieces[rules.P2_KING_IDX]).any():  # opposite king in check
                    if not free_squares.any():
                        return 1

                # check if stalemate
                found_move = 0
                for j in range(rules.NUM_PIECE_TYPES_PER_COLOR, rules.NUM_PIECE_TYPES_TOTAL):
                    if j == rules.P2_KING_IDX:
                        if not (attacking_squares & self.pieces[rules.P2_KING_IDX]).any():  # opposite king not in check
                            if free_squares.any():
                                found_move = 1
                                break
                    else:
                        if self._get_pseudolegal_squares(j, opposite=1).any():
                            found_move = 1
                            break

                if not found_move:
                    return 0

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
    np.set_printoptions(threshold=sys.maxsize)

    board = Board()
    board.pprint()

    moves = [6, 207, 9]

    for move in moves:
        board.move(move)
        board.pprint()
