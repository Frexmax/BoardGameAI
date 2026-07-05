from numba import jit


@jit(nopython=True)
def valid(np_board, x, y):
    """
    Check if a piece can move the position (x, y),
    by verifying that it is withing the bounds of the board (0 - 8 inclusive for both x and y),
    and that there are no other pieces already present in that position

    :param np_board: state of the game board as a numpy array
    :param x: x coordinate on the numpy board
    :param y: y coordinate on the numpy board
    :return: True, if the piece can move to that position, else False
    """

    if 8 > x >= 0 and 0 <= y < 8:
        if np_board[x, y] == 0:
            return True
    return False


@jit(nopython=True)
def valid_capture(np_board, capture_pos, future_pos, player):
    """
    Check if a capture movement can be made by the specified player,
    by verifying that the end position is withing the bounds of the board (0 - 8 inclusive for both x and y),
    and that the captured position is occupied by an enemy piece (-1 * player -> standard, -2 * player -> king)

    :param np_board: state of the game board as a numpy array
    :param capture_pos: numpy array with x, y coordinates of the position captured in the move
    :param future_pos: numpy array with x, y coordinates of the end future in the move
    :param player: player making the capture move 
    :return: True, if the capture move can be made, else False
    """

    if 0 <= future_pos[0] < 8 and 0 <= future_pos[1] < 8:
        if np_board[future_pos] == 0 and (np_board[capture_pos] == -player or
                                          np_board[capture_pos] == -2 * player):
            return True
    return False


def piece_future_positions(np_board, x, y, player):
    """
    Find the possible non-capturing move positions for a regular piece at the given position.
    Regular pieces can only move diagonally forward, in the direction depending on the player

    :param np_board: state of the game board as a numpy array
    :param x: x coordinate of the piece
    :param y: y coordinate of the piece
    :param player: player the piece belongs to
    :return: list of valid future positions for the piece
    """

    future_positions = []
    directions_red = [(-1, -1), (-1, 1)]
    directions_black = [(1, 1), (1, -1)]
    if player == 1:
        used_directions = directions_red
    else:
        used_directions = directions_black
    for direction in used_directions:
        move_x = x + direction[0]
        move_y = y + direction[1]
        if valid(np_board, move_x, move_y):
            future_positions.append((move_x, move_y))
    return future_positions


def king_future_positions(np_board, x, y):
    """
    Find the possible non-capturing move positions for a king at the given position.
    Unlike regular pieces, kings can move diagonally in any of the 4 directions

    :param np_board: state of the game board as a numpy array
    :param x: x coordinate of the king
    :param y: y coordinate of the king
    :return: list of valid future positions for the king
    """

    future_positions = []
    directions = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
    for direction in directions:
        move_x = x + direction[0]
        move_y = y + direction[1]
        if valid(np_board, move_x, move_y):
            future_positions.append((move_x, move_y))
    return future_positions


def get_captures(np_board, start_pos, player, piece_type):
    """
    Find all the possible capturing moves (including chained multi-captures) starting from the given position.
    Every possible capture sequence is explored breadth-first, and any sequence that is a strict prefix
    of a longer available sequence is discarded, since captures must be continued as far as possible

    :param np_board: state of the game board as a numpy array
    :param start_pos: starting position of the piece or king performing the capture
    :param player: player the piece or king belongs to
    :param piece_type: type of the piece performing the capture (regular piece or king)
    :return: list of capture move sequences (each a tuple of positions visited), list of the corresponding
             sequences of captured enemy positions
    """

    directions_red = [(-1, -1), (-1, 1)]
    directions_black = [(1, 1), (1, -1)]
    directions_king = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
    if abs(piece_type) == 1 and player == 1:
        used_directions = directions_red  # RED PIECE CAPTURES
    elif abs(piece_type) == 1 and player == -1:
        used_directions = directions_black  # BLACK PIECE CAPTURES
    else:
        used_directions = directions_king  # KING CAPTURES

    out = [[start_pos]]
    index = 0
    while index < len(out):
        move = out[index]
        for direction in used_directions:
            new_capture = (move[-1][0] + direction[0], move[-1][1] + direction[1])
            new_move = (new_capture[0] + direction[0], new_capture[1] + direction[1])
            if (len(move) == 1 or new_move not in move) and valid_capture(np_board, new_capture, new_move, player):
                out.append(move+[new_move])
        index += 1

    num_deleted = 0
    out_copy = out.copy()
    for i in range(len(out_copy)):
        last_position = out_copy[i][-1]
        for j in range(len(out_copy)):
            if (last_position in out_copy[j]) and last_position != out_copy[j][-1] and i != j:
                del out[i - num_deleted]
                num_deleted += 1
                break

    moves = []
    captures = []
    for i in range(len(out)):
        if len(out[i]) == 1:
            continue
        moves.append([])
        captures.append([])
        for j in range(len(out[i])):
            moves[i].append(out[i][j])
            if j < len(out[i]) - 1:
                direction_x = (out[i][j + 1][0] - out[i][j][0]) / 2
                direction_y = (out[i][j + 1][1] - out[i][j][1]) / 2
                captures[i].append((int(out[i][j][0] + direction_x), int(out[i][j][1] + direction_y)))
    for i in range(len(moves)):
        moves[i] = tuple(moves[i])
        captures[i] = tuple(captures[i])
    return moves, captures
