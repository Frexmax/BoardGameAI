from numba import jit


@jit(nopython=True)
def valid(np_board, x, y):
    if 8 > x >= 0 and 0 <= y < 8:
        if np_board[x, y] == 0:
            return True
    return False


@jit(nopython=True)
def valid_capture(np_board, capture_pos, future_pos, player):
    if 0 <= future_pos[0] < 8 and 0 <= future_pos[1] < 8:
        if np_board[future_pos] == 0 and (np_board[capture_pos] == -player or
                                          np_board[capture_pos] == -2 * player):
            return True
    return False


def piece_future_positions(np_board, x, y, player):
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
    future_positions = []
    directions = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
    for direction in directions:
        move_x = x + direction[0]
        move_y = y + direction[1]
        if valid(np_board, move_x, move_y):
            future_positions.append((move_x, move_y))
    return future_positions


def get_captures(np_board, start_pos, player, piece_type):
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
