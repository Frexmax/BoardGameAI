from PositionFinders import piece_future_positions, king_future_positions, get_captures


class MoveFinder:
    def __init__(self):
        self.piece_positions = {}
        self.king_positions = {}
        self.players = (1, -1)
        self.piece = 1
        self.king = 2

    @staticmethod
    def find_positions(np_board):
        position_pieces = {1: set(), -1: set()}
        position_kings = {1: set(), -1: set()}
        for x in range(8):
            for y in range(8):
                piece = np_board[x, y]
                piece_type = abs(np_board[x, y])
                if piece_type == 1:  # PIECE
                    position_pieces[piece].add((x, y))
                elif piece_type == 2:  # KING
                    position_kings[piece // 2].add((x, y))
        return position_pieces, position_kings

    def initialized_positions(self, np_board):
        self.piece_positions = {}
        self.king_positions = {}
        for player in self.players:
            positions = set()
            for x in range(8):
                for y in range(8):
                    if np_board[x, y] == player:
                        positions.add((x, y))
            self.piece_positions[player] = positions
            self.king_positions[player] = set()

    def update_positions_piece(self, old_position, new_position, player):
        self.piece_positions[player].remove(old_position)
        self.piece_positions[player].add(new_position)

    def update_positions_king(self, old_position, new_position, player):
        self.king_positions[player].remove(old_position)
        self.king_positions[player].add(new_position)

    def remove_position_piece(self, old_position, player):
        self.piece_positions[player].remove(old_position)

    def remove_position_king(self, old_position, player):
        self.king_positions[player].remove(old_position)

    def create_king(self, position, player):
        self.piece_positions[player].remove(position)
        self.king_positions[player].add(position)

    def update_finder(self, np_board, action, player, enemy, moves_list, captures_list):
        old_position = moves_list[action][0]
        new_position = moves_list[action][-1]
        piece_type = abs(int(np_board[old_position]))
        if len(captures_list[action]) != 0 and type(captures_list[action][0]) == int:  # FIX LOOPS LATER ON
            captures = [captures_list[action]]
        else:
            captures = captures_list[action]
        if piece_type == 1:
            self.update_positions_piece(old_position, new_position, player)  # MOVED PIECE
        else:
            self.update_positions_king(old_position, new_position, player)  # MOVED KING
        if len(captures) != 0:
            for capture_position in captures:
                piece_type = abs(np_board[capture_position])
                if piece_type == 1:
                    self.remove_position_piece(capture_position, enemy)  # MOVED PIECE
                else:
                    self.remove_position_king(capture_position, enemy)  # MOVED KING

    def upgrade_piece(self, old_position, new_position, player):
        self.piece_positions[player].remove(old_position)
        self.king_positions[player].add(new_position)

    def moves_capture(self, np_board, piece_type, player, piece_positions, king_positions):
        moves = []  # NEW POSITIONS
        captures = []  # CAPTURE POSITIONS
        if abs(piece_type) == 1:  # PIECE
            for piece_position in piece_positions[player]:
                piece_moves, piece_captures = get_captures(np_board, piece_position, player, self.piece)
                moves.extend(piece_moves)
                captures.extend(piece_captures)
        else:  # KING
            for king_position in king_positions[player]:
                king_moves, king_captures = get_captures(np_board, king_position, player, self.king)
                moves.extend(king_moves)
                captures.extend(king_captures)
        return moves, captures

    @staticmethod
    def moves_no_capture(np_board, piece_type, player, piece_positions, king_positions):
        moves = []  # NEW POSITIONS
        if abs(piece_type) == 1:  # PIECE
            for piece_position in piece_positions[player]:
                x = piece_position[0]
                y = piece_position[1]
                piece_moves = piece_future_positions(np_board, x, y, player)
                for move in piece_moves:
                    moves.append((piece_position, move))
        else:  # KING
            for king_position in king_positions[player]:
                x = king_position[0]
                y = king_position[1]
                king_moves = king_future_positions(np_board, x, y)
                for move in king_moves:
                    moves.append((king_position, move))
        return moves

    def find_moves(self, np_board, player, piece_positions, king_positions, moves_list):
        moves_piece = self.moves_no_capture(np_board, self.piece, player, piece_positions, king_positions)
        moves_king = self.moves_no_capture(np_board, self.king, player, piece_positions, king_positions)
        captures_piece, captured_enemy_piece = self.moves_capture(np_board, self.piece, player,
                                                                  piece_positions, king_positions)
        captures_king, captured_enemy_king = self.moves_capture(np_board, self.king, player,
                                                                piece_positions, king_positions)
        if len(captures_piece) > 0 or len(captures_king) > 0:
            moves = []
            captures = []
            captures.extend(captured_enemy_piece)
            captures.extend(captured_enemy_king)
            moves.extend(captures_piece)
            moves.extend(captures_king)
            return self.filter_moves(moves, moves_list)

        elif len(moves_piece) > 0 or len(moves_king) > 0:
            moves = []
            moves.extend(moves_piece)
            moves.extend(moves_king)
            return self.filter_moves(moves, moves_list)
        return None

    @staticmethod
    def filter_moves(moves, moves_list):
        move_index = []
        for move in moves:
            index = moves_list.index(move)
            move_index.append(index)
        move_index.sort()
        return move_index
