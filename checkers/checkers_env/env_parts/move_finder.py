from checkers.checkers_env.env_parts.position_finders import (piece_future_positions,
                                                              king_future_positions, get_captures)


class MoveFinder:
    def __init__(self):
        """
        Constructor for the MoveFinder class.
        Initializes the containers tracking the positions of pieces and kings for both players,
        as well as the player and piece type constants used throughout the class
        """

        self.piece_positions = {}
        self.king_positions = {}
        self._players = (1, -1)
        self._piece = 1
        self._king = 2

    @staticmethod
    def find_positions(np_board):
        """
        Find the positions of every regular piece and king for both players on the board

        :param np_board: state of the game
        :return: positions of the regular pieces for both players, positions of the kings for both players
        """

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
        """
        Initialize the tracked piece positions for both players from the starting board,
        as at the start of a game no kings are present yet

        :param np_board: state of the game
        """

        self.piece_positions = {}
        self.king_positions = {}
        for player in self._players:
            positions = set()
            for x in range(8):
                for y in range(8):
                    if np_board[x, y] == player:
                        positions.add((x, y))
            self.piece_positions[player] = positions
            self.king_positions[player] = set()

    def _update_positions_piece(self, old_position, new_position, player):
        """
        Update the tracked position of a moved regular piece

        :param old_position: position the piece moved from
        :param new_position: position the piece moved to
        :param player: player the piece belongs to
        """

        self.piece_positions[player].remove(old_position)
        self.piece_positions[player].add(new_position)

    def _update_positions_king(self, old_position, new_position, player):
        """
        Update the tracked position of a moved king

        :param old_position: position the king moved from
        :param new_position: position the king moved to
        :param player: player the king belongs to
        """

        self.king_positions[player].remove(old_position)
        self.king_positions[player].add(new_position)

    def _remove_position_piece(self, old_position, player):
        """
        Remove the tracked position of a captured regular piece

        :param old_position: position of the captured piece
        :param player: player the captured piece belongs to
        """

        self.piece_positions[player].remove(old_position)

    def _remove_position_king(self, old_position, player):
        """
        Remove the tracked position of a captured king

        :param old_position: position of the captured king
        :param player: player the captured king belongs to
        """

        self.king_positions[player].remove(old_position)

    def create_king(self, position, player):
        """
        Move a tracked position from the regular piece positions to the king positions,
        used when a piece reaches the opposite end of the board and is upgraded to a king

        :param position: position of the piece being upgraded
        :param player: player the piece belongs to
        """

        self.piece_positions[player].remove(position)
        self.king_positions[player].add(position)

    def update_finder(self, np_board, action, player, enemy, moves_list, captures_list):
        """
        Update the tracked piece and king positions after the provided action is performed,
        moving the piece/king that performed the action and removing any captured enemy pieces/kings

        :param np_board: state of the game before the move
        :param action: action being performed
        :param player: player performing the action
        :param enemy: enemy of the player performing the action
        :param moves_list: list of all possible moves
        :param captures_list: list of captures associated with each possible move
        """

        old_position = moves_list[action][0]
        new_position = moves_list[action][-1]
        piece_type = abs(int(np_board[old_position]))
        if len(captures_list[action]) != 0 and type(captures_list[action][0]) == int:  # FIX LOOPS LATER ON
            captures = [captures_list[action]]
        else:
            captures = captures_list[action]
        if piece_type == 1:
            self._update_positions_piece(old_position, new_position, player)  # MOVED PIECE
        else:
            self._update_positions_king(old_position, new_position, player)  # MOVED KING
        if len(captures) != 0:
            for capture_position in captures:
                piece_type = abs(np_board[capture_position])
                if piece_type == 1:
                    self._remove_position_piece(capture_position, enemy)  # MOVED PIECE
                else:
                    self._remove_position_king(capture_position, enemy)  # MOVED KING

    def _upgrade_piece(self, old_position, new_position, player):
        """
        Move the tracked position of a piece being upgraded to a king,
        from the regular piece positions to the king positions.
        Currently unused, kept for potential future use

        :param old_position: position the piece moved from
        :param new_position: position the piece moved to (and is upgraded at)
        :param player: player the piece belongs to
        """

        self.piece_positions[player].remove(old_position)
        self.king_positions[player].add(new_position)

    def _moves_capture(self, np_board, piece_type, player, piece_positions, king_positions):
        """
        Find all the possible capturing moves for the given piece type and player

        :param np_board: state of the game
        :param piece_type: type of piece (regular piece or king) to find capturing moves for
        :param player: player the pieces belong to
        :param piece_positions: positions of the regular pieces for both players
        :param king_positions: positions of the kings for both players
        :return: list of new positions after a capture, list of positions of the captured enemy pieces
        """

        moves = []  # NEW POSITIONS
        captures = []  # CAPTURE POSITIONS
        if abs(piece_type) == 1:  # PIECE
            for piece_position in piece_positions[player]:
                piece_moves, piece_captures = get_captures(np_board, piece_position, player, self._piece)
                moves.extend(piece_moves)
                captures.extend(piece_captures)
        else:  # KING
            for king_position in king_positions[player]:
                king_moves, king_captures = get_captures(np_board, king_position, player, self._king)
                moves.extend(king_moves)
                captures.extend(king_captures)
        return moves, captures

    @staticmethod
    def _moves_no_capture(np_board, piece_type, player, piece_positions, king_positions):
        """
        Find all the possible non-capturing moves for the given piece type and player

        :param np_board: state of the game
        :param piece_type: type of piece (regular piece or king) to find non-capturing moves for
        :param player: player the pieces belong to
        :param piece_positions: positions of the regular pieces for both players
        :param king_positions: positions of the kings for both players
        :return: list of (old position, new position) tuples for the possible non-capturing moves
        """

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
        """
        Find the possible moves for the player from the given game state.
        If any capturing moves are available they must be played, so only capturing moves are returned,
        otherwise the possible non-capturing moves are returned

        :param np_board: state of the game
        :param player: player who is now to play
        :param piece_positions: positions of the regular pieces for both players
        :param king_positions: positions of the kings for both players
        :param moves_list: list of all possible moves
        :return: list of possible move indexes for the provided game state, or None if no moves are available
        """

        moves_piece = self._moves_no_capture(np_board, self._piece, player, piece_positions, king_positions)
        moves_king = self._moves_no_capture(np_board, self._king, player, piece_positions, king_positions)
        captures_piece, captured_enemy_piece = self._moves_capture(np_board, self._piece, player,
                                                                  piece_positions, king_positions)
        captures_king, captured_enemy_king = self._moves_capture(np_board, self._king, player,
                                                                piece_positions, king_positions)
        if len(captures_piece) > 0 or len(captures_king) > 0:
            moves = []
            captures = []
            captures.extend(captured_enemy_piece)
            captures.extend(captured_enemy_king)
            moves.extend(captures_piece)
            moves.extend(captures_king)
            return self._filter_moves(moves, moves_list)

        elif len(moves_piece) > 0 or len(moves_king) > 0:
            moves = []
            moves.extend(moves_piece)
            moves.extend(moves_king)
            return self._filter_moves(moves, moves_list)
        return None

    @staticmethod
    def _filter_moves(moves, moves_list):
        """
        Map every move from the provided list of moves
        to the corresponding index in the game's list of all moves

        :param moves: list of moves in the format (old position, new position)
        :param moves_list: list of all possible moves
        :return: list of move indexes
        """

        move_index = []
        for move in moves:
            index = moves_list.index(move)
            move_index.append(index)
        move_index.sort()
        return move_index
