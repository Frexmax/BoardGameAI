import pickle

import numpy as np
from numba import jit

from checkers.checkers_env.env_parts.checkers_board import Board
from checkers.checkers_env.env_parts.move_finder import MoveFinder


class CheckersEnv:
    """
    Class implementing the checkers game, which also functions as the reinforcement learning environment.
    High level game functionality is implemented here,
    the class also acts as an interface for the external player, AI or human,
    with wrapper methods to interact with the Board class.
    """

    def __init__(self, board_parameters, draw_parameters, to_render=False):
        """
        Constructor for the checkers game class.
        Basic data about the possible actions and states is initialized, which is needed for the agent training.
        The game Board and MoveFinder classes are also initialized in this constructor, with the provided board
        and draw parameters being passed to the Board.
        Heuristic values and piece/king positional value maps used for evaluating game states are also set up here.

        :param board_parameters: parameters for the Board
        :param draw_parameters: parameters for the Drawer
        :param to_render: render the board flag
        """

        with open('./checkers/checkers_env/moves_list/captures_list.pkl', 'rb') as f:
            self._captures_list = pickle.load(f)

        with open('./checkers/checkers_env/moves_list/moves_list.pkl', 'rb') as f:
            self.moves_list = pickle.load(f)

        self.optimal_move_count = -1
        self.board = Board(board_parameters, draw_parameters, to_render)
        self._move_finder = MoveFinder()

        # ACTION AND OBSERVATION SPACE
        self.action_space = 2852  # NUM MOVES
        self._observation_space = (8, 8)  # STATE SHAPE
        self.refactored_space = (6, 8, 8)

        self.player = 1
        self._enemy = -1
        self.max_moves = 200
        self.move_counter = 0
        self._previous_action = None

        # HEURISTICS
        self._piece_value = 5
        self._king_value = 10

        # PIECE VALUE MAPS
        piece_value_map_red = np.array([[0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9],
                                        [0.7, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.7],
                                        [0.6, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6],
                                        [0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5],
                                        [0.4, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.4],
                                        [0.3, 0.4, 0.5, 0.5, 0.4, 0.4, 0.4, 0.3],
                                        [0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2],
                                        [0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1]])

        piece_value_map_black = np.array([[0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1],
                                          [0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2],
                                          [0.3, 0.4, 0.5, 0.5, 0.4, 0.4, 0.4, 0.3],
                                          [0.4, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.4],
                                          [0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5],
                                          [0.6, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6],
                                          [0.7, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.7],
                                          [0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9]])

        # KING VALUE MAPS
        king_value_map = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                   [0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1],
                                   [0.2, 0.4, 0.3, 0.3, 0.3, 0.3, 0.4, 0.2],
                                   [0.3, 0.6, 0.7, 0.7, 0.7, 0.7, 0.6, 0.3],
                                   [0.4, 0.8, 1.0, 1.0, 1.0, 1.0, 0.8, 0.4],
                                   [0.3, 0.6, 0.7, 0.7, 0.7, 0.7, 0.6, 0.3],
                                   [0.2, 0.4, 0.3, 0.3, 0.3, 0.3, 0.4, 0.2],
                                   [0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1]])

        self._piece_value_map = {1: piece_value_map_red, -1: piece_value_map_black}
        self._king_value_map = king_value_map

    def reset(self):
        """
        Reset the game state when starting a new round.
        Reset the board and move finder, and make player 1 play the next turn

        :return: game state of the reset board, possible moves to play
        """

        self.player = 1
        self._enemy = -1
        self.board.reset()
        self._move_finder.initialized_positions(self.board.np_board)
        self.move_counter = 0
        moves_player = self._move_finder.find_moves(self.board.np_board, self.player, self._move_finder.piece_positions,
                                                   self._move_finder.king_positions, self.moves_list)
        return np.copy(self.board.np_board), moves_player

    def add_heuristics(self, state, player, possible_moves):
        """
        Approximate the value of a state, used for Monte-Carlo-Tree Search.
        Calculates a weighted value of the pieces and kings for both the player and the enemy,
        using positional value maps and a small bonus for supported pieces, while ignoring
        enemy pieces that would be captured along the evaluated move

        :param state: state of the game
        :param player: player who is about to play
        :param possible_moves: possible moves for the given state, used to find enemy pieces about to be captured
        :return: heuristic value of the state, normalized between -1 and 1
        """

        def check_supported(state, x, y, player, directions):
            """Bonus value for a piece at (x, y) if supported by an adjacent friendly piece"""

            support_value = 0
            for direction in directions[-(player - 1) // 2]:
                x_n = x + direction[0]
                y_n = y + direction[1]
                if (8 > x_n >= 0 and 0 <= y_n < 8) and (state[x_n, y_n] == player or state[x_n, y_n] == 2 * player):
                    support_value += 0.025
            return support_value

        def calculate_value(king_value, piece_value, player_map, enemy_map, king_map,
                            state, player, captured_positions, directions):
            """
            Weighted heuristic value of the player's and the enemy's pieces/kings, using the
            positional value maps and support bonus, ignoring enemy pieces already captured
            along the evaluated move

            :return: heuristic value of the player's pieces, heuristic value of the enemy's pieces
            """

            player_value = 0
            enemy_value = 0
            for x in range(8):
                for y in range(8):
                    if state[x, y] == player:
                        player_value += (piece_value + check_supported(state, x, y, player, directions)) * \
                                        player_map[x, y]
                    elif state[x, y] == 2 * player:
                        player_value += king_value * king_map[x, y]
                    if state[x, y] == -player and (x, y) not in captured_positions:
                        enemy_value += (piece_value + check_supported(state, x, y, -player, directions)) * \
                                       enemy_map[x, y]
                    elif state[x, y] == -2 * player and (x, y) not in captured_positions:
                        enemy_value += king_value * king_map[x, y]
            return player_value, enemy_value

        captured_positions = [(-1, -1)]
        directions = np.array([[[1, 1], [1, -1]], [[-1, -1], [-1, 1]]])
        for move_index in possible_moves:
            move = self.moves_list[move_index]
            if len(move) > 2:
                captures = self._captures_list[move_index]
                for capture in captures:
                    captured_positions.append(capture)
        captured_positions = tuple(captured_positions)
        p_value, e_value = calculate_value(self._king_value, self._piece_value, self._piece_value_map[player],
                                           self._piece_value_map[-player], self._king_value_map,
                                           state, player, captured_positions, directions)
        return -(p_value - e_value) / (p_value + e_value)

    def render(self):
        """
        Wrapper method to render the game board
        """

        self.board.render(self.player)

    def highlight_piece(self, click_position, highlighted_before, possible_moves):
        """
        Wrapper method to highlight a clicked piece and its possible moves on the board

        :param click_position: pixel coordinates of the click
        :param highlighted_before: piece and moves highlighted before this click (if any)
        :param possible_moves: possible moves available to the current player
        :return: the piece and moves now highlighted (from the Board method)
        """

        return self.board.highlight_piece(click_position, self.player, highlighted_before,
                                          possible_moves, self.moves_list)

    def end_highlight(self, highlighted_before):
        """
        Wrapper method to remove the highlight from the previously selected piece and its moves

        :param highlighted_before: piece and moves currently highlighted
        """

        self.board.end_highlight(highlighted_before, self.player)

    @staticmethod
    def hash_state(state):
        """
        Hash state to be used in the Monte-Carlo-Tree Search.
        Experimental feature ...

        :param state: state of the game
        :return: hash code of the game state
        """

        array = []
        for x in range(8):
            row = tuple(state[x])
            array.append(row)
        array = tuple(array)
        hash_value = hash(array)
        return hash_value

    @staticmethod
    def _flip_board_perspective(state):
        """
        Create a new state with a flipped player perspective,
        by rotating the board 180 degrees and inverting the piece values

        :param state: state of the game
        :return: state of the game with flipped perspective
        """

        return np.rot90(np.copy(state)) * -1

    @staticmethod
    @jit(nopython=True)
    def refactor_state(state, player, move_counter):
        """
        Refactor the provided game state for model training.
        The state is split into separate binary layers for player pieces, player kings,
        enemy pieces and enemy kings, with two additional layers filled with the current
        player and move counter values

        :param state: state of the game
        :param player: player who is about to play
        :param move_counter: how many moves have been made in the round
        :return: refactored state of the game
        """

        refactored_state = np.zeros((6, 8, 8), dtype=np.int32)
        for x in range(8):
            for y in range(8):
                piece = abs(state[x][y])
                if piece == 1:
                    refactored_state[0][x][y] = 1
                elif piece == 2:
                    refactored_state[1][x][y] = 1
                elif piece == -1:
                    refactored_state[2][x][y] = 1
                elif piece == -2:
                    refactored_state[3][x][y] = 1
        refactored_state[4] = player
        refactored_state[5] = move_counter
        return np.expand_dims(refactored_state, axis=0)

    @staticmethod
    def _string_representation(state):
        """
        Convert the game state array to a string

        :param state: state of the game
        :return: string representing the provided game state
        """

        return np.copy(state).tostring()

    def find_positions(self, state):
        """
        Wrapper method to find the positions of the regular pieces and kings on the board

        :param state: state of the game
        :return: positions of the regular pieces, positions of the kings (from the MoveFinder method)
        """

        return self._move_finder.find_positions(state)

    def find_moves(self, state, player, piece_positions, king_positions):
        """
        Wrapper method to find the possible moves from a game state

        :param state: state of the game
        :param player: player who is now to play
        :param piece_positions: positions of the regular pieces on the board
        :param king_positions: positions of the king pieces on the board
        :return: list of possible moves for the provided game state
        """

        return self._move_finder.find_moves(state, player, piece_positions, king_positions, self.moves_list)

    def make_move(self, state, action, player):
        """
        Wrapper method to make a move on the board

        :param state: state of the game
        :param action: action to be performed
        :param player: player making the move
        :return: state of the board (NumPy array) after the performed move
        """

        next_state, _ = self.board.make_move(np.copy(state), action, player, self.moves_list,
                                             self._captures_list, to_render=None)
        return np.copy(next_state)

    def state_reward(self, state, player, move_counter):
        """
        Calculate reward for a given state

        :param state: state of the game
        :param player: player who made the last move
        :param move_counter: how many moves have been made in the round
        :return: reward for the state, boolean indicating game over, moves possible for the state
        """

        piece_positions, king_positions = self.find_positions(state)
        game_end, win_state, moves_enemy = self._check_win(state, player, -player, move_counter,
                                                          piece_positions, king_positions)
        reward = self._give_reward(win_state)
        return reward, game_end, moves_enemy

    @staticmethod
    def _determine_piece_winner(state, player, enemy):
        """
        Determine the winner based on the total number of pieces and kings remaining
        for the player and the enemy. Currently unused, kept for potential future use
        in determining a winner once the move limit is reached

        :param state: state of the game
        :param player: player being evaluated
        :param enemy: enemy of the player
        :return: winner of the game based on piece count (or 0 for a draw)
        """

        def count_pieces(state, player, enemy):
            """Count the number of regular pieces belonging to the player and the enemy"""

            return len(np.where(state == player)[0]), len(np.where(state == enemy)[0])

        def count_kings(state, player, enemy):
            """Count the number of king pieces belonging to the player and the enemy"""

            return len(np.where(state == 2 * player)[0]), len(np.where(state == 2 * enemy)[0])

        win_state = None
        pieces_player, pieces_enemy = count_pieces(state, player, enemy)
        kings_player, kings_enemy = count_kings(state, player, enemy)
        total_player = pieces_player + kings_player
        total_enemy = pieces_enemy + kings_enemy
        if total_player > total_enemy:
            win_state = player
        elif total_player < total_enemy:
            win_state = enemy
        elif total_player == total_enemy:
            if kings_player > kings_enemy:
                win_state = player
            elif kings_player < kings_enemy:
                win_state = enemy
            else:
                win_state = 0
        return win_state

    def _check_win(self, state, player, enemy, move_counter, piece_positions, king_positions):
        """
        Check for game win for the last player based on the provided game state.
        3 conditions checked:
        1. check if the enemy has no pieces or kings remaining -> player wins
        2. check if the enemy has no possible moves left -> player wins
        3. check if the move limit has been reached -> draw

        :param state: state of the game
        :param player: player who made the last move
        :param enemy: enemy of the player
        :param move_counter: how many moves have been made in the round
        :param piece_positions: positions of the regular pieces on the board
        :param king_positions: positions of the king pieces on the board
        :return: boolean indicating game over, winner of the game, moves possible for the enemy
        """

        # TRUE => GAME FINISHED
        # FALSE => GAME NOT FINISHED
        moves_enemy = self.find_moves(state, enemy, piece_positions, king_positions)
        win_state = None
        game_end = False
        if len(np.where(state == enemy)[0]) == 0 and \
                len(np.where(state == 2 * enemy)[0]) == 0:
            win_state = player
            game_end = True
        elif moves_enemy is None:
            win_state = player
            game_end = True
        elif move_counter >= self.max_moves:  # IF REACHED MOVE LIMIT
            # win_state = self._determine_piece_winner(state, player, enemy)
            win_state = 0
            game_end = True
        return game_end, win_state, moves_enemy

    @staticmethod
    def _give_reward(win_state):
        """
        Grant reward based on the winner of the game (1 or -1),
        if no winner, then there is no reward

        :param win_state: winner of the game (or lack of one)
        :return: reward for the given winner (or lack thereof)
        """

        if win_state in (1, -1):
            reward = 1
        else:
            reward = 0
        return reward

    def step(self, action):
        """
        Perform the provided action and update the game state.
        The player is next given information about the updated state
        and rewarded for the action (used for reinforcement learning)

        :param action: action to be performed
        :return: state of the game after the move, reward for the new state, game over boolean, moves for the next state
        """

        self._move_finder.update_finder(self.board.np_board, action, self.player,
                                       self._enemy, self.moves_list, self._captures_list)
        current_board, new_king_position = self.board.make_move(self.board.np_board, action, self.player,
                                                                self.moves_list, self._captures_list,
                                                                self.board.to_render)
        self.board.update_board(current_board)
        if new_king_position is not None:
            self._move_finder.create_king(new_king_position, self.player)
        self._previous_action = action
        self.move_counter += 1
        game_end, win_state, moves_enemy = self._check_win(self.board.np_board, self.player, self._enemy,
                                                          self.move_counter, self._move_finder.piece_positions,
                                                          self._move_finder.king_positions)
        reward = self._give_reward(win_state)
        self.player = -self.player
        self._enemy = -self._enemy
        return np.copy(self.board.np_board), reward, game_end, moves_enemy
