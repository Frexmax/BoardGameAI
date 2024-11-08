import pickle

import numpy as np
from numba import jit

from checkers.checkers_env.env_parts.checkers_board import Board
from checkers.checkers_env.env_parts.move_finder import MoveFinder


class CheckersEnv:
    def __init__(self, board_parameters, draw_parameters, to_render=False):
        with open('checkers_env/moves_list/captures_list.pkl', 'rb') as f:
            self.captures_list = pickle.load(f)

        with open('checkers_env/moves_list/moves_list.pkl', 'rb') as f:
            self.moves_list = pickle.load(f)

        self.optimal_move_count = -1
        self.board = Board(board_parameters, draw_parameters, to_render)
        self.move_finder = MoveFinder()

        # ACTION AND OBSERVATION SPACE
        self.action_space = 2852  # NUM MOVES
        self.observation_space = (8, 8)  # STATE SHAPE
        self.refactored_space = (6, 8, 8)

        self.player = 1
        self.enemy = -1
        self.max_moves = 200
        self.move_counter = 0
        self.previous_action = None

        # HEURISTICS
        self.piece_value = 5
        self.king_value = 10

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

        self.piece_value_map = {1: piece_value_map_red, -1: piece_value_map_black}
        self.king_value_map = king_value_map

    def reset(self):
        self.player = 1
        self.enemy = -1
        self.board.reset()
        self.move_finder.initialized_positions(self.board.np_board)
        self.move_counter = 0
        moves_player = self.move_finder.find_moves(self.board.np_board, self.player, self.move_finder.piece_positions,
                                                   self.move_finder.king_positions, self.moves_list)
        return np.copy(self.board.np_board), moves_player

    def add_heuristics(self, state, player, possible_moves):
        def check_supported(state, x, y, player, directions):
            support_value = 0
            for direction in directions[-(player - 1) // 2]:
                x_n = x + direction[0]
                y_n = y + direction[1]
                if (8 > x_n >= 0 and 0 <= y_n < 8) and (state[x_n, y_n] == player or state[x_n, y_n] == 2 * player):
                    support_value += 0.025
            return support_value

        def calculate_value(king_value, piece_value, player_map, enemy_map, king_map,
                            state, player, captured_positions, directions):
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
                captures = self.captures_list[move_index]
                for capture in captures:
                    captured_positions.append(capture)
        captured_positions = tuple(captured_positions)
        p_value, e_value = calculate_value(self.king_value, self.piece_value, self.piece_value_map[player],
                                           self.piece_value_map[-player], self.king_value_map,
                                           state, player, captured_positions, directions)
        return -(p_value - e_value) / (p_value + e_value)

    def render(self):
        self.board.render(self.player)

    def highlight_piece(self, click_position, highlighted_before, possible_moves):
        return self.board.highlight_piece(click_position, self.player, highlighted_before,
                                          possible_moves, self.moves_list)

    def end_highlight(self, highlighted_before):
        self.board.end_highlight(highlighted_before, self.player)

    @staticmethod
    def hash_state(state):
        array = []
        for x in range(8):
            row = tuple(state[x])
            array.append(row)
        array = tuple(array)
        hash_value = hash(array)
        return hash_value

    @staticmethod
    def flip_board_perspective(state):
        return np.rot90(np.copy(state)) * -1

    @staticmethod
    @jit(nopython=True)
    def refactor_state(state, player, move_counter):
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
    def string_representation(state):
        return np.copy(state).tostring()

    def find_positions(self, state):
        return self.move_finder.find_positions(state)

    def find_moves(self, state, player, piece_positions, king_positions):
        return self.move_finder.find_moves(state, player, piece_positions, king_positions, self.moves_list)

    def make_move(self, state, action, player):
        next_state, _ = self.board.make_move(np.copy(state), action, player, self.moves_list,
                                             self.captures_list, to_render=None)
        return np.copy(next_state)

    def state_reward(self, state, player, move_counter):
        piece_positions, king_positions = self.find_positions(state)
        game_end, win_state, moves_enemy = self.check_win(state, player, -player, move_counter,
                                                          piece_positions, king_positions)
        reward = self.give_reward(win_state)
        return reward, game_end, moves_enemy

    @staticmethod
    def determine_piece_winner(state, player, enemy):
        def count_pieces(state, player, enemy):
            return len(np.where(state == player)[0]), len(np.where(state == enemy)[0])

        def count_kings(state, player, enemy):
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

    def check_win(self, state, player, enemy, move_counter, piece_positions, king_positions):
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
            # win_state = self.determine_piece_winner(state, player, enemy)
            win_state = 0
            game_end = True
        return game_end, win_state, moves_enemy

    @staticmethod
    def give_reward(win_state):
        if win_state in (1, -1):
            reward = 1
        else:
            reward = 0
        return reward

    def step(self, action):
        self.move_finder.update_finder(self.board.np_board, action, self.player,
                                       self.enemy, self.moves_list, self.captures_list)
        current_board, new_king_position = self.board.make_move(self.board.np_board, action, self.player,
                                                                self.moves_list, self.captures_list,
                                                                self.board.to_render)
        self.board.update_board(current_board)
        if new_king_position is not None:
            self.move_finder.create_king(new_king_position, self.player)
        self.previous_action = action
        self.move_counter += 1
        game_end, win_state, moves_enemy = self.check_win(self.board.np_board, self.player, self.enemy,
                                                          self.move_counter, self.move_finder.piece_positions,
                                                          self.move_finder.king_positions)
        reward = self.give_reward(win_state)
        self.player = -self.player
        self.enemy = -self.enemy
        return np.copy(self.board.np_board), reward, game_end, moves_enemy
