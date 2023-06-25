import pickle
import numpy as np
from Connect4Board import Board
from numba import jit


class Connect4Env:
    def __init__(self, board_parameters, draw_parameters, to_render=False):
        with open('Connect4Env/MovesList/OptimalStartMovesV3.pkl', 'rb') as f:
            self.optimal_start_moves = pickle.load(f)
        self.optimal_move_count = 7

        self.player = 1
        self.enemy = -1
        self.move_counter = 0
        self.observation_space = (6, 7)
        self.refactored_space = (2, 6, 7)
        self.action_space = 42
        self.max_moves = 42
        self.moves = ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
                      (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
                      (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
                      (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
                      (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
                      (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6))
        self.board = Board(board_parameters, draw_parameters, self.moves, to_render)

    def reset(self):
        self.move_counter = 0
        self.player = 1
        self.enemy = -1
        self.board.reset()
        return np.copy(self.board.np_board), self.find_moves(self.board.np_board, 1, positions=None)

    def add_heuristics(self, state, player, possible_moves):
        return 0

    def start_moves(self):
        pass

    def render(self):
        self.board.render(self.player)

    def mark_moves(self, action_index):
        self.board.mark_moves(action_index)

    def remove_mark(self, action_index, action):
        self.board.remove_mark(action_index, action)

    @staticmethod
    def hash_state(state):
        array = []
        for x in range(6):
            row = tuple(state[x])
            array.append(row)
        array = tuple(array)
        hash_value = hash(array)
        return hash_value

    @staticmethod
    @jit(nopython=True)
    def check_win(state, player):
        win_state = 0
        # CHECK ALL ROWS:
        window_start = 0
        window_end = 4
        for row in range(6):
            for window_num in range(4):
                window = state[row, window_start + window_num:window_end + window_num]
                if np.count_nonzero(window == player) == 4:
                    win_state = player
                    return True, win_state

        # CHECK ALL COLUMNS
        window_start = 0
        window_end = 4
        for column in range(7):
            for window_num in range(3):
                window = state[window_start + window_num:window_end + window_num, column]
                if np.count_nonzero(window == player) == 4:
                    win_state = player
                    return True, win_state

        # DIAGONAL 1
        for y in range(4):
            for x in range(3):
                diagonal = np.array([state[x, y], state[1 + x, 1 + y], state[2 + x, 2 + y], state[3 + x, 3 + y]])
                if np.count_nonzero(diagonal == player) == 4:
                    win_state = player
                    return True, win_state

        # DIAGONAL 2
        for y in range(4):
            for x in range(3):
                diagonal = np.array([state[x, 6 - y], state[1 + x, 5 - y], state[2 + x, 4 - y], state[3 + x, 3 - y]])
                if np.count_nonzero(diagonal == player) == 4:
                    win_state = player
                    return True, win_state

        # CHECK FILLED
        if np.count_nonzero(state != 0) == 42:
            win_state = 0
            return True, win_state
        return False, win_state

    def make_move(self, state, action, player):
        next_state = self.board.make_move(action, player, np.copy(state), to_render=None)
        return np.copy(next_state)

    def filter_moves(self, moves):
        move_index = []
        for move in moves:
            index = self.moves.index(move)
            move_index.append(index)
        move_index.sort()
        return move_index

    def find_positions(self, state):
        return [None]

    def find_moves(self, state, player, positions):
        moves = []
        for column in range(7):
            for row in reversed(range(6)):
                if state[row, column] == 0:
                    moves.append((row, column))
                    break
        return self.filter_moves(moves)

    def state_reward(self, state, player, move_counter):
        game_end, win_state = self.check_win(state, player)
        reward = self.give_reward(win_state)
        next_actions = self.find_moves(state, player, positions=None)
        return reward, game_end, next_actions

    @staticmethod
    def give_reward(win_state):
        reward = 0
        if win_state in (1, -1):
            reward = 1
        return reward

    @staticmethod
    def refactor_state(state, player, move_counter):
        refactored_state = np.zeros((1, 2, 6, 7), dtype=np.int32)
        refactored_state[0][0] = state
        refactored_state[0][1] = player
        return refactored_state

    def step(self, action):
        self.board.np_board = self.board.make_move(action, self.player, self.board.np_board,
                                                   to_render=self.board.to_render)
        next_actions = self.find_moves(self.board.np_board, self.player, positions=None)
        game_end, win_state = self.check_win(self.board.np_board, self.player)
        reward = self.give_reward(win_state)
        self.player *= -1
        self.enemy *= -1
        self.move_counter += 1
        return np.copy(self.board.np_board), reward, game_end, next_actions
