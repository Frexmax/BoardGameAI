import numpy as np
from copy import deepcopy
from env_parts.tic_tac_toe_board import Board


class TicTacToeEnv:
    def __init__(self, board_parameters, draw_parameters, to_render=False):
        self.optimal_move_count = -1
        self.move_counter = 0
        self.player = 1
        self.enemy = -1
        self.observation_space = (3, 3)
        self.refactored_space = (2, 3, 3)
        self.action_space = 9  # NUM MOVES
        self.max_moves = 9
        self.moves = ((0, 0), (0, 1), (0, 2),
                      (1, 0), (1, 1), (1, 2),
                      (2, 0), (2, 1), (2, 2))
        self.board = Board(board_parameters, draw_parameters, self.moves, to_render)

    def render(self):
        self.board.render(self.player)

    def mark_moves(self, action_index):
        self.board.mark_moves(action_index)

    def remove_mark(self, action_index, action):
        self.board.remove_mark(action_index, action)

    def reset(self):
        self.move_counter = 0
        self.player = 1
        self.enemy = -1
        self.board.reset()
        return np.copy(self.board.np_board), self.find_moves(self.board.np_board, 1, positions=None)

    def add_heuristics(self):
        return 0

    def find_positions(self, state):
        return [None]

    def find_moves(self, state, player, positions):
        moves = []
        for x in range(3):
            for y in range(3):
                if state[x, y] == 0:
                    moves.append((x, y))
        return self.filter_moves(moves)

    @staticmethod
    def hash_state(state):
        array = []
        for x in range(3):
            row = tuple(state[x])
            array.append(row)
        array = tuple(array)
        hash_value = hash(array)
        return hash_value

    def filter_moves(self, moves):
        move_index = []
        for move in moves:
            index = self.moves.index(move)
            move_index.append(index)
        move_index.sort()
        return move_index

    def make_move(self, state, action, player, to_render=False):
        return self.board.make_move(state, action, player, to_render=to_render)

    @staticmethod
    def check_win(state, player):
        win_state = 0
        # CHECK ROWS
        for x in range(3):
            if np.count_nonzero(state[x] == player) == 3:
                win_state = player
                return True, win_state

        # CHECK COLUMNS
        for y in range(3):
            if np.count_nonzero(state[:, y] == player) == 3:
                win_state = player
                return True, win_state

        # CHECK DIAGONAL 1
        if np.count_nonzero(np.array([state[0, 0], state[1, 1], state[2, 2]]) == player) == 3:
            win_state = player
            return True, win_state

        # CHECK DIAGONAL 2
        if np.count_nonzero(np.array([state[0, 2], state[1, 1], state[2, 0]]) == player) == 3:
            win_state = player
            return True, win_state

        # CHECK FILLED
        if np.count_nonzero(state != 0) == 9:
            win_state = 0
            return True, win_state
        return False, win_state

    def state_reward(self, state, player, move_counter):
        game_end, win_state = self.check_win(state, player)
        reward = self.give_reward(win_state)
        next_actions = self.find_moves(state, player, positions=None)
        return reward, game_end, next_actions

    @staticmethod
    def refactor_state(state, player, move_counter):
        refactored_state = np.zeros((1, 2, 3, 3), dtype=np.int32)
        refactored_state[0][0] = state
        refactored_state[0][1] = player
        return refactored_state

    @staticmethod
    def flip_board_perspective(state, player):
        return deepcopy(state) * player

    @staticmethod
    def string_representation(state):
        return state.tostring()

    @staticmethod
    def give_reward(win_state):
        reward = 0
        if win_state in (1, -1):
            reward = 1
        return reward

    def step(self, action):
        self.board.np_board = self.board.make_move(self.board.np_board, action,
                                                   self.player, to_render=self.board.to_render)
        next_actions = self.find_moves(self.board.np_board, self.player, positions=None)
        game_end, win_state = self.check_win(self.board.np_board, self.player)
        reward = self.give_reward(win_state)
        self.move_counter += 1
        self.player *= -1
        self.enemy *= -1
        return np.copy(self.board.np_board), reward, game_end, next_actions
