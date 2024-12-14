import pickle

import numpy as np

from simple_games.connect4.connect4_env.env_parts.connect4_board import Board


class Connect4Env:
    """
    Class implementing the connect4 game, which also functions as the reinforcement learning environment.
    High level game functionality is implemented here,
    the class also acts as an interface for the external player, AI or human,
    with wrapper methods to interact with the Board class.
    """

    def __init__(self, board_parameters, draw_parameters, to_render=False):
        """
        Constructor for the connect4 game class.
        Basic data about the possible actions and states is initialized, which is needed for the agent training.
        The game Board class is also initialized in this constructor, with the provided board and draw
        parameters being passed to the Board

        :param board_parameters: parameters for the Board
        :param draw_parameters: parameters for the Drawer
        :param to_render: render the board flag
        """

        # Load the list for the optimal first 7 moves from any position
        with open('connect4_env/moves_list/optimal_start_moves.pkl', 'rb') as f:
            self.optimal_start_moves = pickle.load(f)
        self.optimal_move_count = 7

        # The starting player is set to 1, the second (enemy) set to -1
        # The values of player and enemy interchange, depending on who is about to play
        self.player = 1
        self.enemy = -1

        # Amount of pieces you need connected to win
        self.piece_count_for_win = 4

        # Information used for reinforcement learning
        # observation space - dimensions of the game state
        self.observation_space = (6, 7)

        # refactored space - modified game state used for model training
        self.refactored_space = (2, 6, 7)

        # action space - number of possible actions
        self.action_space = 42

        # Information about how many moves can be made in one round,
        # and how many moves have been made till now
        self.max_moves = 42
        self.move_counter = 0

        # List of all possible moves
        self.moves = ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
                      (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
                      (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
                      (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
                      (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
                      (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6))

        # Initialize the game Board class
        self.board = Board(board_parameters, draw_parameters, self.moves, to_render)

    def reset(self):
        self.move_counter = 0
        self.player = 1
        self.enemy = -1
        self.board.reset()
        return np.copy(self.board.np_board), self.find_moves(self.board.np_board, 1, positions=None)

    @staticmethod
    def add_heuristics(state, player, possible_moves):
        """
        Approximate the value of a state, used for Monte-Carlo-Search.
        However, it is not used connect4.
        Defined only for compatibility reasons

        :return: 0, as no heuristics provided for connect4
        """

        return 0

    def render(self):
        """
        Wrapper method to render the game board
        """

        self.board.render(self.player)

    def mark_moves(self, action_index):
        """
        Wrapper method to mark the provided moves on the board

        :param action_index: list of indexes of moves from the move list
        """

        self.board.mark_moves(action_index)

    def remove_mark(self, action_index):
        """
        Wrapper method to remove marks from previous moves

        :param action_index: list of indexes of moves from the move list
        """

        self.board.remove_mark(action_index)

    @staticmethod
    def hash_state(state):
        """
        Hash state to be used in the Monte-Carlo-Tree Search.
        Experimental feature ...

        :param state: state of the game
        :return: hash code of the game state
        """

        array = []
        for x in range(6):
            row = tuple(state[x])
            array.append(row)
        array = tuple(array)
        hash_value = hash(array)
        return hash_value

    def check_win(self, state, player):
        """
        Check for game win for the last player based on the provided game state.
        5 conditions checked:
        1. check if in at least 1 row, the required amount of pieces connected by the player -> player wins
        2. check if in at least 1 column, the required amount of pieces connected by the player -> player wins
        3. check if in at least 1 decreasing diagonal,
                 the required amount of pieces connected by the player -> player wins
        4. check if in at least 1 increasing diagonal,
                 the required amount of pieces connected by the player -> player wins
        5. check if board full -> draw

        :param state: state of the game
        :param player: player who made last move
        :return: boolean indicating game over, winner of the game
        """

        # Initialize state as draw with game continuing
        win_state = 0
        game_over = False

        # 1. Check if in any row, the required amount of pieces is connected
        for row in range(6):
            for window_num in range(4):
                # Create an array with the next four cells on the row, starting from window_num
                window = state[row, window_num:window_num + self.piece_count_for_win]
                if np.count_nonzero(window == player) == self.piece_count_for_win:
                    win_state = player
                    game_over = True
                    return game_over, win_state

        # 2. Check if in any column, the required amount of pieces is connected
        for column in range(7):
            for window_num in range(3):
                # Create an array with the next four cells on the column, starting from window_num
                window = state[window_num:window_num + self.piece_count_for_win, column]
                if np.count_nonzero(window == player) == self.piece_count_for_win:
                    win_state = player
                    game_over = True
                    return game_over, win_state

        # 3. Check if in any decreasing diagonal, the required amount of pieces is connected
        for y in range(4):
            for x in range(3):
                # Create an array with the next four cells on the decreasing diagonal, starting from coordinate (x, y)
                diagonal = np.array([state[x, y],
                                     state[1 + x, 1 + y],
                                     state[2 + x, 2 + y],
                                     state[3 + x, 3 + y]])
                if np.count_nonzero(diagonal == player) == self.piece_count_for_win:
                    win_state = player
                    game_over = True
                    return game_over, win_state

        # 4. Check if in any increasing diagonal, the required amount of pieces is connected
        for y in range(4):
            for x in range(3):
                # Create an array with the next four cells on the increasing diagonal, starting from coordinate (x, y)
                diagonal = np.array([state[x, self.observation_space[0] - y],
                                     state[1 + x, (self.observation_space[0] - 1) - y],
                                     state[2 + x, (self.observation_space[0] - 2) - y],
                                     state[3 + x, (self.observation_space[0] - 3) - y]])
                if np.count_nonzero(diagonal == player) == self.piece_count_for_win:
                    win_state = player
                    game_over = True
                    return game_over, win_state

        # 5. Check if the whole board is filled, if so, then end game
        if np.count_nonzero(state != 0) == self.observation_space[0] * self.observation_space[1]:
            win_state = 0
            game_over = True
            return game_over, win_state

        return game_over, win_state

    def make_move(self, state, action, player):
        """
        Wrapper method to make a move on the board.

        :param state: state of the game
        :param action: action to be performed
        :param player: player making the move
        :return: state of the board (NumPy array) after the performed move (from the Board method)
        """

        next_state = self.board.make_move(action, player, np.copy(state), to_render=None)
        return np.copy(next_state)

    def filter_moves(self, moves):
        """
        Map every move from the provided list of moves
        to the corresponding index in the game's list of all moves

        :param moves: list of moves in the format (x, y)
        :return: list of move indexes
        """

        move_index = []
        for move in moves:
            index = self.moves.index(move)
            move_index.append(index)
        move_index.sort()
        return move_index

    @staticmethod
    def find_positions(state):
        """
        Find the position of specific pieces.
        However, it is not used for connect4.
        Defined only for compatibility reasons

        :param state: state of the game
        :return: [None], as method not used connect4
        """

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
        """
        Calculate reward for a given state

        :param state: state of the game
        :param player: player who made last move
        :param move_counter: how many moves have been made in round
                             (not necessary connect4, only for compatibility reasons)
        :return: reward for the state, boolean indicating game over, moves possible for the state
        """

        game_end, win_state = self.check_win(state, player)
        reward = self.give_reward(win_state)
        next_actions = self.find_moves(state, player, positions=None)
        return reward, game_end, next_actions

    @staticmethod
    def give_reward(win_state):
        """
        Grant reward based on the winner of the game (1 or -1),
        if no winner, then there is no reward

        :param win_state: winner of the game (or lack of one)
        :return: reward for the given winner (or lack thereof)
        """

        reward = 0
        if win_state in (1, -1):
            reward = 1
        return reward

    @staticmethod
    def refactor_state(state, player, move_counter):
        """
        Refactor the provided game state for model training.
        On top of the game state there is stacked another 6 x 7 array filled with the player value (1 or -1).
        Additionally, the stacked state is extended to the 4th dimensions, to fit the expected state dimensions
        for model predictions

        :param state: state of the game
        :param player: player who is about to play
        :param move_counter: how many moves have been made in round
                             (not necessary for connect4, only for compatibility reasons)
        :return: refactored state of the game
        """

        refactored_state = np.zeros((1, 2, 6, 7), dtype=np.int32)
        refactored_state[0][0] = state
        refactored_state[0][1] = player
        return refactored_state

    def step(self, action):
        """
        Perform the provided action and update the game state.
        The player is next given information about the updated state
        and rewarded for the action (used for reinforcement learning)

        :param action: action to be performed
        :return: state of the game after move, reward for the new state, game over boolean, moves for the next state
        """

        # Update the board based on the provided move
        self.board.np_board = self.board.make_move(action, self.player, self.board.np_board,
                                                   to_render=self.board.to_render)

        # Get the moves possible from the game state after making the move
        next_actions = self.find_moves(self.board.np_board, self.player, positions=None)

        # Check if game over and who may have won, after making the provided move
        game_end, win_state = self.check_win(self.board.np_board, self.player)

        # Give the reward for the move
        reward = self.give_reward(win_state)

        # Update move counter, after move is made
        self.move_counter += 1

        # Swap the player and enemy values, for next turn (as the player making move switches)
        self.player *= -1
        self.enemy *= -1

        return np.copy(self.board.np_board), reward, game_end, next_actions
