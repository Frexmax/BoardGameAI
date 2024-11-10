from copy import deepcopy

import numpy as np

from simple_games.tic_tac_toe.tic_tac_toe_env.env_parts.tic_tac_toe_board import Board


class TicTacToeEnv:
    """
    Class implementing the tic-tac-toe game, which also functions as the reinforcement learning environment.
    High level game functionality is implemented here, the class also acts as an interface for the external player, AI or human, 
    with wrapper methods to interact with the Board class.
    """

    def __init__(self, board_parameters, draw_parameters, to_render=False):
        """
        Constructor for the tic-tac-toe game class.
        Basic data about the possible actions and states is initialized, which is needed for the agent training.
        The game Board class is also initialized in this constructor, with the provided board and draw
        parameters being passed to the Board

        :param board_parameters: parameters for the Board
        :param draw_parameters: parameters for the Drawer
        :param to_render: render the board flag
        """

        # The starting player is set to 1, the second (enemy) set to -1
        # The values of player and enemy interchange, depending on who is about to play
        self.player = 1
        self.enemy = -1

        # Information used for reinforcement learning
        # observation space - dimensions of the game state
        self.observation_space = (3, 3)
        # refactored space - modified game state used for model training
        self.refactored_space = (2, 3, 3)
        # action space - number of possible actions
        self.action_space = 9

        self.max_moves = 9
        self.move_counter = 0
        self.optimal_move_count = -1

        # List of all possible moves
        self.moves = ((0, 0), (0, 1), (0, 2),
                      (1, 0), (1, 1), (1, 2),
                      (2, 0), (2, 1), (2, 2))

        # Initialize the game Board class
        self.board = Board(board_parameters, draw_parameters, self.moves, to_render)

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

    def reset(self):
        """
        Reset the game state when starting a new round.
        Reset the board and make player 1 play the next turn

        :return: game state of the reset board, possible moves to play
        """

        self.move_counter = 0
        self.player = 1
        self.enemy = -1
        self.board.reset()
        return np.copy(self.board.np_board), self.find_moves(self.board.np_board, 1, positions=None)

    @staticmethod
    def add_heuristics():
        """
        Approximate the value of a state, used for Monte-Carlo-Search.
        However, it is not used for tic-tac-toe.
        Defined only for compatibility reasons

        :return: 0, as no heuristics provided for tic-tac-toe
        """

        return 0

    @staticmethod
    def find_positions(state):
        """
        Find the position of specific pieces.
        However, it is not used for tic-tac-toe.
        Defined only for compatibility reasons

        :param state: state of the game
        :return: [None], as method not used for tic-tac-toe
        """

        return [None]

    def find_moves(self, state, player, positions):
        """
        Find the possible moves from a game state.
        A valid move is the coordinate of an empty grid cell

        :param state: state of the game
        :param player: player who is now to play (not necessary for tic-tac-toe, only for compatibility reasons)
        :param positions: position of pieces (not necessary for tic-tac-toe, only for compatibility reasons)
        :return: list of possible moves for the provided game state
        """

        moves = []
        for x in range(3):
            for y in range(3):
                if state[x, y] == 0:
                    moves.append((x, y))
        return self.filter_moves(moves)

    @staticmethod
    def hash_state(state):
        """
        Hash state to be used in the Monte-Carlo-Tree Search.
        Experimental feature ...

        :param state: state of the game
        :return: hash code of the game state
        """

        array = []
        for x in range(3):
            row = tuple(state[x])
            array.append(row)
        array = tuple(array)
        hash_value = hash(array)
        return hash_value

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

    def make_move(self, state, action, player, to_render=False):
        """
        Wrapper method to make a move on the board.

        :param state: state of the game
        :param action: action to be performed
        :param player: player making the move
        :param to_render: render the board flag
        :return: state of the board (NumPy array) after the performed move (from the Board method)
        """

        return self.board.make_move(state, action, player, to_render=to_render)

    @staticmethod
    def check_win(state, player):
        """
        Check for game win for the last player based on the provided game state.
        5 conditions checked:
        1. check if row filled by the player -> player wins
        2. check if column filled by the player -> player wins
        3. check if the descending diagonal filled by the player -> player wins
        4. check if the ascending diagonal filled by the player -> player wins
        5. check if board full -> draw

        :param state: state of the game
        :param player: player who made last move
        :return: boolean indicating game over, winner of the game
        """

        win_state = 0

        # 1. Check if row filled by player
        for x in range(3):
            if np.count_nonzero(state[x] == player) == 3:
                win_state = player
                return True, win_state

        # 2 .Check if column filled by player
        for y in range(3):
            if np.count_nonzero(state[:, y] == player) == 3:
                win_state = player
                return True, win_state

        # 3. Check if descending diagonal filled by player
        if np.count_nonzero(np.array([state[0, 0], state[1, 1], state[2, 2]]) == player) == 3:
            win_state = player
            return True, win_state

        # 4. Check if ascending diagonal filled by player
        if np.count_nonzero(np.array([state[0, 2], state[1, 1], state[2, 0]]) == player) == 3:
            win_state = player
            return True, win_state

        # 5. Check if the board is full
        if np.count_nonzero(state != 0) == 9:
            win_state = 0
            return True, win_state
        return False, win_state

    def state_reward(self, state, player, move_counter):
        """
        Calculate reward for a given state

        :param state: state of the game
        :param player: player who made last move
        :param move_counter: how many moves have been made in round
                             (not necessary for tic-tac-toe, only for compatibility reasons)
        :return: reward for the state, boolean indicating game over, moves possible for the state
        """

        game_end, win_state = self.check_win(state, player)
        reward = self.give_reward(win_state)
        next_actions = self.find_moves(state, player, positions=None)
        return reward, game_end, next_actions

    @staticmethod
    def refactor_state(state, player, move_counter):
        """
        Refactor the provided game state for model training.
        On top of the game state there is stacked another 3 x 3 array filled with the player value (1 or -1).
        Additionally, the stacked state is extended to the 4th dimensions, to fit the expected state dimensions
        for model predictions

        :param state: state of the game
        :param player: player who is about to play
        :param move_counter: how many moves have been made in round
                             (not necessary for tic-tac-toe, only for compatibility reasons)
        :return: refactored state of the game
        """

        refactored_state = np.zeros((2, 3, 3), dtype=np.int32)
        refactored_state[0] = state
        refactored_state[1] = player
        return np.expand_dims(refactored_state, axis=0)

    @staticmethod
    def flip_board_perspective(state, player):
        """
        Create a new state with a flipped player perspective.
        Previous 1's become -1, and vice versa.

        :param state: state of the game
        :param player: player who is about to play
        :return:
        """

        return deepcopy(state) * player

    @staticmethod
    def string_representation(state):
        """
        Convert the game state array to a string

        :param state: state of the game
        :return: string representing the provided game state
        """

        return np.array2string(state)

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

    def step(self, action):
        """
        Perform the provided action and update the game state.
        The player is next given information about the updated state
        and rewarded for the action (used for reinforcement learning)

        :param action: action to be performed
        :return: state of the game after move, reward for the new state, game over boolean, moves for the next state
        """

        # Update the board based on the provided move
        self.board.np_board = self.board.make_move(self.board.np_board, action,
                                                   self.player, to_render=self.board.to_render)

        # Get the moves possible from the game state after making the move
        next_actions = self.find_moves(self.board.np_board, self.player, positions=None)

        # Check if game over and who may have won, after making the provided move
        game_end, win_state = self.check_win(self.board.np_board, self.player)

        # Give the reward for the move
        reward = self.give_reward(win_state)

        # Update move counter, after move made
        self.move_counter += 1

        # Swap the player and enemy values, for next turn (as the player making move switches)
        self.player *= -1
        self.enemy *= -1

        return np.copy(self.board.np_board), reward, game_end, next_actions
