import os
import random

import tensorflow as tf

from checkers.checkers_env.checkers_env import CheckersEnv
from simple_games.tic_tac_toe.tic_tac_toe_env.tic_tac_toe_env import TicTacToeEnv
from simple_games.connect4.connect4_env.connect4_env import Connect4Env
from trainer.monte_carlo_tree_search.monte_carlo_tree_search import MonteCarloTreeSearch
from trainer.self_play_model.self_play_model import SelfPlayModel


def play_test_game_pair(env_name, board_parameters, draw_parameters, num_simulations, heuristic_weight):
    """
    Play 2 test games of the model against a random agent, the sides switch in-between rounds.
    Save and return the win/loss/draw statistics

    :param env_name: name of the environment (game) to be played
    :param board_parameters: parameters for the game board
    :param draw_parameters: parameters for the game drawer
    :param num_simulations: number of simulation steps to be run in the Monte-Carlo-Tree Search
    :param heuristic_weight: weight the heuristics are assigned in the Monte-Carlo-Tree Search
    :return: list with the win/loss/draw counts
    """

    # Disable GPU, force all operations to run on the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    tf.config.set_visible_devices([], 'GPU')

    # Initialize the environment based on the provided environment name
    env = None
    if env_name == "TIC_TAC_TOE":
        env = TicTacToeEnv(board_parameters, draw_parameters)
    elif env_name == "CONNECT4":
        env = Connect4Env(board_parameters, draw_parameters)
    elif env_name == "CHECKERS":
        env = CheckersEnv(board_parameters, draw_parameters)

    # Load the model to be tested
    test_model = SelfPlayModel("saved_models/test_models/actor-critic-test.h5")

    test_model_wins = 0
    test_model_losses = 0
    test_model_draws = 0

    # Run the 2 test games
    for test_game in range(2):
        # Set up round info
        done = False
        reward = 0

        # Reset the environment and get the starting state
        current_state, actions_index = env.reset()
        state_player = env.refactor_state(current_state, env.player, env.move_counter)

        # Initialize the Monte-Carlo-Tree Search for 1 round
        mcts = MonteCarloTreeSearch(env, test_model, num_simulations, heuristic_weight)
        root = None

        # In the first game, the model is always the starting player (red),
        # in the second round, the random agent starts, the model only plays after it (black)
        if test_game == 0:
            red_player = "agent"
            black_player = "random"
        else:
            red_player = "random"
            black_player = "agent"

        # Main loop of 1 round
        while not done:
            # If it's the model's turn, then run a Monte-Carlo-Tree Search simulation to find the optimal action
            # else, it's the random agent's turn, then take a random action
            if (env.player == 1 and red_player == "agent") or \
                    (env.player == -1 and black_player == "agent"):
                root = mcts.run(current_state, state_player, env.player, root)
                action = root.select_action(temperature=0)
            else:
                action = random.sample(actions_index, 1)[0]

            # Shift the root of the tree to the node of the chosen action
            if root is not None:
                root = root.children[action]

            # Update the environment based on the chosen action
            new_state, reward, done, actions_index = env.step(action)

            # Set the new state as the current state and create a new refactored state
            current_state = new_state
            state_player = env.refactor_state(new_state, env.player, env.move_counter)

        # If the end reward is 1, then someone won, update win/loss statistics
        # if the reward is 0, then the game was a draw
        if reward == 1:
            # If after game end, and the model was supposed to move,
            # then that means that the random agent made the winning move and the model lost
            # otherwise, the model must have won
            if (red_player == "agent" and env.player == -1) or \
                    (black_player == "agent" and env.player == 11):
                test_model_losses += 1
            else:
                test_model_wins += 1
        else:
            test_model_draws += 1
    return [test_model_wins, test_model_losses, test_model_draws]
