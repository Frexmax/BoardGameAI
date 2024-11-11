import os

import tensorflow as tf

from checkers.checkers_env.checkers_env import CheckersEnv
from simple_games.tic_tac_toe.tic_tac_toe_env.tic_tac_toe_env import TicTacToeEnv
from simple_games.connect4.connect4_env.connect4_env import Connect4Env
from trainer.monte_carlo_tree_search.monte_carlo_tree_search import MonteCarloTreeSearch
from trainer.self_play_model.self_play_model import SelfPlayModel



def tournament_pair(env_name, board_parameters, draw_parameters, num_simulations, heuristic_weight):
    """
    Play 2 games of the currently trained model, against an older model, which till now had the best performance.
    The sides switch in-between rounds.
    Save and return the wins of both models

    :param env_name: name of the environment (game) to be played
    :param board_parameters: parameters for the game board
    :param draw_parameters: parameters for the game drawer
    :param num_simulations: number of simulation steps to be run in the Monte-Carlo-Tree Search
    :param heuristic_weight: weight the heuristics are assigned in the Monte-Carlo-Tree Search
    :return: list with training model win count and old (target) model win count
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

    # Load the training model and the old model to play against each other
    trained_model = SelfPlayModel("saved_models/tournament_models/actor-critic-tournament-target-False.h5")
    target_model = SelfPlayModel("saved_models/tournament_models/actor-critic-tournament-target-True.h5")

    wins_training = 0
    wins_target = 0

    # Run two tournament games
    for game in range(2):
        # Set up round info
        done = False
        reward = 0

        # Reset the environment and get the starting state
        current_state, action_index = env.reset()
        state_player = env.refactor_state(current_state, env.player, env.move_counter)

        # Initialize the Monte-Carlo-Tree Search for 1 round for both models
        mcts_trained = MonteCarloTreeSearch(env, trained_model, num_simulations, heuristic_weight)
        mcts_target = MonteCarloTreeSearch(env, target_model, num_simulations, heuristic_weight)
        root_trained = None
        root_target = None

        # In the first round, the training model is the starting player
        # in the second round, the old model starts
        if game == 0:
            player_trained = 1
        else:
            player_trained = -1

        # Main loop of 1 round
        while not done:
            # Run Monte-Carlo-Tree Search simulation from either trees, based on the player's turn, to
            # find the chosen action
            if env.player == player_trained:
                root_trained = mcts_trained.run(current_state, state_player, env.player, root_trained)
                action = root_trained.select_action(temperature=0)
            else:
                root_target = mcts_target.run(current_state, state_player, env.player, root_target)
                action = root_target.select_action(temperature=0)

            # Shift the root of one of the trees to the node of the chosen action
            if root_trained is not None:
                root_trained = root_trained.children[action]
            if root_target is not None:
                root_target = root_target.children[action]

            # Update the environment based on the chosen action
            new_state, reward, done, action_index = env.step(action)

            # Set the new state as the current state and create a new refactored state
            current_state = new_state
            state_player = env.refactor_state(new_state, env.player, env.move_counter)

        # If the reward is 1, then someone won, therefore update the win statistics
        if reward == 1:
            # If during the first round, after game end, player 1 (training model) was supposed to move,
            # then the old (target) model won, otherwise the training model won
            if game == 0:
                if env.player == 1:
                    wins_target += 1
                else:
                    wins_training += 1
            # The reverse of game 1, if after game end, player 1 (target model) was supposed to move,
            # then the training model won, otherwise the target model won
            else:
                if env.player == 1:
                    wins_training += 1
                else:
                    wins_target += 1
    return [wins_training, wins_target]


def tournament(env, model_trained, model_target, num_games, num_simulations, heuristic_weight, threshold=0.55):
    """
    Play a specified amount of rounds of the currently trained model against an older model,
    which till now had the best performance.
    The sides switch after half of the rounds to be played.

    :param env: environment to be used for the rounds
    :param model_trained: the currently trained model
    :param model_target: an older model, which till now had the best performance
    :param num_games: number of games to be played
    :param num_simulations: number of simulation steps to be run in the Monte-Carlo-Tree Search
    :param heuristic_weight: weight the heuristics are assigned in the Monte-Carlo-Tree Search
    :param threshold: win rate needed for the trained model to determined its victory
    :return: boolean indicating the trained model's victory or not
    """

    wins_training = 0
    wins_target = 0

    # Run the specified number of rounds
    for game in range(num_games):
        # Set up round info
        done = False
        reward = 0

        # Reset the environment and get the starting state
        current_state, action_index = env.reset()
        state_player = env.refactor_state(current_state, env.player, env.move_counter)

        # In the first half of the rounds to be played, the currently trained model is the starting player (red).
        # In the last half, the old model starts
        if game < num_games // 2:
            red_models = model_trained
            black_models = model_target
        else:
            red_models = model_target
            black_models = model_trained

        # Main loop of 1 round
        while not done:
            # If there is only 1 action to be picked, then don't run Monte-Carlo-Tree Search simulation,
            # just pick the available action
            # Otherwise initialize a tree for one of the models, depending whose turn it is
            if len(action_index) == 1:
                action = action_index[0]
            else:
                if env.player == 1:  # PLAYER RED
                    mcts = MonteCarloTreeSearch(env, red_models, num_simulations, heuristic_weight)
                else:  # PLAYER BLACK
                    mcts = MonteCarloTreeSearch(env, black_models, num_simulations, heuristic_weight)

                # Select an action based on the tree simulation
                root = mcts.run(current_state, state_player, env.player)
                action = root.select_action(temperature=0)

            # Update the environment based on the chosen action
            new_state, reward, done, action_index = env.step(action)

            # Set the new state as the current state and create a new refactored state
            current_state = new_state
            state_player = env.refactor_state(new_state, env.player, env.move_counter)

        # If the reward is 1, then someone won, therefore update the win statistics
        if reward == 1:  # WIN RED
            # If during the first half of the rounds, after game end, player 1 (training model) was supposed to move,
            # then the old (target) model won, otherwise the training model won
            if game < num_games // 2:
                if env.player == 1:
                    wins_target += 1
                else:
                    wins_training += 1
            # The reverse of first half, if after game end, player 1 (target model) was supposed to move,
            # then the training model won, otherwise the target model won
            else:
                if env.player == 1:
                    wins_training += 1
                else:
                    wins_target += 1

    # Print the tournament results
    print(f"SCORE: ONLINE: {wins_training} - TARGET: {wins_target}")

    # Return the tournament result
    if wins_training + wins_target == 0 or wins_training / (wins_training + wins_target) > threshold:
        return True
    else:
        return False
