import os

import tensorflow as tf
import numpy as np

from checkers.checkers_env.checkers_env import CheckersEnv
from trainer.monte_carlo_tree_search.monte_carlo_tree_search import MonteCarloTreeSearch
from trainer.self_play_model.self_play_model import SelfPlayModel
from simple_games.tic_tac_toe.tic_tac_toe_env.tic_tac_toe_env import TicTacToeEnv
from simple_games.connect4.connect4_env.connect4_env import Connect4Env


def filter_actions(root, env):
    """
    Get the visit counts of nodes, which determines the probability of choosing the action associated with the node.
    Then normalize the probabilities to 1 by dividing by the sum of visit counts

    :param root: root node of Monte-Carlo-Tree
    :param env: environment used for training
    :return: the normalized probabilities (sum of 1) of choosing an action
    """

    filtered_action_probs = np.zeros(env.action_space)
    for k, v in root.children.items():
        filtered_action_probs[k] = v.visit_count
    filtered_action_probs = filtered_action_probs / np.sum(filtered_action_probs)
    return filtered_action_probs


def finish_episode_np(state_buffer, action_probs_buffer, reward_buffer,
                      train_data, reward, last_player, decay, last_index):
    """
    Add the data generated during an episode to the data arrays

    :param state_buffer: numpy array storing the input to the model (usually game states)
    :param action_probs_buffer: numpy array with action probabilities at a given state
    :param reward_buffer: numpy array with rewards at a given state
    :param train_data: generated training data from one episode
    :param reward: reward for the end of a given episode
    :param last_player: the player who should have played after game over
    :param decay: reward decay rate
    :param last_index: number of generated elements already in the arrays
    :return: the number of generated elements after the adding the episode data
    """

    count = 0

    # Iterate over the list, starting from the last elements - necessary for decay
    for state, player, action_probs, mcts_value in reversed(train_data):
        index = count + last_index

        # Decay the reward, as the reward is most influential towards the end of the game
        adjusted_reward = (reward * (decay ** count))

        # If the data element was generated for the player who lost, then invert the environment reward
        if last_player == player:
            adjusted_reward *= -1

        # Consider the value of the state predicted by the Monte-Carlo-Tree Search by taking the average
        # of the (adjusted) environment reward and the value of the state in Monte-Carlo-Tree
        average_reward = (adjusted_reward + mcts_value) / 2

        # Add the generated elements to the arrays
        state_buffer[index] = state
        action_probs_buffer[index] = action_probs
        reward_buffer[index] = average_reward

        count += 1
    return last_index + count


def generate_data(env_name, board_parameters, draw_parameters, generation_episodes,
                  num_workers, num_simulations, decay, heuristic_weight):
    """

    :param env_name:
    :param board_parameters:
    :param draw_parameters:
    :param generation_episodes:
    :param num_workers:
    :param num_simulations:
    :param decay:
    :param heuristic_weight:
    :return:
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

    num_elements = 0

    # Calculate the episodes which should be run by this worker and calculate the highest possible
    # amount of steps, which could occur to create properly sized arrays
    worker_episodes = generation_episodes // num_workers
    max_steps = env.max_moves * worker_episodes

    # Create arrays for the newly generated data
    generate_data_state = np.zeros((max_steps, *env.refactored_space), dtype=np.float32)
    generate_data_action_probs = np.zeros((max_steps, env.action_space), dtype=np.float32)
    generate_data_reward = np.zeros(max_steps, dtype=np.float32)

    # Load the model used in the data generation
    model = SelfPlayModel("saved_models/data_generation_models/actor-critic-self_play.h5")

    # Run the data generation episodes
    for data_episode in range(worker_episodes):
        # Set up round info
        reward = 0
        done = False
        episode_train_examples = []

        # Reset the environment and get the starting state
        current_state, actions_index = env.reset()
        state_player = env.refactor_state(current_state, env.player, env.move_counter)

        # Initialize the Monte-Carlo-Tree Search for 1 round, the temperature parameter adjusts the randomness of
        # action selection, infinity is random, 0 is deterministic.
        mcts = MonteCarloTreeSearch(env, model, num_simulations, heuristic_weight)
        root = None
        temperature = 2

        # Main loop for 1 round
        while not done:
            # Run a Monte-Carlo-Tree Search simulation to find the optimal action
            root = mcts.run(current_state, state_player, env.player, root)
            action = root.select_action(temperature=temperature)
            action_probs = filter_actions(root, env)

            # Shift the root of the tree to the node of the chosen action
            root = root.children[action]

            # Add the data from the step to the episode data list
            episode_train_examples.append((state_player, env.player, action_probs, root.value()))

            # Update the environment based on the chosen action
            new_state, reward, done, actions_index = env.step(action)

            # After 30 moves make the data generation deterministic (remove randomness)
            if env.move_counter == 30:
                temperature = 0

            # Set the new state as the current state and create a new refactored state
            current_state = new_state
            state_player = env.refactor_state(new_state, env.player, env.move_counter)

        # Add the episode data the data arrays and update the total element count in buffers
        num_elements = finish_episode_np(generate_data_state, generate_data_action_probs, generate_data_reward,
                                         episode_train_examples, reward, env.player, decay, num_elements)
    return generate_data_state, generate_data_action_probs, generate_data_reward, num_elements
