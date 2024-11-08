import os

import tensorflow as tf
import numpy as np

from checkers.checkers_env.checkers_env import CheckersEnv
from trainer.monte_carlo_tree_search.monte_carlo_tree_search import MonteCarloTreeSearch
from trainer.self_play_model.self_play_model import SelfPlayModel
from simple_games.tic_tac_toe.tic_tac_toe_env.tic_tac_toe_env import TicTacToeEnv
from simple_games.connect4.connect4_env.connect4_env import Connect4Env


def filter_actions(root, env):
    filtered_action_probs = np.zeros(env.action_space)
    for k, v in root.children.items():
        filtered_action_probs[k] = v.visit_count
    filtered_action_probs = filtered_action_probs / np.sum(filtered_action_probs)
    return filtered_action_probs


def finish_episode(data_buffer, train_data, reward, last_player, decay):
    count = 0
    for state, player, action_probs, mcts_value in reversed(train_data):
        adjusted_reward = (reward * (decay ** count))
        if last_player == player:
            adjusted_reward *= -1
        average_reward = (adjusted_reward + mcts_value) / 2
        data_buffer.append((state, action_probs, average_reward))
        count += 1
    return data_buffer


def finish_episode_np(state_buffer, action_probs_buffer, reward_buffer,
                      train_data, reward, last_player, decay, last_index):
    count = 0
    for state, player, action_probs, mcts_value in reversed(train_data):
        index = count + last_index
        adjusted_reward = (reward * (decay ** count))
        if last_player == player:
            adjusted_reward *= -1
        average_reward = (adjusted_reward + mcts_value) / 2

        # ADD COLLECTED ELEMENTS TO BUFFER
        state_buffer[index] = state
        action_probs_buffer[index] = action_probs
        reward_buffer[index] = average_reward
        count += 1
    return last_index + count


def generate_data(env_name, board_parameters, draw_parameters, generation_episodes,
                  num_workers, num_simulations, decay, heuristic_weight):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    tf.config.set_visible_devices([], 'GPU')
    # CREATE ENV FOR GENERATION
    env = None
    if env_name == "TIC_TAC_TOE":
        env = TicTacToeEnv(board_parameters, draw_parameters)
    elif env_name == "CONNECT4":
        env = Connect4Env(board_parameters, draw_parameters)
    elif env_name == "CHECKERS":
        env = CheckersEnv(board_parameters, draw_parameters)

    num_elements = 0
    worker_episodes = generation_episodes // num_workers
    max_steps = env.max_moves * worker_episodes

    # GENERATED DATA ARRAY
    generate_data_state = np.zeros((max_steps, *env.refactored_space), dtype=np.float32)
    generate_data_action_probs = np.zeros((max_steps, env.action_space), dtype=np.float32)
    generate_data_reward = np.zeros(max_steps, dtype=np.float32)
    model = SelfPlayModel("saved_models/data_generation_models/actor-critic-self_play.h5")
    for data_episode in range(generation_episodes // num_workers):
        reward = 0
        done = False
        current_state, actions_index = env.reset()
        state_player = env.refactor_state(current_state, env.player, env.move_counter)

        root = None
        mcts = MonteCarloTreeSearch(env, model, num_simulations, heuristic_weight)
        episode_train_examples = []
        temperature = 2
        while not done:
            # RUN MONTE CARLO SEARCH
            root = mcts.run(current_state, state_player, env.player, root)
            action = root.select_action(temperature=temperature)
            action_probs = filter_actions(root, env)
            root = root.children[action]
            episode_train_examples.append((state_player, env.player, action_probs, root.value()))

            # PERFORM ACTION
            new_state, reward, done, actions_index = env.step(action)

            if env.move_counter == 30:
                temperature = 0
            current_state = new_state
            state_player = env.refactor_state(new_state, env.player, env.move_counter)
        num_elements = finish_episode_np(generate_data_state, generate_data_action_probs, generate_data_reward,
                                         episode_train_examples, reward, env.player, decay, num_elements)
    return generate_data_state, generate_data_action_probs, generate_data_reward, num_elements
