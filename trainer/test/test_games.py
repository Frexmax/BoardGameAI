import os
import random
import tensorflow as tf
from TicTacToeEnv import TicTacToeEnv
from Connect4Env import Connect4Env
from CheckersEnv import CheckersEnv
from MonteCarloTreeSearch import MonteCarloTreeSearch
from SelfPlayModel import SelfPlayModel


def play_test_game_pair(env_name, board_parameters, draw_parameters, num_simulations, heuristic_weight):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    tf.config.set_visible_devices([], 'GPU')

    env = None
    if env_name == "TIC_TAC_TOE":
        env = TicTacToeEnv(board_parameters, draw_parameters)
    elif env_name == "CONNECT4":
        env = Connect4Env(board_parameters, draw_parameters)
    elif env_name == "CHECKERS":
        env = CheckersEnv(board_parameters, draw_parameters)

    test_model_wins = 0
    test_model_losses = 0
    test_model_draws = 0
    test_model = SelfPlayModel("saved_models/test_models/actor-critic-test.h5")
    for test_game in range(2):
        done = False
        reward = 0
        current_state, actions_index = env.reset()
        state_player = env.refactor_state(current_state, env.player, env.move_counter)
        mcts = MonteCarloTreeSearch(env, test_model, num_simulations, heuristic_weight)
        root = None
        if test_game == 0:
            red_player = "agent"
            black_player = "random"
        else:
            red_player = "random"
            black_player = "agent"
        while not done:
            if (env.player == 1 and red_player == "agent") or \
                    (env.player == -1 and black_player == "agent"):
                root = mcts.run(current_state, state_player, env.player, root)
                action = root.select_action(temperature=0)
            else:
                action = random.sample(actions_index, 1)[0]
            if root is not None:
                root = root.children[action]
            new_state, reward, done, actions_index = env.step(action)
            current_state = new_state
            state_player = env.refactor_state(new_state, env.player, env.move_counter)
        if reward == 1:
            if (red_player == "agent" and env.player == 1) or \
                    (black_player == "agent" and env.player == -1):
                test_model_losses += 1
            else:
                test_model_wins += 1
        else:
            test_model_draws += 1
    return [test_model_wins, test_model_losses, test_model_draws]
