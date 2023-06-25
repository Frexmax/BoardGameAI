import os
import tensorflow as tf
from TicTacToeEnv import TicTacToeEnv
from Connect4Env import Connect4Env
from CheckersEnv import CheckersEnv
from MonteCarloTreeSearch import MonteCarloTreeSearch
from SelfPlayModel import SelfPlayModel


def tournament_pair(env_name, board_parameters, draw_parameters, num_simulations, heuristic_weight):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    tf.config.set_visible_devices([], 'GPU')

    env = None
    if env_name == "TIC_TAC_TOE":
        env = TicTacToeEnv(board_parameters, draw_parameters)
    elif env_name == "CONNECT4":
        env = Connect4Env(board_parameters, draw_parameters)
    elif env_name == "CHECKERS":
        env = CheckersEnv(board_parameters, draw_parameters)

    wins_training = 0
    wins_target = 0
    trained_model = SelfPlayModel("SavedModels/TournamentModels/actor-critic-tournament-target-False.h5")
    target_model = SelfPlayModel("SavedModels/TournamentModels/actor-critic-tournament-target-True.h5")
    for game in range(2):
        done = False
        reward = 0
        current_state, action_index = env.reset()
        state_player = env.refactor_state(current_state, env.player, env.move_counter)
        mcts_trained = MonteCarloTreeSearch(env, trained_model, num_simulations, heuristic_weight)
        mcts_target = MonteCarloTreeSearch(env, target_model, num_simulations, heuristic_weight)
        root_trained = None
        root_target = None
        if game == 0:
            player_trained = 1
        else:
            player_trained = -1
        while not done:
            if env.player == player_trained:  # TRAINED NETWORK
                root_trained = mcts_trained.run(current_state, state_player, env.player, root_trained)
                action = root_trained.select_action(temperature=0)
            else:
                root_target = mcts_target.run(current_state, state_player, env.player, root_target)
                action = root_target.select_action(temperature=0)
            if root_trained is not None:
                root_trained = root_trained.children[action]
            if root_target is not None:
                root_target = root_target.children[action]
            new_state, reward, done, action_index = env.step(action)
            current_state = new_state
            state_player = env.refactor_state(new_state, env.player, env.move_counter)
        if reward == 1:  # WIN RED
            if game == 0:  # ONLINE == RED
                if env.player == 1:  # RED LOST
                    wins_target += 1
                else:
                    wins_training += 1
            else:  # ONLINE == BLACK
                if env.player == 1:  # RED LOST
                    wins_training += 1
                else:
                    wins_target += 1
    return [wins_training, wins_target]


def tournament(env, model_trained, model_target, num_games, num_simulations, heuristic_weight, threshold=0.55):
    wins_training = 0
    wins_target = 0
    for game in range(num_games):
        done = False
        reward = 0
        current_state, action_index = env.reset()
        state_player = env.refactor_state(current_state, env.player, env.move_counter)
        if game < num_games // 2:
            red_models = model_trained
            black_models = model_target
        else:
            red_models = model_target
            black_models = model_trained
        while not done:
            if len(action_index) == 1:
                action = action_index[0]
            else:
                if env.player == 1:  # PLAYER RED
                    mcts = MonteCarloTreeSearch(env, red_models, num_simulations, heuristic_weight)
                else:  # PLAYER BLACK
                    mcts = MonteCarloTreeSearch(env, black_models, num_simulations, heuristic_weight)
                root = mcts.run(current_state, state_player, env.player)
                action = root.select_action(temperature=0)
            new_state, reward, done, action_index = env.step(action)
            current_state = new_state
            state_player = env.refactor_state(new_state, env.player, env.move_counter)
        if reward == 1:  # WIN RED
            if game < num_games // 2:  # ONLINE == RED
                if env.player == 1:  # RED LOST
                    wins_target += 1
                else:
                    wins_training += 1
            else:  # ONLINE == BLACK
                if env.player == 1:  # RED LOST
                    wins_training += 1
                else:
                    wins_target += 1
    print(f"SCORE: ONLINE: {wins_training} - TARGET: {wins_target}")
    if wins_training + wins_target == 0 or wins_training / (wins_training + wins_target) > threshold:
        return True
    else:
        return False
