import time
import random
import pygame as pg

from simple_games.tic_tac_toe.tic_tac_toe_env.tic_tac_toe_env import TicTacToeEnv
from trainer.self_play_model.self_play_model import SelfPlayModel
from trainer.monte_carlo_tree_search.monte_carlo_tree_search import MonteCarloTreeSearch
from simple_games.tic_tac_toe.tic_tac_toe_env.env_parameters.tic_tac_toe_env_parameters import board_parameters, draw_parameters


def test_tic_tac_toe(human, episodes, num_simulations, ai_player, model_name):
    env = TicTacToeEnv(board_parameters, draw_parameters, to_render=True)
    model = SelfPlayModel(f"./tic_tac_toe/saved_models/{model_name}")
    for episode in range(episodes):
        done = False
        current_state, actions_index = env.reset()
        state_player = env.refactor_state(current_state, env.player, env.move_counter)
        reward = 0
        env.render()
        while not done:
            if env.player == ai_player:
                mcts = MonteCarloTreeSearch(env, model, num_simulations, heuristic_weight=0)
                root = mcts.run(current_state, state_player, env.player)
                action = root.select_action(temperature=0)
            else:
                if not human:
                    action = random.sample(actions_index, 1)[0]
                    time.sleep(1)
                else:
                    action = None
                    moved = False
                    env.mark_moves(actions_index)
                    env.render()
                    while not moved:
                        events = pg.event.get()
                        for event in events:
                            if event.type == pg.MOUSEBUTTONUP:
                                mouse_position = pg.mouse.get_pos()
                                coord_x = mouse_position[0] // env.board.cell_size[0]
                                coord_y = mouse_position[1] // env.board.cell_size[1]
                                coords = (coord_y, coord_x)
                                if len(actions_index) > 0:
                                    for move in actions_index:
                                        if env.moves[move] == coords:
                                            action = move
                                            moved = True
                                            break
                        pg.event.clear()
                    env.remove_mark(actions_index)
            new_state, reward, done, actions_index = env.step(action)
            current_state = new_state
            state_player = env.refactor_state(new_state, env.player, env.move_counter)
            env.render()
        print(f"LAST PLAYER: {env.player}, REWARD: {reward}")
        print("=============================================")
        time.sleep(3)
