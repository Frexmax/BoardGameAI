import time
import pickle
import random

import pygame as pg
from pygame.locals import KEYDOWN, K_ESCAPE

from checkers.checkers_env.checkers_env import CheckersEnv
from checkers.checkers_env.env_parameters.checkers_env_parameters import board_parameters, draw_parameters
from trainer.self_play_model.self_play_model import SelfPlayModel
from trainer.monte_carlo_tree_search.monte_carlo_tree_search import MonteCarloTreeSearch


def test_checkers(human, episodes, num_simulations, ai_player, heuristic_weight, model_name):
    env = CheckersEnv(board_parameters, draw_parameters, to_render=True)
    model = SelfPlayModel(f"./checkers/saved_models/{model_name}")
    for episode in range(episodes):
        reward = 0
        done = False
        current_state, actions_index = env.reset()
        state_player = env.refactor_state(current_state, env.player, env.move_counter)
        env.render()
        while not done:
            if env.player == ai_player:
                mcts = MonteCarloTreeSearch(env, model, num_simulations, heuristic_weight)
                root = mcts.run(current_state, state_player, env.player)
                action = root.select_action(temperature=0)
            else:  # HUMAN OR RANDOM PLAYER
                if not human:
                    action = random.sample(actions_index, 1)[0]
                else:
                    action = None
                    highlighted_piece = None
                    moved = False
                    highlighting = False
                    while not moved:
                        events = pg.event.get()
                        for event in events:
                            if event.type == pg.MOUSEBUTTONUP:
                                mouse_position = pg.mouse.get_pos()
                                highlighted_piece = env.highlight_piece(mouse_position, highlighted_piece, actions_index)
                                if highlighted_piece:
                                    highlighting = True
                                else:
                                    highlighting = False
                                env.render()
                        pg.event.clear()
                        while highlighting:
                            if highlighted_piece:
                                highlight_events = pg.event.get()
                                for highlight_event in highlight_events:
                                    if highlight_event.type == pg.MOUSEBUTTONUP:
                                        mouse_position = pg.mouse.get_pos()
                                        coord_x = mouse_position[0] // env.board.cell_size[0]
                                        coord_y = mouse_position[1] // env.board.cell_size[1]
                                        coords = (coord_y, coord_x)
                                        if len(actions_index) > 0:
                                            for move in actions_index:
                                                if env.moves_list[move][-1] == coords and env.moves_list[move][0] == \
                                                        highlighted_piece["board_piece_coordinates"]:
                                                    action = move
                                                    moved = True
                                                    highlighting = False
                                                    break
                                    elif highlight_event.type == KEYDOWN:
                                        if highlight_event.key == K_ESCAPE:
                                            highlighting = False
                                            highlighted_piece = env.end_highlight(highlighted_piece)
                                            env.render()
                                pg.event.clear()
                    highlighted_piece = env.end_highlight(highlighted_piece)
            new_state, reward, done, actions_index = env.step(action)
            current_state = new_state
            state_player = env.refactor_state(new_state, env.player, env.move_counter)
            env.render()
        time.sleep(3)
        print(f"LAST PLAYER: {env.player}, REWARD: {reward}")
        print("==================================================")

"""
HUMAN = True
EPISODES = 5
NUM_SIMULATIONS = 100
AI_PLAYER = 1

test_checkers(HUMAN, EPISODES, NUM_SIMULATIONS, AI_PLAYER, 0.5)
"""