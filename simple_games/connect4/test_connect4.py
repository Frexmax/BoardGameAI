import time
import random

import pygame as pg

from connect4_env.connect4_env import Connect4Env
from connect4_env.env_parameters.connect4_env_parameters import board_parameters, draw_parameters
from trainer.self_play_model.self_play_model import SelfPlayModel
from trainer.monte_carlo_tree_search.monte_carlo_tree_search import MonteCarloTreeSearch

HUMAN = True
EPISODES = 5
NUM_SIMULATIONS = 10
AI_PLAYER = 1


def test_connect4(human, episodes, num_simulations, ai_player):
    env = Connect4Env(board_parameters, draw_parameters, to_render=True)
    model = SelfPlayModel("saved_models/actor-critic--month-5-day-23-7-100%.h5")
    for episode in range(EPISODES):
        done = False
        reward = 0
        current_state, actions_index = env.reset()
        state_player = env.refactor_state(current_state, env.player, env.move_counter)
        env.render()
        mcts = MonteCarloTreeSearch(env, model, NUM_SIMULATIONS, heuristic_weight=0)
        root = None
        while not done:
            if env.player == AI_PLAYER:
                root = mcts.run(current_state, state_player, env.player, root)
                action = root.select_action(temperature=0)
            else:
                if not HUMAN:
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
                    env.remove_mark(actions_index, action)
            if root is not None:
                root = root.children[action]
            new_state, reward, done, actions_index = env.step(action)
            current_state = new_state
            state_player = env.refactor_state(new_state, env.player, env.move_counter)
            env.render()
        time.sleep(3)
        print(f"LAST PLAYER: {env.player}, REWARD: {reward}")
        print("====================================================")
