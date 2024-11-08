import pickle
import pygame as pg
import time
import random
from SelfPlayModel import SelfPlayModel
from MonteCarloTreeSearch import MonteCarloTreeSearch
from checkers_env import CheckersEnv
from CheckersEnvParameters import board_parameters, draw_parameters
from pygame.locals import KEYDOWN, K_ESCAPE

HUMAN = True
EPISODES = 5
NUM_SIMULATIONS = 1000
AI_PLAYER = 1
HEURISTIC_WEIGHT = 1
env = CheckersEnv(board_parameters, draw_parameters, to_render=True)
model = SelfPlayModel("saved_models/actor-critic-CHECKERS-month-6-day-9-ep-154647-50%.h5")
with open("saved_models/SavedLogs/log-CHECKERS-month-6-day-9-ep-154647-50.0%.pkl", "rb") as logger_file:
    logger = pickle.load(logger_file)
logger.graph_log()
for episode in range(EPISODES):
    reward = 0
    done = False
    current_state, actions_index = env.reset()
    state_player = env.refactor_state(current_state, env.player, env.move_counter)
    env.render()
    while not done:
        if env.player == AI_PLAYER:  # AI PLAYER
            mcts = MonteCarloTreeSearch(env, model, NUM_SIMULATIONS, HEURISTIC_WEIGHT)
            root = mcts.run(current_state, state_player, env.player)
            action = root.select_action(temperature=0)
        else:  # HUMAN OR RANDOM PLAYER
            if not HUMAN:
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
