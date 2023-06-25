import pygame as pg
import numpy as np
import warnings
from Connect4Drawer import Drawer


class Board:
    def __init__(self, board_parameters, draw_parameters, moves, to_render=False):
        # BOARD INFO
        self.width = board_parameters["BOARD_WIDTH"]
        self.height = board_parameters["BOARD_HEIGHT"]
        self.border_width = board_parameters["BORDER_WIDTH"]
        self.border_colour = board_parameters["BORDER_COLOUR"]
        self.red_player_border = board_parameters["RED_PLAYER_BORDER"]
        self.black_player_border = board_parameters["BLACK_PLAYER_BORDER"]

        # MOVES
        self.moves = moves

        # CALCULATE CELL SIZE
        self.cell_size = ((self.width - self.border_width) // 7,
                          (self.height - self.border_width) // 6)
        self.centre_offset = (self.cell_size[0] * 0.5, self.cell_size[1] * 0.5)
        self.np_board = np.zeros((6, 7))
        self.centre_of_cells = {}

        # DRAWER
        self.drawer = Drawer(draw_parameters)

        # RENDER AND GAME DISPLAY
        self.to_render = to_render
        if self.to_render:
            self.pg_board = pg.Surface((self.width - self.border_width, self.height - self.border_width))
        else:
            self.pg_board = None

    def reset(self):
        if self.to_render:
            self.initialize_display()
        self.initialize_board()

    def initialize_board(self):
        self.np_board.fill(0)

    def initialize_display(self):
        self.pg_board.fill((10, 7, 148))
        for x in range(7):
            for y in range(6):
                # DRAW EMPTY SPACES
                x_pixel = x * self.cell_size[0] + self.centre_offset[0]
                y_pixel = y * self.cell_size[1] + self.centre_offset[1]
                pg.draw.circle(self.pg_board, (255, 255, 255), (x_pixel, y_pixel), self.drawer.piece_radius)
                # GET CENTRE OF CELLS - FOR PYGAME DISPLAY
                self.centre_of_cells[(y, x)] = (self.cell_size[1] // 2 + y * self.cell_size[1],
                                                self.cell_size[0] // 2 + x * self.cell_size[0])

    def render_player_border(self, display, current_player):
        if current_player == 1:
            pg.draw.rect(display, self.red_player_border, (0, 0, self.width, self.height), self.border_width // 4)
        else:
            pg.draw.rect(display, self.black_player_border, (0, 0, self.width, self.height), self.border_width // 4)

    def mark_moves(self, action_index):
        for move in action_index:
            x_pixel = self.centre_of_cells[self.moves[move]][1]
            y_pixel = self.centre_of_cells[self.moves[move]][0]
            self.drawer.add_dot(self.pg_board, x_pixel, y_pixel)

    def remove_mark(self, action_index, action):
        for move in action_index:
            if move != action:
                x_pixel = self.centre_of_cells[self.moves[move]][1]
                y_pixel = self.centre_of_cells[self.moves[move]][0]
                self.drawer.remove_dot(self.pg_board, x_pixel, y_pixel)

    def render(self, current_player):
        if self.to_render:
            pg.init()
            game_display = pg.display.set_mode((self.width, self.height))
            pg.display.set_caption("Connect4Board")
            pg.event.pump()
            game_display.blit(self.pg_board, self.pg_board.get_rect(center=(self.width // 2, self.height // 2)))
            pg.draw.rect(game_display, self.border_colour, (0, 0, self.width, self.height), self.border_width // 2)
            self.render_player_border(game_display, current_player)
            pg.display.update()
        else:
            warnings.warn("Can't render without display. Set 'to_render' = True")

    def make_move(self, action, player, state, to_render=False):
        x = self.moves[action][0]
        y = self.moves[action][1]
        if to_render:
            x_pixel = self.centre_of_cells[self.moves[action]][1]
            y_pixel = self.centre_of_cells[self.moves[action]][0]
            self.drawer.draw_piece(self.pg_board, x_pixel, y_pixel, player)
        new_state = np.copy(state)
        new_state[x][y] = player
        return new_state
