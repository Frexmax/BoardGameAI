import pygame as pg
import numpy as np
import warnings
from checkers_drawer import Drawer


class Board:
    def __init__(self, board_parameters, draw_parameters, to_render=False):
        # BOARD INFO
        self.width = board_parameters["BOARD_WIDTH"]
        self.height = board_parameters["BOARD_HEIGHT"]
        self.border_width = board_parameters["BORDER_WIDTH"]
        self.border_colour = board_parameters["BORDER_COLOUR"]
        self.red_player_border = board_parameters["RED_PLAYER_BORDER"]
        self.black_player_border = board_parameters["BLACK_PLAYER_BORDER"]

        # DRAWER
        self.drawer = Drawer(draw_parameters)

        # CALCULATE CELL SIZE
        self.cell_size = ((self.width - self.border_width) // 8,
                          (self.height - self.border_width) // 8)
        self.np_board = np.zeros((8, 8))
        self.centre_of_cells = {}

        # RENDER AND GAME DISPLAY
        self.to_render = to_render
        if self.to_render:
            self.pg_board = pg.Surface((self.width - self.border_width, self.height - self.border_width))
            self.cell_size = ((self.width - self.border_width) // 8,
                              (self.height - self.border_width) // 8)  # CALCULATE CELL SIZE
        else:
            self.pg_board = None

    def reset(self):
        if self.to_render:
            self.initialize_display()
        self.initialize_board()

    def render_player_border(self, display, current_player):
        if current_player == 1:
            pg.draw.rect(display, self.red_player_border, (0, 0, self.width, self.height), self.border_width // 4)
        else:
            pg.draw.rect(display, self.black_player_border, (0, 0, self.width, self.height), self.border_width // 4)

    def render(self, current_player):
        if self.to_render:
            pg.init()
            game_display = pg.display.set_mode((self.width, self.height))
            pg.display.set_caption("CheckersBoard")
            pg.event.pump()
            game_display.blit(self.pg_board, self.pg_board.get_rect(center=(self.width // 2, self.height // 2)))
            pg.draw.rect(game_display, self.border_colour, (0, 0, self.width, self.height), self.border_width // 2)
            self.render_player_border(game_display, current_player)
            pg.display.update()
        else:
            warnings.warn("Can't render without display. Set 'to_render' = True")

    def end_highlight(self, highlighted_before, player):
        if highlighted_before:
            start_coords_piece = highlighted_before["piece_coordinates"]
            start_coords_rectangle = highlighted_before["rectangle_coordinates"]
            start_coords_dot = highlighted_before["dot_coordinates"]
            start_piece_type = highlighted_before["piece_type"]
            self.drawer.remove_highlight(self.pg_board, start_coords_rectangle[0], start_coords_rectangle[1],
                                         start_coords_piece[0], start_coords_piece[1], start_piece_type,
                                         player, self.cell_size)
            for dot_coord in start_coords_dot:
                self.drawer.remove_dot(self.pg_board, dot_coord[0] * self.cell_size[0],
                                       dot_coord[1] * self.cell_size[1], self.cell_size)

    def highlight_piece(self, click_position, player, highlighted_before, possible_moves, moves_list):
        x = click_position[1] // self.cell_size[1]
        y = click_position[0] // self.cell_size[0]
        if x >= 8 or y >= 8:
            return False
        coords_piece = self.centre_of_cells[(x, y)]
        coords_rectangle = (y * self.cell_size[0], x * self.cell_size[1])
        piece_type = abs(self.np_board[x][y])
        self.end_highlight(highlighted_before, player)
        if piece_type != 0:
            self.drawer.add_highlight(self.pg_board, coords_rectangle[0], coords_rectangle[1], coords_piece[0],
                                      coords_piece[1], piece_type, player, self.cell_size)
            dot_coords = []
            for action in possible_moves:
                if moves_list[action][0] == (x, y):
                    end_pos = moves_list[action][-1]
                    self.drawer.add_dot(self.pg_board, self.centre_of_cells[end_pos][0],
                                        self.centre_of_cells[end_pos][1])
                    dot_coords.append(end_pos[::-1])
            return {"board_piece_coordinates": (x, y), "piece_coordinates": coords_piece,
                    "rectangle_coordinates": coords_rectangle, "dot_coordinates": dot_coords, "piece_type": piece_type}
        return False

    def initialize_display(self):
        self.pg_board.fill((255, 255, 255))
        for x in range(0, 8):
            for y in range(0, 8):
                if x % 2 == 0:
                    # DRAW BLACK SQUARES
                    y_pixel = y * self.cell_size[1]
                    if (y + 1) % 2 != 0:
                        x_pixel = (x + 1) * self.cell_size[0]
                        pg.draw.rect(self.pg_board, (0, 0, 0), (x_pixel, y_pixel, self.cell_size[0], self.cell_size[1]))
                    else:
                        x_pixel = x * self.cell_size[0]
                        pg.draw.rect(self.pg_board, (0, 0, 0), (x_pixel, y_pixel, self.cell_size[0], self.cell_size[1]))
                # GET CENTRE OF CELLS - FOR PYGAME DISPLAY
                self.centre_of_cells[(x, y)] = (self.cell_size[1] // 2 + y * self.cell_size[1],
                                                self.cell_size[0] // 2 + x * self.cell_size[0])

    def initialize_board(self):
        self.np_board.fill(0)
        # DRAW BLACK
        for x in range(0, 3):
            if x % 2 != 0:
                for y in range(0, 8, 2):
                    if self.to_render:
                        centre = self.centre_of_cells[(x, y)]
                        pixel_x = centre[0]
                        pixel_y = centre[1]
                        self.drawer.draw_piece(self.pg_board, pixel_x, pixel_y, -1)
                    self.np_board[x][y] = -1
            else:
                for y in range(1, 8, 2):
                    if self.to_render:
                        centre = self.centre_of_cells[(x, y)]
                        pixel_x = centre[0]
                        pixel_y = centre[1]
                        self.drawer.draw_piece(self.pg_board, pixel_x, pixel_y, -1)
                    self.np_board[x][y] = -1

        # DRAW RED
        for x in range(5, 8):
            if x % 2 != 0:
                for y in range(0, 8, 2):
                    if self.to_render:
                        centre = self.centre_of_cells[(x, y)]
                        pixel_x = centre[0]
                        pixel_y = centre[1]
                        self.drawer.draw_piece(self.pg_board, pixel_x, pixel_y, 1)
                    self.np_board[x][y] = 1
            else:
                for y in range(1, 8, 2):
                    if self.to_render:
                        centre = self.centre_of_cells[(x, y)]
                        pixel_x = centre[0]
                        pixel_y = centre[1]
                        self.drawer.draw_piece(self.pg_board, pixel_x, pixel_y, 1)
                    self.np_board[x][y] = 1

    def update_board(self, state):
        self.np_board = np.copy(state)

    def make_move(self, state, action, player, moves_list, captures_list, to_render=None):
        new_king_position = None
        starting_position = moves_list[action][0]
        new_position = moves_list[action][-1]
        piece_type = int(abs(state[starting_position]))
        start_pixel_x = None
        start_pixel_y = None
        new_pixel_x = None
        new_pixel_y = None

        if len(captures_list[action]) != 0 and type(captures_list[action][0]) == int:  # FIX LOOPS LATER ON
            captures = [captures_list[action]]
        else:
            captures = captures_list[action]

        if to_render:
            start_pixel_x = self.centre_of_cells[starting_position][0]
            start_pixel_y = self.centre_of_cells[starting_position][1]
            new_pixel_x = self.centre_of_cells[new_position][0]
            new_pixel_y = self.centre_of_cells[new_position][1]

        if len(captures) == 0:  # NO CAPTURE MOVES
            if to_render:
                self.drawer.remove_piece(self.pg_board, start_pixel_x, start_pixel_y)
            if piece_type == 1:
                state[new_position] = player  # UPDATE BOARD WITH NEW PIECE POSITION
                if to_render:
                    self.drawer.draw_piece(self.pg_board, new_pixel_x, new_pixel_y, player)  # DRAW PIECE
            else:
                state[new_position] = 2 * player  # UPDATE BOARD WITH NEW KING POSITION
                if to_render:
                    self.drawer.draw_king(self.pg_board, new_pixel_x, new_pixel_y, player)  # DRAW KING
            state[starting_position] = 0  # REMOVE OLD POSITION

        else:  # CAPTURE MOVES
            for capture in captures:
                if to_render:  # REMOVE STARTING AND CAPTURED PIECE
                    capture_pixel_x = self.centre_of_cells[capture][0]
                    capture_pixel_y = self.centre_of_cells[capture][1]
                    self.drawer.remove_piece(self.pg_board, start_pixel_x, start_pixel_y)
                    self.drawer.remove_piece(self.pg_board, capture_pixel_x, capture_pixel_y)
                if piece_type == 1:
                    state[new_position] = player  # UPDATE BOARD WITH NEW PIECE POSITION
                    if to_render:
                        self.drawer.draw_piece(self.pg_board, new_pixel_x, new_pixel_y, player)  # DRAW PIECE
                else:
                    state[new_position] = 2 * player  # UPDATE BOARD WITH NEW KING POSITION
                    if to_render:
                        self.drawer.draw_king(self.pg_board, new_pixel_x, new_pixel_y, player)  # DRAW KING
                state[capture] = 0  # REMOVE CAPTURED
                state[starting_position] = 0  # REMOVE OLD POSITION

        if len(np.where(state[0] == 1)[0]) > 0:  # CHECK IF NEW KINGS FOR RED
            y = np.where(state[0] == 1)[0][0]
            state[0][np.where(state[0] == 1)[0][0]] = 2
            new_king_position = (0, y)
            if to_render:
                self.drawer.draw_king(self.pg_board, new_pixel_x, new_pixel_y, player)
        if len(np.where(state[7] == -1)[0]) > 0:  # CHECK IF NEW KINGS FOR BLACK
            y = np.where(state[7] == -1)[0][0]
            state[7][np.where(state[7] == -1)[0][0]] = -2
            new_king_position = (7, y)
            if to_render:
                self.drawer.draw_king(self.pg_board, new_pixel_x, new_pixel_y, player)
        return np.copy(state), new_king_position
