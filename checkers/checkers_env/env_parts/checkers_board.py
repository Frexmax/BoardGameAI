import warnings

import numpy as np
import pygame as pg

from checkers.checkers_env.env_parts.checkers_drawer import Drawer


class Board:
    """
    Class responsible for the checkers board.
    The board state is kept as a numpy array, with 1 representing player 1 (red)
    and -1 (gray) representing player -1.

    When a player makes a move, the state of this board is updated.

    It handles rendering of the board, using the Drawer class, rendering of the pieces,
    and also the highlighting of selected pieces
    """

    def __init__(self, board_parameters, draw_parameters, to_render=False):
        """
        Constructor for the checkers Board class.
        All parameters, from the provided parameter dictionaries
        are stored within the class as attributes

        Moreover, the necessary tools for the board to function are also set up (if necessary).
        - The 8x8 NumPy board
        - Drawer class for drawing on the PyGame display
        - The PyGame Surface used for drawing the board
        - Dictionary, which places the NumPy array indexes, representing each cell in the game grid,
          to the corresponding pixel coordinate in PyGame

        :param board_parameters: parameters for the Board
        :param draw_parameters: parameters for the Drawer
        :param to_render: render the board flag
        """

        # Store board parameters in the class
        self.width = board_parameters["BOARD_WIDTH"]
        self.height = board_parameters["BOARD_HEIGHT"]
        self.border_width = board_parameters["BORDER_WIDTH"]

        # Initialize the Drawer class
        self.drawer = Drawer(draw_parameters)

        # Initialize the NumPy board
        self.np_board = np.zeros((8, 8))

        # Store the rendering flag
        self.to_render = to_render

        if self.to_render:
            # If the board should be rendered, then calculate size of each cell
            # and the grid PyGame pixel coordinates
            self.pg_board = pg.Surface((self.width - self.border_width, self.height - self.border_width))
            self.cell_size = ((self.width - self.border_width) // 8,
                              (self.height - self.border_width) // 8)
            self.centre_of_cells = {}
        else:
            # If the board will not be rendered, then there is no need to keep any data necessary for rendering,
            # therefore the attributes can be simply assigned to None
            self.cell_size = None
            self.centre_offset = None
            self.centre_of_cells = None
            self.pg_board = None

    def reset(self):
        """
        Reset the board state, by re-initializing the Numpy array,
        and the PyGame Surface (if the board is to be rendered)
        """

        if self.to_render:
            self.initialize_display()
        self.initialize_board()

    def render(self, player):
        """
        If the to_render flag is set to True, then render the checkers board,
        by placing the board (PyGame Surface) on a PyGame display

        :param player: player who is now to play
        """

        if self.to_render:
            pg.init()

            # Create display (PyGame window)
            game_display = pg.display.set_mode((self.width, self.height))
            pg.display.set_caption("CheckersBoard")

            # Place the board on the newly created display
            game_display.blit(self.pg_board, self.pg_board.get_rect(center=(self.width // 2, self.height // 2)))

            # Draw both the standard border and the player turn border
            self.drawer.draw_border(game_display, self.width, self.height, self.border_width)
            self.drawer.draw_player_border(game_display, self.width, self.height, self.border_width, player)
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

                if self.to_render:
                    # Get the centre of a given cell
                    self.centre_of_cells[(y, x)] = (self.cell_size[1] // 2 + y * self.cell_size[1],
                                                    self.cell_size[0] // 2 + x * self.cell_size[0])

    def initialize_board(self):
        """
        Initialize NumPy board by setting all grid cells to 0
        """

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
        """
        Set the current board state to the one provided

        :param state: new board state
        """

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
