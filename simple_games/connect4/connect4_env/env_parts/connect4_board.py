import warnings

import pygame as pg
import numpy as np

from simple_games.connect4.connect4_env.env_parts.connect4_drawer import Drawer


class Board:
    """
    Class responsible for the connect4 board.
    The board state is kept as a numpy array, with 1 representing player 1 (red)
    and -1 (gray) representing player -1.

    When a player makes a move, the state of this board is updated.

    It handles rendering of the board, using the Drawer class,
    and also the rendering of the pieces
    """

    def __init__(self, board_parameters, draw_parameters, moves, to_render=False):
        """
        Constructor for the connect4 Board class.
        All parameters, from the provided parameter dictionaries
        are stored within the class as attributes

        Moreover, the necessary tools for the board to function are also set up (if necessary).
        - The 6x7 NumPy board
        - Drawer class for drawing on the PyGame display
        - The PyGame Surface used for drawing the board
        - Dictionary, which places the NumPy array indexes, representing each cell in the game grid,
          to the corresponding pixel coordinate in PyGame

        :param board_parameters: parameters for the Board
        :param draw_parameters: parameters for the Drawer
        :param moves: list of all possible moves in connect4
        :param to_render: render the board flag
        """

        # Store board parameters in the class
        self.width = board_parameters["BOARD_WIDTH"]
        self.height = board_parameters["BOARD_HEIGHT"]
        self.border_width = board_parameters["BORDER_WIDTH"]

        # Store the move list
        self.moves = moves

        # Initialize the NumPy board
        self.np_board = np.zeros((6, 7))

        # Initialize the Drawer class
        self.drawer = Drawer(draw_parameters)

        # Store the rendering flag
        self.to_render = to_render

        if self.to_render:
            # If the board should be rendered, then calculate size of each cell
            # and the grid PyGame pixel coordinates
            self.cell_size = ((self.width - self.border_width) // 7,
                              (self.height - self.border_width) // 6)
            self.centre_offset = (self.cell_size[0] * 0.5, self.cell_size[1] * 0.5)
            self.centre_of_cells = {}

            # Initialize the PyGame Surface, on which the board will be drawn
            self.pg_board = pg.Surface((self.width - self.border_width, self.height - self.border_width))
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

    def initialize_board(self):
        """
        Initialize NumPy board by setting all grid cells to 0
        """

        self.np_board.fill(0)

    def initialize_display(self):
        """
        Initialize the PyGame Surface, by drawing the connect4 grid and
        calculating the center of grid cells
        """

        self.drawer.fill_display(self.pg_board, self.drawer.DARK_BLUE_BACKGROUND)

        for x in range(7):
            for y in range(6):
                # Draw white circles acting as empty spaces
                x_pixel = x * self.cell_size[0] + self.centre_offset[0]
                y_pixel = y * self.cell_size[1] + self.centre_offset[1]

                self.drawer.draw_circle(self.pg_board, x_pixel, y_pixel, self.drawer.piece_radius, self.drawer.WHITE)

                if self.to_render:
                    # Get the centre of a given cell
                    self.centre_of_cells[(y, x)] = (self.cell_size[1] // 2 + y * self.cell_size[1],
                                                    self.cell_size[0] // 2 + x * self.cell_size[0])

    @staticmethod
    def fill_display(display, color):
        """
        Fill the provided PyGame display with a color, used for e.g. setting the background

        :param display: PyGame display which should be filled with a certain color
        :param color: color which should fill the display
        """

        display.fill(color)

    def mark_moves(self, action_index):
        """
        Mark all positions on the board, where a player can move (i.e. place his symbol),
        by iterating through the given list with move indexes

        :param action_index: list of indexes of moves from the move list
        """

        for move in action_index:
            # Set x and y PyGame coordinates from the move grid values
            # pixel x is dependent on the column of NumPy array (index 1)
            # pixel y on the row (index 0)
            x_pixel = self.centre_of_cells[self.moves[move]][1]
            y_pixel = self.centre_of_cells[self.moves[move]][0]

            self.drawer.add_dot(self.pg_board, x_pixel, y_pixel)

    def remove_mark(self, action_index):
        """
        Remove all marks of previous moves from the board,
        by iterating through the give list with move indexes

        :param action_index: list of indexes of moves from the move list
        """

        for move in action_index:
            # Set x and y PyGame coordinates from the move grid values
            # pixel x is dependent on the column of NumPy array (index 1)
            # pixel y on the row (index 0)
            x_pixel = self.centre_of_cells[self.moves[move]][1]
            y_pixel = self.centre_of_cells[self.moves[move]][0]

            self.drawer.remove_dot(self.pg_board, x_pixel, y_pixel)

    def render(self, player):
        """
        If the to_render flag is set to True, then render the connect4 board,
        by placing the board (PyGame Surface) on a PyGame display

        :param player: player who is now to play
        """

        if self.to_render:
            pg.init()

            # Create display (PyGame window)
            game_display = pg.display.set_mode((self.width, self.height))
            pg.display.set_caption("Connect4Board")

            # Place the board on the newly created display
            game_display.blit(self.pg_board, self.pg_board.get_rect(center=(self.width // 2, self.height // 2)))

            # Draw both the standard border and the player turn border
            self.drawer.draw_border(game_display, self.width, self.height, self.border_width)
            self.drawer.draw_player_border(game_display, self.width, self.height, self.border_width, player)

            # Update PyGame display after these updates
            pg.display.update()
        else:
            warnings.warn("Can't render without display. Set 'to_render' = True")

    def make_move(self, action, player, state, to_render=False):
        """
        Update the state of a board (Numpy array) based on the performed move by a player,
        if the board is to be rendered, then also draw the appropriate symbol on the board (PyGame Surface)

        :param state: the current state of board (Numpy array), necessary for Monte-Carlo-Tree Search
        :param action: the index of the performed move
        :param player: player who performed the move
        :param to_render: an additional to_render flag, necessary for Monte-Carlo-Tree Search
        :return: state of the board (NumPy array) after the performed move
        """

        x = self.moves[action][0]
        y = self.moves[action][1]
        if to_render:
            x_pixel = self.centre_of_cells[self.moves[action]][1]
            y_pixel = self.centre_of_cells[self.moves[action]][0]
            self.drawer.draw_piece(self.pg_board, x_pixel, y_pixel, player)
        new_state = np.copy(state)
        new_state[x][y] = player
        return new_state
