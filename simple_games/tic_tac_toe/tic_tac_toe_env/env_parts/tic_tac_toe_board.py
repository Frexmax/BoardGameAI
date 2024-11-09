import warnings

import numpy as np
import pygame as pg

from simple_games.tic_tac_toe.tic_tac_toe_env.env_parts.tic_tac_toe_drawer import Drawer


class Board:
    """
    Class responsible for the tic-tac-toe board.
    The board state is kept as a numpy array, with 1 representing player 1 (crosses)
    and -1 representing player -1 (circles).

    When a player makes a move, the state of this board is updated.

    It handles rendering of the board, and using the Drawer class,
    it also the rendering of crosses and circles.
    """

    def __init__(self, board_parameters, draw_parameters, moves, to_render=False):
        """
        Constructor for the tic-tac-toe Board class.
        All parameters, from the provided parameter dictionaries
        are stored within the class as attributes

        Moreover, the necessary tools for the board to function are also set up (if necessary).
        - The 3x3 NumPy board
        - Drawer class for drawing on the PyGame display
        - The PyGame Surface used for drawing the board
        - Dictionary, which places the NumPy array indexes, representing each cell in the game grid,
          to the corresponding pixel coordinate in PyGame

        :param board_parameters: parameters for the Board
        :param draw_parameters: parameters for the Drawer
        :param moves: list of all possible moves in tic-tac-toe
        :param to_render: render the board flag
        """

        # Store board parameters in the class
        self.width = board_parameters["BOARD_WIDTH"]
        self.height = board_parameters["BOARD_HEIGHT"]
        self.border_width = board_parameters["BORDER_WIDTH"]
        self.line_width = board_parameters["LINE_WIDTH"]

        # Store the move list
        self.moves = moves

        # Initialize the NumPy board
        self.np_board = np.zeros((3, 3))

        # Initialize the Drawer class
        self.drawer = Drawer(draw_parameters)

        # Store the rendering flag
        self.to_render = to_render
        if self.to_render:
            # If the board should be rendered, then calculate size of each cell and
            # the grid PyGame pixel coordinates
            self.cell_size = ((self.width - self.border_width) // 3,
                              (self.height - self.border_width) // 3)
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

    def initialize_display(self):
        """
        Initialize the PyGame Surface, by drawing the tic-tac-toe grid and
        calculating the center of grid cells
        """

        self.drawer.fill_display(self.pg_board)
        for x in range(3):
            for y in range(3):

                if x != 0:
                    # Draw the horizontal lines of a grid
                    start_coordinate = (0, y * self.cell_size[1])
                    end_coordinate = (self.width, y * self.cell_size[1])
                    self.drawer.draw_line(self.pg_board, start_coordinate, end_coordinate, self.line_width, True)

                if y != 0:
                    # Draw the vertical lines of a grid
                    start_coordinate = (x * self.cell_size[0], 0)
                    end_coordinate = (x * self.cell_size[0], self.height)
                    self.drawer.draw_line(self.pg_board, start_coordinate, end_coordinate, self.line_width, True)

                # Get the centre of a given cell
                self.centre_of_cells[(y, x)] = (self.cell_size[1] // 2 + y * self.cell_size[1],
                                                self.cell_size[0] // 2 + x * self.cell_size[0])

    def initialize_board(self):
        """
        Initialize NumPy board by setting all grid cells to 0
        """
        self.np_board.fill(0)

    def mark_moves(self, action_index):
        """
        Mark all positions on the board, where a player can move (i.e. place his symbol),
        by iterating through the given list with move indexes

        :param action_index: list of indexes of moves from the board's move list
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

        :param action_index:
        """
        for move in action_index:
            # Set x and y PyGame coordinates from the move grid values
            # pixel x is dependent on the column of NumPy array (index 1)
            # pixel y on the row (index 0)
            x_pixel = self.centre_of_cells[self.moves[move]][1]
            y_pixel = self.centre_of_cells[self.moves[move]][0]
            self.drawer.remove_dot(self.pg_board, x_pixel, y_pixel)

    def render(self, current_player):
        """
        If the to_render flag is set to True, then render the tic-tac-toe board,
        by placing the board (PyGame Surface) on a PyGame display

        :param current_player: player who is now to play
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
            self.drawer.draw_player_border(game_display, self.width, self.height, self.border_width, current_player)

            # Update PyGame display after these updates
            pg.display.update()
        else:
            warnings.warn("Can't render without display. Set 'to_render' = True")

    def make_move(self, state, action, player, to_render=False):
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
            if player == 1:
                self.drawer.draw_cross(self.pg_board, x_pixel, y_pixel)
            else:
                self.drawer.draw_circle(self.pg_board, x_pixel, y_pixel)
        new_state = np.copy(state)
        new_state[x][y] = player
        return new_state
