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
        """
        Remove rendering for a highlighted piece and remove dots indicating possible movements for the
        highlighted piece

        :param highlighted_before: dict with information about the last highlighted piece, null if none highlighted
        :param player: player whose about to play
        """

        if highlighted_before:
            # Get highlight information
            start_coords_piece = highlighted_before["piece_coordinates"]
            start_coords_rectangle = highlighted_before["rectangle_coordinates"]
            start_coords_dot = highlighted_before["dot_coordinates"]
            start_piece_type = highlighted_before["piece_type"]

            # Remove old highlight
            self.drawer.remove_highlight(self.pg_board, start_coords_rectangle[0], start_coords_rectangle[1],
                                         start_coords_piece[0], start_coords_piece[1], start_piece_type,
                                         player, self.cell_size)

            # Remove all dots indicating possible moves
            for dot_coord in start_coords_dot:
                self.drawer.remove_dot(self.pg_board, dot_coord[0] * self.cell_size[0],
                                       dot_coord[1] * self.cell_size[1], self.cell_size)

    def highlight_piece(self, click_position, player, highlighted_before, possible_moves, moves_list):
        """
        Highlight piece and draw the possible movements for that piece using dots

        :param click_position: the x, y position of the mouse click
        :param player: player who is about to play
        :param highlighted_before: dict with information about the last highlighted piece, null if none highlighted
        :param possible_moves: list indexes for the possible moves the current player can make
        :param moves_list: list of all possible moves with coordinates
        :return: highlight information dict, if the highlight was successful. Otherwise False
        """

        x = click_position[1] // self.cell_size[1]
        y = click_position[0] // self.cell_size[0]

        if x >= 8 or y >= 8:
            # If the mouse click is outside the board grid, then return False
            return False

        coords_piece = self.centre_of_cells[(x, y)]
        coords_rectangle = (y * self.cell_size[0], x * self.cell_size[1])
        piece_type = abs(self.np_board[x][y])

        # Remove the last highlight
        self.end_highlight(highlighted_before, player)

        if piece_type == 0:
            # If the mouse click is on an empty grid cell, then return False
            return False

        piece_owner = self.np_board[x][y] / abs(self.np_board[x][y])
        if piece_owner != player:
            # If the mouse click is on an enemy piece, then return False
            return False

        # Highlight the selected grid cell
        self.drawer.add_highlight(self.pg_board, coords_rectangle[0], coords_rectangle[1], coords_piece[0],
                                  coords_piece[1], piece_type, player, self.cell_size)

        # Get the possible moves for the highlighted piece and draw dots indicating these moves on the display
        dot_coords = []
        for action in possible_moves:
            if moves_list[action][0] == (x, y):
                end_pos = moves_list[action][-1]
                self.drawer.add_dot(self.pg_board, self.centre_of_cells[end_pos][0],
                                    self.centre_of_cells[end_pos][1])
                dot_coords.append(end_pos[::-1])

        return {"board_piece_coordinates": (x, y), "piece_coordinates": coords_piece,
                "rectangle_coordinates": coords_rectangle, "dot_coordinates": dot_coords, "piece_type": piece_type}

    def initialize_display(self):
        """
        Initialize the PyGame Surface, by drawing the checkers grid and
        calculating the center of grid cells
        """

        self.drawer.fill_display(self.pg_board, self.drawer.WHITE)

        for x in range(8):
            for y in range(8):
                if x % 2 == 0:
                    # Calculate the PyGame pixel coordinates of the grid cell's top-left corner
                    y_pixel = y * self.cell_size[1]
                    if (y + 1) % 2 != 0:
                        # Calculate x coordinates for rows starting with white cell
                        x_pixel = (x + 1) * self.cell_size[0]
                    else:
                        # Calculate x coordinates for rows starting with black cell
                        x_pixel = x * self.cell_size[0]

                    # Draw the black grid cells, on a white background
                    self.drawer.draw_rectangle(self.pg_board, x_pixel, y_pixel,
                                               self.cell_size[0], self.cell_size[1], self.drawer.BLACK)

                if self.to_render:
                    # Get the centre of a given cell
                    self.centre_of_cells[(x, y)] = (self.cell_size[1] // 2 + y * self.cell_size[1],
                                                    self.cell_size[0] // 2 + x * self.cell_size[0])

    def initialize_board(self):
        """
        First reset the numpy board, by setting all cells to 0,
        and then place the starting pieces for both players.
        If the board is set to render, then also draw these pieces on the PyGame display
        """

        def set_up_piece(piece_x, piece_y, player):
            """
            Place a piece for the provided player at the specified coordinates on the numpy board.
            If the board is set to render, then obtain the corresponding PyGame pixel coordinates,
            and draw the piece on the PyGame board

            :param piece_x: x coordinate of the piece on the numpy board
            :param piece_y: y coordinate of the piece on the numpy board
            :param player: player to whom the piece belongs
            """

            if self.to_render:
                centre = self.centre_of_cells[(piece_x, piece_y)]
                pixel_x = centre[0]
                pixel_y = centre[1]
                self.drawer.draw_piece(self.pg_board, pixel_x, pixel_y, player)
            self.np_board[piece_x][piece_y] = -1

        self.np_board.fill(0)

        # Draw black pieces (player -1 pieces)
        for x in range(0, 3):
            if x % 2 != 0:
                for y in range(0, 8, 2):
                    set_up_piece(x, y, player=-1)
            else:
                for y in range(1, 8, 2):
                    if self.to_render:
                        set_up_piece(x, y, player=-1)

        # Draw red pieces (player 1 pieces)
        for x in range(5, 8):
            if x % 2 != 0:
                for y in range(0, 8, 2):
                    set_up_piece(x, y, player=1)
            else:
                for y in range(1, 8, 2):
                    set_up_piece(x, y, player=1)

    def update_board(self, state):
        """
        Set the current board state to the one provided

        :param state: new board state
        """

        self.np_board = np.copy(state)

    def make_move(self, state, action, player, moves_list, captures_list, to_render=None):
        """
        Perform the chosen action for the provided state and player.
        Update the numpy board and if necessary also re-draw the PyGame board cells

        :param state: board state from which the move should be made
        :param action: move to be made
        :param player: player making the move
        :param moves_list: list of all possible moves
        :param captures_list: list of all possible captures
        :param to_render: render the board flag
        :return: board state after performing the move, positions of new king (if one has been made, else None)
        """

        starting_position = moves_list[action][0]
        new_position = moves_list[action][-1]
        piece_type = int(abs(state[starting_position]))

        if len(captures_list[action]) != 0 and isinstance(captures_list[action][0], int):
            captures = [captures_list[action]]
        else:
            captures = captures_list[action]

        if to_render:
            # If the board has to be rendered, then get the pixel values for old and new piece coordinates
            start_pixel_x = self.centre_of_cells[starting_position][0]
            start_pixel_y = self.centre_of_cells[starting_position][1]
            new_pixel_x = self.centre_of_cells[new_position][0]
            new_pixel_y = self.centre_of_cells[new_position][1]
        else:
            start_pixel_x = None
            start_pixel_y = None
            new_pixel_x = None
            new_pixel_y = None

        if to_render:
            # Remove the piece at old position, if the game is being rendered
            self.drawer.remove_piece(self.pg_board, start_pixel_x, start_pixel_y)

        # Set the old piece position to 0, indicating empty space
        state[starting_position] = 0

        if len(captures) == 0:
            self.make_no_capture_move(state, piece_type, player, new_position, new_pixel_x, new_pixel_y, to_render)

        else:
            # If a capture has been made
            self.make_capture_move(state, piece_type, player, new_position,
                                   new_pixel_x, new_pixel_y, captures, to_render)

        new_king_position = self.check_new_king_made(state, player, new_position, new_pixel_x, new_pixel_y, to_render)

        return np.copy(state), new_king_position

    def check_new_king_made(self, state, player, new_position, new_pixel_x, new_pixel_y, to_render):
        """
        Check if a new king has been made during the move.
        This check is made by looking at the new position of the moved piece.
        If the new position is at the top row for the red player, or the bottom row for the black player,
        then the piece transforms to a king, for the respective player

        :param state: board state from which the move should be made
        :param player: player making the move
        :param new_position: position of the piece after movement
        :param new_pixel_x: x pixel coordinate of the new piece position on PyGame display
        :param new_pixel_y: y pixel coordinate of the new piece position on PyGame display
        :param to_render: render the board flag
        :return: new king position, if a new king was made, else None
        """

        new_king_position = None
        if (new_position[0] == 0 and player == 1) or (new_position[0] == 7 and player == -1):
            state[new_position] = 2 * player
            new_king_position = (new_position[0], new_position[1])
            if to_render:
                self.drawer.draw_king(self.pg_board, new_pixel_x, new_pixel_y, player)
        return new_king_position

    def make_no_capture_move(self, state, piece_type, player,
                             new_position, new_pixel_x, new_pixel_y, to_render):
        """
        Move a piece with no captures.
        Achieved by updating numpy board and PyGame display

        :param state: board state from which the move should be made
        :param piece_type: type of the moved piece; 1 - standard piece, 2 - king piece
        :param player: player making the move
        :param new_position: position of the piece after movement
        :param new_pixel_x: x pixel coordinate of the new piece position on PyGame display
        :param new_pixel_y: y pixel coordinate of the new piece position on PyGame display
        :param to_render: render the board flag
        :return: new king position, if a new king was made, else None
        """

        if piece_type == 1:
            # Set in the numpy board the value for the standard piece (-1 or 1, equal to player value)
            # Draw the standard piece to the display, if the game is being rendered
            state[new_position] = player
            if to_render:
                self.drawer.draw_piece(self.pg_board, new_pixel_x, new_pixel_y, player)
        else:
            # Set in the numpy board the value for the king piece (-2 or 2, double to player value)
            # Draw the king piece to the display, if the game is being rendered
            state[new_position] = 2 * player
            if to_render:
                self.drawer.draw_king(self.pg_board, new_pixel_x, new_pixel_y, player)

    def make_capture_move(self, state, piece_type, player, new_position,
                          new_pixel_x, new_pixel_y, captures, to_render):
        """
        Move a piece with captures.
        Achieved by iterating through all captures,
        and updating numpy board and PyGame display for each captured position (i.e. eliminating the captured piece).
        Lastly, the capturing piece is moved in the "standard" way

        updating numpy board and PyGame display

        :param state: board state from which the move should be made
        :param piece_type: type of the moved piece; 1 - standard piece, 2 - king piece
        :param player: player making the move
        :param new_position: position of the piece after movement
        :param new_pixel_x: x pixel coordinate of the new piece position on PyGame display
        :param new_pixel_y: y pixel coordinate of the new piece position on PyGame display
        :param captures: list of captures made in the move
        :param to_render: render the board flag
        :return: new king position, if a new king was made, else None
        """

        for capture in captures:
            if to_render:
                # If the game is being rendered, remove the capture piece from the display
                capture_pixel_x = self.centre_of_cells[capture][0]
                capture_pixel_y = self.centre_of_cells[capture][1]
                # self.drawer.remove_piece(self.pg_board, start_pixel_x, start_pixel_y)
                self.drawer.remove_piece(self.pg_board, capture_pixel_x, capture_pixel_y)

            # Set the captured position to 0, indicating empty space
            state[capture] = 0

        # Move the piece without any captures
        self.make_no_capture_move(state, piece_type, player, new_position, new_pixel_x, new_pixel_y, to_render)
