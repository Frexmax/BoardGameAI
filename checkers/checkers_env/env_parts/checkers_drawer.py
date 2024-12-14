import pygame as pg


class Drawer:
    """
    Helper class for the checkers board, tasked with drawing shapes on PyGame displays.
    """

    def __init__(self, draw_parameters):
        """
        Constructor for the checkers helper drawing class.
        All parameters, from the provided draw_parameters are
        stored in the class as attributes.

        Additionally, the BLACK colors are defined from RGB

        :param draw_parameters: parameters for the Drawer
        """

        # Store basic size parameters for the shapes
        self.piece_radius = draw_parameters["PIECE_R"]
        self.king_radius = draw_parameters["KING_R"]
        self.border_radius = draw_parameters["BORDER_R"]
        self.king_border_radius = draw_parameters["KING_BORDER_R"]
        self.dot_radius = draw_parameters["DOT_R"]

        # Store colors for the shapes, highlights and borders
        self.BLACK = (0, 0, 0)
        self.border_color = draw_parameters["BORDER_COLOR"]
        self.player_colors = {1: draw_parameters["RED_PLAYER_PIECE_COLOR"],
                              -1: draw_parameters["BLACK_PLAYER_PIECE_COLOR"]}
        self.player_border_colors = {1: draw_parameters["RED_PLAYER_BORDER_COLOR"],
                                     -1: draw_parameters["BLACK_PLAYER_BORDER_COLOR"]}
        self.highlight_color = draw_parameters["HIGHLIGHT_COLOR"]

    pg.draw.rect(game_display, self.border_colour, (0, 0, self.width, self.height), self.border_width // 2)

    def draw_border(self, display, width, height, border_width):
        """
        Draw a border around the grid on the provided PyGame display according to the
        provided display size and border with.

        :param display: PyGame display where the border should be drawn
        :param width: width of the PyGame display
        :param height: height of the PyGame display
        :param border_width: width of the border to be drawn
        """

        pg.draw.rect(display, self.border_color, (0, 0, width, height), border_width // 2)

    def draw_player_border(self, display, width, height, border_width, player):
        """
        Draw a secondary border on the main border, based on the player who is about to play.
        This border is similarly dependent on the width and height of the provided PyGame display
        and the specified border width. This border is half as wide as the main one.

        :param display: PyGame display where the border should be drawn
        :param width: width of the PyGame display
        :param height: height of the PyGame display
        :param border_width: width of the border to be drawn
        :param player: player who is about to play
        """

        pg.draw.rect(display, self.player_border_colors[player], (0, 0, width, height), border_width // 4)

    def draw_piece(self, display, x, y, player):
        """
        Draw the piece on the provided PyGame display, with the center at coordinates (x, y).
        By drawing a filled circle and then a surrounding border

        :param display: PyGame display where the piece should be drawn
        :param x: x coordinate of the center of the piece
        :param y: y coordinate of the center of the piece
        :param player: the player to whom the piece belongs
        """

        color = self.player_colors[player]
        border_color = self.player_border_colors[player]

        pg.draw.circle(display, color, (x, y), self.piece_radius)
        pg.draw.circle(display, border_color, (x, y), self.piece_radius, self.border_radius)

    def draw_king(self, display, x, y, player):
        """
        Draw the king piece on the provided PyGame display, with the center at coordinates (x, y).
        By first drawing a standard piece and on top of it, a gray marking circle,
        to differentiate the standard piece from the king piece

        :param display: PyGame display where the king piece should be drawn
        :param x: x coordinate of the center of the king piece
        :param y: y coordinate of the center of the king piece
        :param player: the player to whom the king piece belongs
        """

        self.draw_piece(display, x, y, player)
        pg.draw.circle(display, self.border_color, (x, y), self.king_radius, self.king_border_radius)

    def remove_piece(self, display, x, y):
        """
        Remove a checkers piece from the provided PyGame display by drawing a black circle on its place

        :param display: PyGame display where the piece should be drawn
        :param x: x coordinate of the center of the piece
        :param y: y coordinate of the center of the piece
        """

        pg.draw.circle(display, self.BLACK, (x, y), self.piece_radius)

    def add_dot(self, display, x, y):
        """
         Draws a small dot (circle) on the provided PyGame display, with the center at coordinates: (x, y).
         Used for highlighting possible moves

         :param display: PyGame display where the circle should be drawn
         :param x: x coordinate of the center of the dot
         :param y: y coordinate of the center of the dot
         """

        pg.draw.circle(display, self.highlight_color, (x, y), self.dot_radius)

    def remove_dot(self, display, x, y, cell_size):
        """
        Remove the small dots used for highlighting possible moves,
        by re-drawing the grid cell where it was placed
        (checkers only played on black squares, therefore fill black)

        :param display: PyGame display where the grid cell should be drawn
        :param x: x coordinate of the top-left corner of the grid cell rectangle
        :param y: y coordinate of the top-left corner of the grid cell rectangle
        :param cell_size: size of the grid cell rectangle
        """

        pg.draw.rect(display, self.BLACK, (x, y, cell_size[0], cell_size[1]))

    def add_highlight(self, display, x_rect, y_rect, x_piece, y_piece, piece_type, player, cell_size):
        """
        Draw a highlighted grid cell on the provided PyGame display,
        with its top-left corner at position (x_rect, x_rect), and size determined by the tuple 'cell_size'.

        After drawing the highlighted grid cell, re-draw again the piece, which was highlighted,
        according to the 'piece' variable. If 'piece' = 1, then draw standard piece, 2 for king.
        The center of this piece is indicated by (x_piece, y_piece)

        :param display: PyGame display where the grid cell should be drawn
        :param x_rect: x coordinate of the top-left corner of the grid cell rectangle
        :param y_rect: y coordinate of the top-left corner of the grid cell rectangle
        :param x_piece: x coordinate of the center of the piece
        :param y_piece: y coordinate of the center of the piece
        :param piece_type: type of piece, which was highlighted
        :param player: player who highlighted the piece
        :param cell_size: size of the grid cell rectangle
        """

        pg.draw.rect(display, self.highlight_color, (x_rect, y_rect, cell_size[0], cell_size[1]))
        if piece_type == 1:
            self.draw_piece(display, x_piece, y_piece, player)
        else:
            self.draw_king(display, x_piece, y_piece, player)

    def remove_highlight(self, display, x_rect, y_rect, x_piece, y_piece, piece_type, player, cell_size):
        """
        Remove the highlight from the grid cell on the provided PyGame display by filling the cell black,
        with its top-left corner at position (x_rect, x_rect), and size determined by the tuple 'cell_size'.

        Afterward, the piece present there is re-drawn

        :param display: PyGame display where the grid cell should be drawn
        :param x_rect: x coordinate of the top-left corner of the grid cell rectangle
        :param y_rect: y coordinate of the top-left corner of the grid cell rectangle
        :param x_piece: x coordinate of the center of the piece
        :param y_piece: y coordinate of the center of the piece
        :param piece_type: type of piece, which was highlighted
        :param player: player who highlighted the piece
        :param cell_size: size of the grid cell rectangle
        """

        pg.draw.rect(display, self.BLACK, (x_rect, y_rect, cell_size[0], cell_size[1]))
        if abs(piece_type) == 1:
            self.draw_piece(display, x_piece, y_piece, player)
        else:
            self.draw_king(display, x_piece, y_piece, player)
