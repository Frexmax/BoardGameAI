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
        self.player_colors = {1: draw_parameters["RED_PLAYER_PIECE_COLOR"],
                              -1: draw_parameters["BLACK_PLAYER_PIECE_COLOR"]}
        self.player_border_colors = {1: draw_parameters["RED_PLAYER_BORDER_COLOR"],
                                     -1: draw_parameters["BLACK_PLAYER_BORDER_COLOR"]}
        self.highlight_color = draw_parameters["HIGHLIGHT_COLOR"]

    def draw_piece(self, display, x, y, player):
        color = self.player_colors[player]
        border_color = self.player_border_colors[player]

        pg.draw.circle(display, color, (x, y), self.piece_radius)
        pg.draw.circle(display, border_color, (x, y), self.piece_radius, self.border_radius)

    def draw_king(self, display, x, y, player):
        border_color = self.player_border_colors[player]

        self.draw_piece(display, x, y, player)
        pg.draw.circle(display, border_color, (x, y), self.king_radius, self.king_border_radius)

    def remove_piece(self, display, x, y):
        """
        Remove a checkers piece from the provided PyGame display by drawing a black circle on its place

        :param display: PyGame display where the circle should be drawn
        :param x: x coordinate of the center of the circle
        :param y: y coordinate of the center of the circle
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
        pg.draw.rect(display, self.highlight_color, (x_rect, y_rect, cell_size[0], cell_size[1]))
        if piece_type == 1:
            self.draw_piece(display, x_piece, y_piece, player)
        else:
            self.draw_king(display, x_piece, y_piece, player)

    def remove_highlight(self, display, x_rect, y_rect, x_piece, y_piece, piece_type, player, cell_size):
        pg.draw.rect(display, (0, 0, 0), (x_rect, y_rect, cell_size[0], cell_size[1]))
        if abs(piece_type) == 1:
            self.draw_piece(display, x_piece, y_piece, player)
        else:
            self.draw_king(display, x_piece, y_piece, player)
