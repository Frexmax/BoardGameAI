import pygame as pg


class Drawer:
    """
    Helper class for the connect-4 board, tasked with drawing shapes on PyGame displays.
    """
    def __init__(self, draw_parameters):
        """
        Constructor for the connect4 helper drawing class.
        All parameters, from the provided draw_parameters are
        stored in the class as attributes.

        Additionally, the BLACK, and WHITE colors are defined from RGB

        :param draw_parameters: parameters for the Drawer
        """

        # Store basic size parameters for the shapes
        self.piece_radius = draw_parameters["PIECE_R"]
        self.border_radius = draw_parameters["BORDER_R"]
        self.dot_radius = draw_parameters["DOT_R"]

        # Store colors for the shapes, highlights and borders
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

    def remove_piece(self, display, x, y):
        pg.draw.circle(display, (255, 255, 255), (x, y), self.piece_radius)

    def add_dot(self, display, x, y):
        pg.draw.circle(display, self.highlight_color, (x, y), self.dot_radius)

    def remove_dot(self, display, x, y):
        pg.draw.circle(display, (255, 255, 255), (x, y), self.dot_radius)
