import pygame as pg


class Drawer:
    def __init__(self, draw_parameters):
        # RADIUS
        self.piece_radius = draw_parameters["PIECE_R"]
        self.border_radius = draw_parameters["BORDER_R"]
        self.dot_radius = draw_parameters["DOT_R"]

        # COLORS
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
