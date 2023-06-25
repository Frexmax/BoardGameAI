import pygame as pg


class Drawer:
    def __init__(self, draw_parameters):
        # RADIUS
        self.piece_radius = draw_parameters["PIECE_R"]
        self.king_radius = draw_parameters["KING_R"]
        self.border_radius = draw_parameters["BORDER_R"]
        self.king_border_radius = draw_parameters["KING_BORDER_R"]
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

    def draw_king(self, display, x, y, player):
        border_color = self.player_border_colors[player]

        self.draw_piece(display, x, y, player)
        pg.draw.circle(display, border_color, (x, y), self.king_radius, self.king_border_radius)

    def remove_piece(self, display, x, y):
        pg.draw.circle(display, (0, 0, 0), (x, y), self.piece_radius)

    def add_dot(self, display, x, y):
        pg.draw.circle(display, self.highlight_color, (x, y), self.dot_radius)

    @staticmethod
    def remove_dot(display, x, y, cell_size):
        pg.draw.rect(display, (0, 0, 0), (x, y, cell_size[0], cell_size[1]))

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
