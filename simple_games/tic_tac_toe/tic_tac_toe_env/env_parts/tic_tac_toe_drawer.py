import pygame as pg


class Drawer:
    def __init__(self, draw_parameters):
        # DIMENSIONS
        self.dot_radius = draw_parameters["DOT_R"]
        self.circle_radius = draw_parameters["CIRCLE_R"]
        self.inner_circle_radius = draw_parameters["INNER_CIRCLE_R"]
        self.cross_width = draw_parameters["CROSS_WIDTH"]
        self.cross_height = draw_parameters["CROSS_HEIGHT"]
        self.cross_line_width = draw_parameters["CROSS_LINE_WIDTH"]

        # COLORS
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.border_color = draw_parameters["BORDER_COLOR"]
        self.cross_player_border_color = draw_parameters["CROSS_PLAYER_BORDER_COLOR"]
        self.circle_player_border_color = draw_parameters["CIRCLE_PLAYER_BORDER_COLOR"]
        self.cross_color = draw_parameters["CROSS_COLOR"]
        self.circle_color = draw_parameters["CIRCLE_COLOR"]
        self.highlight_color = draw_parameters["HIGHLIGHT_COLOR"]

    def fill_display(self, display):
        display.fill(self.WHITE)

    def draw_line(self, display, start_coordinate, end_coordinate, line_width, filled):
        if filled:
            color = self.BLACK
        else:
            color = self.WHITE
        pg.draw.line(display, color, start_coordinate, end_coordinate, line_width)

    def draw_border(self, display, width, height, border_width):
        pg.draw.rect(display, self.border_color, (0, 0, width, height), border_width // 2)

    def draw_player_border(self, display, width, height, border_width, player):
        if player == 1:
            pg.draw.rect(display, self.cross_player_border_color, (0, 0, width, height), border_width // 4)
        else:
            pg.draw.rect(display, self.circle_player_border_color, (0, 0, width, height), border_width // 4)

    def draw_cross(self, display, x, y):
        pg.draw.line(display, self.cross_color, (x - self.cross_width, y - self.cross_height),
                     (x + self.cross_width, y + self.cross_height), width=self.cross_line_width)
        pg.draw.line(display, self.cross_color, (x - self.cross_width, y + self.cross_height),
                     (x + self.cross_width, y - self.cross_height), width=self.cross_line_width)

    def draw_circle(self, display, x, y):
        pg.draw.circle(display, self.circle_color, (x, y), self.circle_radius)
        pg.draw.circle(display, (255, 255, 255), (x, y), self.inner_circle_radius)

    def add_dot(self, display, x, y):
        pg.draw.circle(display, self.highlight_color, (x, y), self.dot_radius)

    def remove_dot(self, display, x, y):
        pg.draw.circle(display, (255, 255, 255), (x, y), self.dot_radius)
