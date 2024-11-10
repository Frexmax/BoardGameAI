import pygame as pg


class Drawer:
    """
    Helper class for the tic-tac-toe board, tasked with drawing shapes on PyGame displays.
    """

    def __init__(self, draw_parameters):
        """
        Constructor for the tic-tac-toe helper drawing class.
        All parameters, from the provided draw_parameters are
        stored in the class as attributes.

        Additionally, the BLACK, and WHITE colors are defined from RGB

        :param draw_parameters: parameters for the Drawer
        """

        # Store basic size parameters for the shapes
        self.dot_radius = draw_parameters["DOT_R"]
        self.circle_radius = draw_parameters["CIRCLE_R"]
        self.inner_circle_radius = draw_parameters["INNER_CIRCLE_R"]
        self.cross_width = draw_parameters["CROSS_WIDTH"]
        self.cross_height = draw_parameters["CROSS_HEIGHT"]
        self.cross_line_width = draw_parameters["CROSS_LINE_WIDTH"]

        # Store colors for the shapes, highlights and borders
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.border_color = draw_parameters["BORDER_COLOR"]
        self.cross_player_border_color = draw_parameters["CROSS_PLAYER_BORDER_COLOR"]
        self.circle_player_border_color = draw_parameters["CIRCLE_PLAYER_BORDER_COLOR"]
        self.cross_color = draw_parameters["CROSS_COLOR"]
        self.circle_color = draw_parameters["CIRCLE_COLOR"]
        self.highlight_color = draw_parameters["HIGHLIGHT_COLOR"]

    @staticmethod
    def fill_display(display, color):
        """
        Fill the provided PyGame display with a color, used for e.g. setting the background

        :param display: PyGame display which should be filled with a certain color
        :param color: color which should fill the display
        """

        display.fill(color)

    @staticmethod
    def draw_line(display, start_coordinate, end_coordinate, line_width, color):
        """
        Draw a line on the provided PyGame display from the start to end coordinates
        with the specified width

        :param display: PyGame display where the line should be drawn
        :param start_coordinate: starting coordinate of the line
        :param end_coordinate: ending coordinate of the line
        :param line_width: width of the line
        :param color: color of the line
        """

        pg.draw.line(display, color, start_coordinate, end_coordinate, line_width)

    def draw_border(self, display, width, height, border_width):
        """
        Draw a border around the grid on the provided PyGame display according to the
        provided display size and border with.

        :param display: PyGame display where the border should be drawn
        :param width: width of the PyGame display
        :param height: height of the PyGame display
        :param border_width: width of the border to be drawn
        """

        pg.draw.rect(display, self.border_color, (0, 0, width , height), border_width // 2)

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

        if player == 1:
            pg.draw.rect(display, self.cross_player_border_color, (0, 0, width, height), border_width // 4)
        else:
            pg.draw.rect(display, self.circle_player_border_color, (0, 0, width, height), border_width // 4)

    def draw_cross(self, display, x, y):
        """
        Draws a cross on the provided PyGame display, with the center at coordinates: (x, y).
        Used for marking first player moves

        :param display: PyGame display where the cross should be drawn
        :param x: x coordinate of the center of the cross
        :param y: y coordinate of the center of the cross
        """

        pg.draw.line(display, self.cross_color, (x - self.cross_width, y - self.cross_height),
                     (x + self.cross_width, y + self.cross_height), width=self.cross_line_width)
        pg.draw.line(display, self.cross_color, (x - self.cross_width, y + self.cross_height),
                     (x + self.cross_width, y - self.cross_height), width=self.cross_line_width)

    def draw_circle(self, display, x, y):
        """
        Draws a circle on the provided PyGame display, with the center at coordinates: (x, y).
        Used for marking second player moves

        :param display: PyGame display where the circle should be drawn
        :param x: x coordinate of the center of the circle
        :param y: y coordinate of the center of the circle
        """

        pg.draw.circle(display, self.circle_color, (x, y), self.circle_radius)
        pg.draw.circle(display, (255, 255, 255), (x, y), self.inner_circle_radius)

    def add_dot(self, display, x, y):
        """
        Draws a small dot (circle) on the provided PyGame display, with the center at coordinates: (x, y).
        Used for highlighting possible moves

        :param display: PyGame display where the circle should be drawn
        :param x: x coordinate of the center of the dot
        :param y: y coordinate of the center of the dot
        """

        pg.draw.circle(display, self.highlight_color, (x, y), self.dot_radius)

    def remove_dot(self, display, x, y):
        """
        Remove the small dots used for highlighting possible moves,
        by drawing a white (background color) dot in their place

        :param display: PyGame display where the circle should be drawn
        :param x: x coordinate of the center of the dot
        :param y: y coordinate of the center of the dot
        """

        pg.draw.circle(display, (255, 255, 255), (x, y), self.dot_radius)
