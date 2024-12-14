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
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.DARK_BLUE_BACKGROUND = (10, 7, 148)
        self.player_colors = {1: draw_parameters["RED_PLAYER_PIECE_COLOR"],
                              -1: draw_parameters["BLACK_PLAYER_PIECE_COLOR"]}
        self.player_border_colors = {1: draw_parameters["RED_PLAYER_BORDER_COLOR"],
                                     -1: draw_parameters["BLACK_PLAYER_BORDER_COLOR"]}
        self.highlight_color = draw_parameters["HIGHLIGHT_COLOR"]
        self.border_color = draw_parameters["BORDER_COLOR"]

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

    @staticmethod
    def fill_display(display, color):
        """
        Fill the provided PyGame display with a color, used for e.g. setting the background

        :param display: PyGame display which should be filled with a certain color
        :param color: color which should fill the display
        """

        display.fill(color)

    @staticmethod
    def draw_circle(display, x, y, radius, color):
        """
        Utility function to draw a simple circle on the provided PyGame display,
        with the center at coordinates: (x, y).

        :param display: PyGame display where the circle should be drawn
        :param x: x coordinate of the center of the circle
        :param y: y coordinate of the center of the circle
        :param radius: radius of the circle
        :param color: color of the circle (filled)
        """

        pg.draw.circle(display, color, (x, y), radius)

    def draw_piece(self, display, x, y, player):
        """
        Draw a connect4 piece on the provided PyGame display, with the color dependent
        on the player to whom the piece belongs

        :param display: PyGame display where the circle should be drawn
        :param x: x coordinate of the center of the piece
        :param y: y coordinate of the center of the piece
        :param player: player who is now to play
        """

        color = self.player_colors[player]
        border_color = self.player_border_colors[player]
        pg.draw.circle(display, color, (x, y), self.piece_radius)
        pg.draw.circle(display, border_color, (x, y), self.piece_radius, self.border_radius)

    def remove_piece(self, display, x, y):
        """
        Remove a connect4 piece from the provided PyGame display by drawing a white circle on its place

        :param display: PyGame display where the circle should be drawn
        :param x: x coordinate of the center of the circle
        :param y: y coordinate of the center of the circle
        """

        self.draw_circle(display, x, y, self.piece_radius, self.WHITE)

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

        pg.draw.circle(display, self.WHITE, (x, y), self.dot_radius)
