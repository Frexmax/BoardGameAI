import sys

from checkers.test_checkers import test_checkers
from checkers.train_checkers import train_checkers


if __name__ == "__main__":
    game_to_play = sys.argv[1]
    run_mode = sys.argv[2]

    if game_to_play == "CHECKERS":
        if run_mode == "TRAIN":
            train_checkers()
        else:
            test_checkers()

    elif game_to_play == "CONNECT4":
        pass
    else:
        pass
