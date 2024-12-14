import sys

from checkers.test_checkers import test_checkers
from checkers.train_checkers import train_checkers
from simple_games.connect4.test_connect4 import test_connect4
from simple_games.connect4.train_connect4 import train_connect4
from simple_games.tic_tac_toe.train_tic_tac_toe import train_tic_tac_toe
from simple_games.tic_tac_toe.test_tic_tac_toe import test_tic_tac_toe

# HUMAN = True
# EPISODES = 5
# NUM_SIMULATIONS = 500
# AI_PLAYER = -1


if __name__ == "__main__":
    game = sys.argv[1]
    mode = sys.argv[2]

    if mode == "train":
        if game == "checkers":
            train_checkers()
        elif game == "connect4":
            train_connect4()
        else:
            train_tic_tac_toe()
    else:
        human = bool(sys.argv[3])
        episodes = int(sys.argv[4])
        num_simulations = int(sys.argv[5])
        ai_player = int(sys.argv[6])
        heuristic_weight = float(sys.argv[7])

        if game == "checkers":
            test_checkers(human, episodes, num_simulations, ai_player, heuristic_weight)
        elif game == "connect4":
            test_connect4(human, episodes, num_simulations, ai_player)
        else:
            test_tic_tac_toe(human, episodes, num_simulations, ai_player)
