import sys

from checkers.test_checkers import test_checkers
from checkers.train_checkers import train_checkers
from simple_games.connect4.test_connect4 import test_connect4
from simple_games.connect4.train_connect4 import train_connect4
from simple_games.tic_tac_toe.train_tic_tac_toe import train_tic_tac_toe
from simple_games.tic_tac_toe.test_tic_tac_toe import test_tic_tac_toe

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "train":
        if len(sys.argv) != 3:
            raise TypeError("Usage make train game:=<game name>")

        game = sys.argv[2]
        if game == "checkers":
            train_checkers()
        elif game == "connect4":
            train_connect4()
        elif game == "tic_tac_toe":
            train_tic_tac_toe()
        else:
            raise ValueError("Invalid game")

    elif mode == "test":
        if len(sys.argv) != 9:
            raise TypeError(
                """
                Usage: make test game:=<game name> human:=<is human> eps:=<ep count> 
                num_sim:=<MCTS sim num> ai_player:=<which player ai> heuristic_w:=<heuristic weight (0-1)>
                model_name:=<model name>
                """
            )

        game = sys.argv[2]
        human = bool(sys.argv[3])
        episodes = int(sys.argv[4])
        num_simulations = int(sys.argv[5])
        ai_player = int(sys.argv[6])
        heuristic_weight = float(sys.argv[7])
        model_name = sys.argv[8]

        if game == "checkers":
            test_checkers(human, episodes, num_simulations, ai_player, heuristic_weight, model_name)
        elif game == "connect4":
            test_connect4(human, episodes, num_simulations, ai_player, model_name)
        elif game == "tic_tac_toe":
            test_tic_tac_toe(human, episodes, num_simulations, ai_player, model_name)
        else:
            raise ValueError("Invalid game")

    else:
        raise ValueError("Invalid mode")
