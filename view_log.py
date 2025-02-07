import sys
import pickle


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise TypeError("Usage:  make view-log <game name> <log name>")

    game = sys.argv[1]
    log_name = sys.argv[2]
    with open(f"./{game}/saved_models/saved_logs/{log_name}", "rb") as logger_file:
        logger = pickle.load(logger_file)
    logger.graph_log()
