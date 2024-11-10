import pickle
from datetime import date

import matplotlib.pyplot as plt


class Logger:
    """
    Class used for logging the progress of model training and plotting the collected data as matplotlib graphs
    """

    def __init__(self, env_name):
        """
        Constructor for the Logger class.
        All data points are initialized as empty.
        Logger state values are set to 0

        :param env_name: name of the training environment (i.e. name of the game)
        """

        self.env_name = env_name
        self.num_items, self.load_epoch, self.load_episode, self.load_step = 0, 0, 0, 0
        self.info = {"epochs": [0], "episodes": [0], "steps": [0], "loss_rate": [0],
                     "win_rate": [0], "loss": [0]}

    def update_log(self, epoch, episode, step, loss_rate, win_rate, loss):
        """
        Append the Logger with data from a training step

        :param epoch: training epoch
        :param episode: training episode
        :param step: training step
        :param loss_rate: loss rate in the testing
        :param win_rate: win rate in the testing
        :param loss: average loss value during model training
        """

        self.info["episodes"].append(episode + self.load_episode)
        self.info["epochs"].append(epoch + self.load_epoch)
        self.info["steps"].append(step + self.load_step)

        self.info["loss_rate"].append(loss_rate)
        self.info["win_rate"].append(win_rate)
        self.info["loss"].append(loss)

        self.num_items += 1

    def print_log(self):
        """
        Print the latest data from the log to the terminal
        """

        print("============================== LOG =================================")
        print("====================================================================")
        for key in self.info:
            if key in ("steps", "episodes", "epochs"):
                # Format training steps / episodes / epochs terminal output <- integer values
                print(key.upper(), ":", self.info[key][-1])
            elif key in ("win_rate", "loss_rate"):
                # Format win rate / loss rate terminal output <- percentage values
                print(key.upper(), ":", self.info[key][-1] * 100, "%")
            else:
                # Format model training loss terminal output <- floating point values
                print(key.upper(), ":", "{:.5f}".format(self.info[key][-1]))
        print("====================================================================")

    def graph_log(self):
        """
        Graph all Logger data as matplotlib graphs
        """

        # Create figure with two graphs
        figure, axis = plt.subplots(2, figsize=(10, 10))

        # Graph win and loss rate
        ax2 = axis[0].twinx()
        _ = axis[0].plot(self.info["episodes"], self.info["win_rate"], label='win rate',
                                color="green", linewidth=2.0)
        _ = axis[0].plot(self.info["episodes"], self.info["loss_rate"], label='loss rate',
                                 color="red", linewidth=2.0)
        axis[0].set_xlabel('Episodes')
        axis[0].set_ylabel('Win Rate (%)')
        ax2.set_ylabel("Loss Rate (%)")
        axis[0].set_title("Win And Loss Rates History")
        axis[0].legend(loc="upper left")

        # Graph loss rate
        _ = axis[1].plot(self.info["episodes"], self.info["loss"], label='actor loss rate',
                                  color="blue", linewidth=2.0)
        axis[1].set_xlabel('Episodes')
        axis[1].set_ylabel('Loss')
        axis[1].set_title("Loss History")
        axis[1].legend(loc="upper left")

        # Display the graphs
        plt.show()

    def save(self):
        """
        Save the Logger as a pickle file to the directory of the saved_logs directory of the environment.
        The file name is determined by the logged training data and month/day of training
        """

        self.load_epoch = self.info["epochs"][-1]
        self.load_episode = self.info["episodes"][-1]
        self.load_step = self.info["steps"][-1]
        today = date.today()
        file_name = f"log-{self.env_name}-month-{today.month}-day-{today.day}-ep-" \
                    f"{self.info['steps'][-1]}-{self.info['win_rate'][-1]*100}%"

        # Save logger with the specified file name
        with open(f"saved_models/saved_logs/{file_name}.pkl", 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def clear(self):
        """
        Clear the Logger of all values, reset to starting state
        """

        self.info = {"epochs": [], "episodes": [], "steps": [], "reward": [],
                     "win_rate": [], "loss": []}
        self.num_items, self.load_epoch, self.load_episode, self.load_step = 0, 0, 0, 0
