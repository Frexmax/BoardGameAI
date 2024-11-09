import matplotlib.pyplot as plt
import pickle
from datetime import date


class Logger:
    def __init__(self, env_name):
        self.env_name = env_name
        self.num_items, self.load_epoch, self.load_episode, self.load_step = 0, 0, 0, 0
        self.info = {"epochs": [0], "episodes": [0], "steps": [0], "loss_rate": [0],
                     "win_rate": [0], "loss": [0]}

    def update_log(self, epoch, episode, step, loss_rate, win_rate, loss):

        self.info["episodes"].append(episode + self.load_episode)
        self.info["epochs"].append(epoch + self.load_epoch)
        self.info["steps"].append(step + self.load_step)

        self.info["loss_rate"].append(loss_rate)
        self.info["win_rate"].append(win_rate)
        self.info["loss"].append(loss)
        self.num_items += 1

    def print_log(self):
        for key in self.info:
            if key in ("steps", "episodes", "ëpochs"):
                print(key.upper(), ":", self.info[key][-1])
            elif key in ("win_rate", "loss_rate"):
                print(key.upper(), ":", self.info[key][-1] * 100, "%")
            else:
                print(key.upper(), ":", "{:.5f}".format(self.info[key][-1]))
        print("============================")

    def graph_log(self):
        figure, axis = plt.subplots(2)

        ax2 = axis[0].twinx()
        win_rate = axis[0].plot(self.info["episodes"], self.info["win_rate"], label='win rate',
                                color="green", linewidth=2.0)
        loss_rate = axis[0].plot(self.info["episodes"], self.info["loss_rate"], label='loss rate',
                                 color="red", linewidth=2.0)
        axis[0].set_xlabel('Episodes')
        axis[0].set_ylabel('Win Rate (%)')
        ax2.set_ylabel("Loss Rate (%)")
        axis[0].set_title("Win And Loss Rates History")
        axis[0].legend(loc="upper left")
        # ax2.legend(loc="upper left", bbox_to_anchor=(0, 0.825))

        actor_loss = axis[1].plot(self.info["episodes"], self.info["loss"], label='actor loss rate',
                                  color="blue", linewidth=2.0)
        axis[1].set_xlabel('Episodes')
        axis[1].set_ylabel('Loss')
        axis[1].set_title("Loss History")
        axis[1].legend(loc="upper left")

        # GRAPH DATA
        plt.show()

    def save(self):
        self.load_epoch = self.info["epochs"][-1]
        self.load_episode = self.info["episodes"][-1]
        self.load_step = self.info["steps"][-1]
        today = date.today()
        file_name = f"log-{self.env_name}-month-{today.month}-day-{today.day}-ep-" \
                    f"{self.info['steps'][-1]}-{self.info['win_rate'][-1]*100}%"
        with open(f"saved_models/saved_logs/{file_name}.pkl", 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def clear(self):
        self.info = {"epochs": [], "episodes": [], "steps": [], "reward": [],
                     "win_rate": [], "loss": []}
        self.num_items, self.load_epoch, self.load_episode, self.load_step = 0, 0, 0, 0
