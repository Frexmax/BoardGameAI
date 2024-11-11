import gc
import pickle
from multiprocessing import Pool

from checkers.checkers_env.checkers_env import CheckersEnv
from simple_games.tic_tac_toe.tic_tac_toe_env.tic_tac_toe_env import TicTacToeEnv
from simple_games.connect4.connect4_env.connect4_env import Connect4Env
from trainer.tournament.tournament import tournament_pair
from trainer.algorithms.actor_critic.actor_critic import ActorCritic
from trainer.buffers.uniform_buffer_actor_critic import UniformBufferActorCritic
from trainer.test.test_games import play_test_game_pair
from trainer.data_generator.data_generator import generate_data
from trainer.logger.logger import Logger


class TrainerActorCriticV2:
    """
    Trainer Class responsible for the high-level view of the training process.
    The main training loop is run here, with method calls to other classes tasked with the sub-parts.
    """

    def __init__(self, trainer_parameters, actor_critic_parameters, env_name, draw_parameters, board_parameters):
        """
        Constructor the Trainer Class
        Initializes necessary objects and stores the provided parameters.
        If necessary load the previous states of models and logs

        :param trainer_parameters: parameters for the training process
        :param actor_critic_parameters: parameters for the Actor-Critic model
        :param env_name: name of the environment to be used for training
        :param draw_parameters: parameters for the environment Drawer
        :param board_parameters: parameters for the environment Board
        """

        # Initialize the environment based on the provided environment name,
        # else raise an error when an invalid name provided
        if env_name == "TIC_TAC_TOE":
            self.env = TicTacToeEnv(board_parameters, draw_parameters)
        elif env_name == "CONNECT4":
            self.env = Connect4Env(board_parameters, draw_parameters)
        elif env_name == "CHECKERS":
            self.env = CheckersEnv(board_parameters, draw_parameters)
        else:
            raise NameError(f"ENV NAME '{env_name}' IS INCORRECT, TRY - 'TIC_TAC_TOE', 'CONNECT4', 'CHECKERS'")

        # Store basic training process parameters
        self.env_name = env_name
        self.iterations = trainer_parameters["ITERATIONS"]
        self.test_games = trainer_parameters["TEST_GAMES"]

        # Store data generation parameters
        self.num_workers = trainer_parameters["NUM_WORKERS"]
        self.data_generation_episodes = trainer_parameters["DATA_GENERATION_EPISODES"]

        # Store environment parameters
        self.board_parameters = board_parameters
        self.draw_parameters = draw_parameters

        # Store reinforcement learning training parameters
        self.decay = trainer_parameters["DECAY"]
        self.tournament_games = trainer_parameters["TOURNAMENT_GAMES"]
        self.threshold = trainer_parameters["THRESHOLD"]

        # Initialize the Logger
        self.logger = Logger(self.env_name)

        # Store Actor-Critic model parameters and initialize Actor-Critic model
        self.batch_size = actor_critic_parameters["BATCH_SIZE"]
        self.actor_critic = ActorCritic(actor_critic_parameters, self.env.refactored_space,
                                        self.env.action_space, env_name, training_mode="ITERATION")
        self.target_actor_critic = ActorCritic(actor_critic_parameters, self.env.refactored_space,
                                               self.env.action_space, env_name, training_mode="ITERATION")

        # If a previous logger and model state is supposed to be loaded, then load it from the provided paths
        if trainer_parameters["LOAD"]:
            self.actor_critic.load_model(trainer_parameters["MODEL_PATH"])
            self.target_actor_critic.set_weights(self.actor_critic.get_weights())
            with open(trainer_parameters["LOGGER_PATH"], "rb") as logger_file:
                self.logger = pickle.load(logger_file)

        # Store experience buffer parameters and initialize the buffer
        self.memory_size = trainer_parameters["MEMORY_SIZE"]
        self.buffer = UniformBufferActorCritic(self.env.refactored_space, self.env.action_space,
                                               self.memory_size, self.batch_size)

        # Store Monte-Carlo-Tree Search parameters
        self.generator_simulations = trainer_parameters["MCTS_BUDGET"]
        self.tournament_simulations = trainer_parameters["TOURNAMENT_BUDGET"]
        self.test_simulations = trainer_parameters["TEST_BUDGET"]

        # Store and calculate the Monte-Carlo-Tree Search heuristics parameters
        self.heuristic_start_weight = trainer_parameters["HEURISTIC_START_WEIGHT"]
        self.heuristic_end_weight = trainer_parameters["HEURISTIC_END_WEIGHT"]
        self.heuristic_steps = trainer_parameters["HEURISTIC_STEPS"]
        self.heuristic_weight = self.heuristic_start_weight
        self.heuristic_weight_update = (self.heuristic_start_weight - self.heuristic_end_weight) / self.heuristic_steps

    def generate_training_data(self, total_steps):
        """
        Generate training data for the model using multiprocessing,
        with the number of workers specified in the parameters.

        :param total_steps: total steps covered by this point in the training, updated during this data generation
        :return: the update total step count
        """

        # Save the model to be loaded in each separate process
        self.actor_critic.save_self_play()

        # Generate data function parameters
        parameters = [(self.env_name, self.board_parameters, self.draw_parameters, self.data_generation_episodes,
                      self.num_workers, self.generator_simulations, self.decay, self.heuristic_weight)]

        # Set up process pool, generate data in each separate process, and receive the data in this method
        pool = Pool(self.num_workers)
        data = pool.starmap(generate_data, parameters*self.num_workers)

        # Clean up the pool
        pool.close()
        pool.join()
        gc.collect()

        # Load the generated data to the buffer and update the total step count
        for worker in range(self.num_workers):
            num_elements = data[worker][-1]
            for index in range(num_elements):
                state, action_probs, reward = data[worker][0][index], data[worker][1][index], data[worker][2][index]
                self.buffer.record((state, action_probs, reward))
                total_steps += 1
        return total_steps

    def update_logger(self, test_log, loss, iteration, steps):
        """
        Add the latest training data to the Logger

        :param test_log: results of the test matches
        :param loss: average loss during model training
        :param iteration: the current training iteration
        :param steps: total steps made till now
        """

        self.logger.update_log(1, iteration + 1, steps, test_log["loss"] / self.test_games,
                               test_log["win"] / self.test_games, loss)

    def print_log(self):
        """
        Wrapper method to print the latest Logger data
        """

        self.logger.print_log()

    def test_network(self):
        """
        Test the currently trained network against a random agent using multiprocessing,
        with the number of test rounds specified in the parameters

        :return: the results of the test matches
        """

        # Save the model to be loaded in each separate process
        self.actor_critic.save_test()

        # Test function parameters
        parameters = [(self.env_name, self.board_parameters, self.draw_parameters,
                       self.test_simulations, self.heuristic_weight)]

        # Set up the process pool, play a test game (with 2 rounds) in each process,
        # and receive the results of the games
        pool = Pool(self.test_games // 2)
        test_results = pool.starmap(play_test_game_pair, parameters*(self.test_games // 2))

        # Clean up the pool
        pool.close()
        pool.join()
        gc.collect()

        # Save and return the test game results by going through each played game (with 2 rounds)
        test_log = {"win": 0, "loss": 0, "draw": 0}
        for game in test_results:
            # Add the number of wins in the game
            test_log["win"] += game[0]

            # Add the number of losses in the game
            test_log["loss"] += game[1]

            # Add the number of draw in the game
            test_log["draw"] += game[2]
        return test_log

    def play_tournament(self):
        """
        Play a tournament of the currently trained model against the old (target) model using multiprocessing,
        with the number of tournament rounds specified in the parameters.

        If the trained model performs better than required by the threshold specified in the parameters, then
        the training progress is saved and the trained model becomes the target model (weights copied).
        Otherwise, the target model weights are copied to the trained model (training progress reversed)
        """

        # Save the currently trained model and the old (target) model to be loaded in the separate processes
        self.actor_critic.save_tournament(target=False)
        self.target_actor_critic.save_tournament(target=True)

        # Tournament function parameters
        parameters = [(self.env_name, self.board_parameters, self.draw_parameters,
                       self.tournament_simulations, 0)]

        # Set up the process pool and perform a two round tournament in each process,
        # with the result of each of these 2 round games being returned to tournament results
        pool = Pool(self.tournament_games // 2)
        tournament_results = pool.starmap(tournament_pair, parameters*(self.tournament_games // 2))

        # Clean up the pool
        pool.close()
        pool.join()
        gc.collect()

        # Get the final score of the trained and old (target) models, from all the rounds played
        score = {"trained": 0, "target": 0}
        for game in tournament_results:
            score["trained"] += game[0]
            score["target"] += game[1]

        # Display the result of the rounds
        print(f"TOURNAMENT SCORE: TRAINING: {score['trained']}, TARGET: {score['target']}")

        # If the trained model performed better than required by the threshold, then the model is saved and its weights
        # are copied to the target model. Training progress (log, model) is also saved and heuristic weight updated.
        # Otherwise, any weight changes made during training are reset, and the old (target) model weights are copied
        # to the currently trained one
        if (score["trained"] + score["target"] == 0
                or score["trained"] / (score["trained"] + score["target"]) >= self.threshold):
            print("COPYING AND SAVING")
            self.target_actor_critic.set_weights(self.actor_critic.get_weights())  # UPDATE TARGET MODEL

            # Save training progress
            self.logger.save()
            self.actor_critic.save(self.logger.info["steps"][-1], self.logger.info["win_rate"][-1])
            self.heuristic_weight = max(0, self.heuristic_weight - self.heuristic_weight_update)  # UPDATE HEURISTICS
        else:
            self.actor_critic.set_weights(self.target_actor_critic.get_weights())

    def run(self):
        """
        Main training loop, run for the specified number of training iterations.
        Each training iteration consists of:
        - data generation stage
        - model training stage on data in the buffer
        - tournament, where the trained model plays against the old model
        - test of the model against a random agent
        - add the latest training data to the Logger
        - printing the latest training data from the Logger
        """

        total_steps = 0
        for iteration in range(self.iterations):
            total_steps = self.generate_training_data(total_steps)
            loss = self.actor_critic.train(self.buffer)
            self.play_tournament()
            test_log = self.test_network()
            self.update_logger(test_log, loss, iteration, total_steps)
            self.print_log()
