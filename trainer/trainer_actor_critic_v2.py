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
    def __init__(self, trainer_parameters, actor_critic_parameters, env_name, draw_parameters, board_parameters):
        # CREATE ENV
        if env_name == "TIC_TAC_TOE":
            self.env = TicTacToeEnv(board_parameters, draw_parameters)
        elif env_name == "CONNECT4":
            self.env = Connect4Env(board_parameters, draw_parameters)
        elif env_name == "CHECKERS":
            self.env = CheckersEnv(board_parameters, draw_parameters)
        else:
            raise NameError(f"ENV NAME '{env_name}' IS INCORRECT, TRY - 'TIC_TAC_TOE', 'CONNECT4', 'CHECKERS'")

        # GENERAL PARAMETERS
        self.env_name = env_name
        self.iterations = trainer_parameters["ITERATIONS"]
        self.data_generation_episodes = trainer_parameters["DATA_GENERATION_EPISODES"]

        self.model_path = trainer_parameters["MODEL_PATH"]
        self.logger_path = trainer_parameters["LOGGER_PATH"]
        self.test_games = trainer_parameters["TEST_GAMES"]
        self.num_workers = trainer_parameters["NUM_WORKERS"]

        # ENV PARAMETERS
        self.board_parameters = board_parameters
        self.draw_parameters = draw_parameters

        # RL PARAMETERS
        self.decay = trainer_parameters["DECAY"]
        self.tournament_games = trainer_parameters["TOURNAMENT_GAMES"]
        self.threshold = trainer_parameters["THRESHOLD"]

        # LOGGER
        self.logger = Logger(self.env_name)

        # ACTOR CRITIC PARAMETERS
        self.batch_size = actor_critic_parameters["BATCH_SIZE"]
        self.actor_critic = ActorCritic(actor_critic_parameters, self.env.refactored_space,
                                        self.env.action_space, env_name, training_mode="ITERATION")
        self.target_actor_critic = ActorCritic(actor_critic_parameters, self.env.refactored_space,
                                               self.env.action_space, env_name, training_mode="ITERATION")
        if trainer_parameters["LOAD"]:
            self.actor_critic.load_model(self.model_path)
            self.target_actor_critic.set_weights(self.actor_critic.get_weights())
            with open(self.logger_path, "rb") as logger_file:
                self.logger = pickle.load(logger_file)

        # BUFFER PARAMETERS
        self.memory_size = trainer_parameters["MEMORY_SIZE"]
        self.buffer = UniformBufferActorCritic(self.env.refactored_space, self.env.action_space,
                                               self.memory_size, self.batch_size)

        # MCTS PARAMETERS
        self.generator_simulations = trainer_parameters["MCTS_BUDGET"]
        self.tournament_simulations = trainer_parameters["TOURNAMENT_BUDGET"]
        self.test_simulations = trainer_parameters["TEST_BUDGET"]

        # MCTS HEURISTICS PARAMETERS
        self.heuristic_start_weight = trainer_parameters["HEURISTIC_START_WEIGHT"]
        self.heuristic_end_weight = trainer_parameters["HEURISTIC_END_WEIGHT"]
        self.heuristic_steps = trainer_parameters["HEURISTIC_STEPS"]
        self.heuristic_weight = self.heuristic_start_weight
        self.heuristic_weight_update = (self.heuristic_start_weight - self.heuristic_end_weight) / self.heuristic_steps

        self.actor_critic.save(self.logger.info["episodes"][-1], self.logger.info["win_rate"][-1])

    def generate_training_data(self, total_steps):
        self.actor_critic.save_self_play()
        parameters = [(self.env_name, self.board_parameters, self.draw_parameters, self.data_generation_episodes,
                      self.num_workers, self.generator_simulations, self.decay, self.heuristic_weight)]
        pool = Pool(self.num_workers)
        data = pool.starmap(generate_data, parameters*self.num_workers)
        pool.close()
        pool.join()
        gc.collect()

        # LOAD GENERATED DATA TO THE BUFFER
        for worker in range(self.num_workers):
            num_elements = data[worker][-1]
            for index in range(num_elements):
                state, action_probs, reward = data[worker][0][index], data[worker][1][index], data[worker][2][index]
                self.buffer.record((state, action_probs, reward))
                total_steps += 1
        return total_steps

    def update_logger(self, test_log, loss, iteration, steps):
        self.logger.update_log(1, iteration + 1, steps, test_log["loss"] / self.test_games,
                               test_log["win"] / self.test_games, loss)

    def print_log(self):
        self.logger.print_log()

    def test_network(self):
        self.actor_critic.save_test()
        parameters = [(self.env_name, self.board_parameters, self.draw_parameters,
                       self.test_simulations, self.heuristic_weight)]
        pool = Pool(self.test_games // 2)
        test_results = pool.starmap(play_test_game_pair, parameters*(self.test_games // 2))
        pool.close()
        pool.join()
        gc.collect()

        # LOG TEST GAME RESULTS
        test_log = {"win": 0, "loss": 0, "draw": 0}
        for game in test_results:
            test_log["win"] += game[0]
            test_log["loss"] += game[1]
            test_log["draw"] += game[2]
        return test_log

    def play_tournament(self):
        self.actor_critic.save_tournament(target=False)
        self.target_actor_critic.save_tournament(target=True)
        parameters = [(self.env_name, self.board_parameters, self.draw_parameters,
                       self.tournament_simulations, 0)]
        pool = Pool(self.tournament_games // 2)
        tournament_results = pool.starmap(tournament_pair, parameters*(self.tournament_games // 2))
        pool.close()
        pool.join()
        gc.collect()

        # COMPARE SCORES OF THE TRAINED AND TARGET MODELS
        score = {"trained": 0, "target": 0}
        for game in tournament_results:
            score["trained"] += game[0]
            score["target"] += game[1]
        print(f"TOURNAMENT SCORE: TRAINING: {score['trained']}, TARGET: {score['target']}")
        if score["trained"] + score["target"] == 0 or score["trained"] / (score["trained"] + score["target"]) >= self.threshold:
            print("COPYING AND SAVING")
            self.target_actor_critic.set_weights(self.actor_critic.get_weights())  # UPDATE TARGET MODEL
            # SAVE PROGRESS
            self.logger.save()
            self.actor_critic.save(self.logger.info["steps"][-1], self.logger.info["win_rate"][-1])
            self.heuristic_weight = max(0, self.heuristic_weight - self.heuristic_weight_update)  # UPDATE HEURISTICS
        else:
            self.actor_critic.set_weights(self.target_actor_critic.get_weights())

    def run(self):
        total_steps = 0
        for iteration in range(self.iterations):
            total_steps = self.generate_training_data(total_steps)
            loss = self.actor_critic.train(self.buffer)
            self.play_tournament()
            test_log = self.test_network()
            self.update_logger(test_log, loss, iteration, total_steps)
            self.print_log()
