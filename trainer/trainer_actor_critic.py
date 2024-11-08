import random
import numpy as np
from ActorCritic import ActorCritic
from logger import logger
from tournament import tournament
from UniformBufferActorCritic import UniformBufferActorCritic
from TicTacToeEnv import TicTacToeEnv
from Connect4Env import Connect4Env
from CheckersEnv import CheckersEnv
from monte_carlo_tree_search import monte_carlo_tree_search


class TrainerActorCritic:
    def __init__(self, trainer_parameters, actor_critic_parameters, env_name, draw_parameters, board_parameters):
        # ENV
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
        self.load = trainer_parameters["LOAD"]
        self.model_path = trainer_parameters["MODEL_PATH"]
        self.test_every = trainer_parameters["TEST_EVERY"]
        self.test_games = trainer_parameters["TEST_GAMES"]
        self.save_every = trainer_parameters["SAVE_EVERY"]

        # RL PARAMETERS
        self.episodes = trainer_parameters["EPISODES"]
        self.decay = trainer_parameters["DECAY"]

        self.test_simulations = trainer_parameters["TEST_BUDGET"]
        self.tournament_every = trainer_parameters["TOURNAMENT_EVERY"]
        self.tournament_games = trainer_parameters["TOURNAMENT_GAMES"]
        self.tournament_simulations = trainer_parameters["TOURNAMENT_BUDGET"]

        # ACTOR CRITIC PARAMETERS
        self.batch_size = actor_critic_parameters["BATCH_SIZE"]
        self.actor_critic = ActorCritic(actor_critic_parameters, self.env.refactored_space,
                                        self.env.action_space, env_name, training_mode="ITERATION")
        self.target_actor_critic = ActorCritic(actor_critic_parameters, self.env.refactored_space,
                                               self.env.action_space, env_name, training_mode="ITERATION")

        # BUFFER PARAMETERS
        self.memory_size = trainer_parameters["MEMORY_SIZE"]
        self.min_replay_size = trainer_parameters["MIN_REPLAY_SIZE"]
        self.buffer = UniformBufferActorCritic(self.env.refactored_space, self.env.action_space,
                                               self.memory_size, self.batch_size)

        # MCTS PARAMETERS
        self.num_simulations = trainer_parameters["MCTS_BUDGET"]

        # LOGGER
        self.logger = Logger(self.env_name)

    def filter_actions(self, root):
        filtered_action_probs = np.zeros(self.env.action_space)
        for k, v in root.children.items():
            filtered_action_probs[k] = v.visit_count
        filtered_action_probs = filtered_action_probs / np.sum(filtered_action_probs)
        return filtered_action_probs

    def finish_episode(self, train_data, reward, last_player):
        count = 0
        for state, player, action_probs, mcts_value in reversed(train_data):
            adjusted_reward = (reward * (self.decay ** count))
            if last_player == player:
                adjusted_reward *= -1
            average_reward = (adjusted_reward + mcts_value) / 2
            self.buffer.record((state, action_probs, average_reward))
            count += 1

    def run(self):
        total_steps = 0
        losses = {"loss": []}
        for episode in range(self.episodes):
            done = False
            current_state, actions_index = self.env.reset()
            state_player = self.env.refactor_state(current_state, self.env.player, self.env.move_counter)
            train_examples = []
            reward = 0
            temperature = 1
            while not done:
                # RUN MONTE CARLO SEARCH
                mcts = MonteCarloTreeSearch(self.env, self.actor_critic, self.num_simulations, heuristic_weight=0)
                root = mcts.run(current_state, state_player, self.env.player)
                action = root.select_action(temperature=temperature)
                action_probs = self.filter_actions(root)
                train_examples.append((state_player, self.env.player, action_probs, root.value()))

                # PERFORM ACTION
                new_state, reward, done, actions_index = self.env.step(action)
                total_steps += 1

                # TRAIN ACTOR-CRITIC
                if self.buffer.buffer_counter > self.min_replay_size:
                    # print("TRAINING")
                    loss = self.actor_critic.train(self.buffer)
                    losses["loss"].append(loss)

                if self.env.move_counter == 30:
                    temperature = 0

                current_state = new_state
                state_player = self.env.refactor_state(new_state, self.env.player, self.env.move_counter)

            # AUGMENT AND RECORD TRAIN DATA
            self.finish_episode(train_examples, reward, self.env.player)

            # PLAY TOURNAMENT
            if episode % self.tournament_every == 0 and episode != 0:
                if tournament(self.env, self.actor_critic, self.target_actor_critic, self.tournament_games,
                              self.tournament_simulations, heuristic_weight=0):  # CHECK MODEL WIN
                    print("COPYING  AND SAVING")
                    model_weights = self.actor_critic.get_weights()
                    self.target_actor_critic.set_weights(model_weights)
                    self.logger.save()
                    self.actor_critic.save(self.logger.info["episodes"][-1], self.logger.info["win_rate"][-1])
                else:
                    target_model_weights = self.target_actor_critic.get_weights()
                    self.actor_critic.set_weights(target_model_weights)

            # TEST MODEL
            if episode % self.test_every == 0 and episode != 0:  # TEST CURRENT_MODEL
                test_log = {"reward": 0, "win": 0, "draw": 0, "loss": 0}
                for test_game in range(self.test_games):
                    done = False
                    reward = 0
                    current_state, actions_index = self.env.reset()
                    state_player = self.env.refactor_state(current_state, self.env.player, self.env.move_counter)
                    if test_game < self.test_games // 2:
                        red_player = "agent"
                        black_player = "random"
                    else:
                        red_player = "random"
                        black_player = "agent"
                    while not done:
                        if (self.env.player == 1 and red_player == "agent") or \
                                (self.env.player == -1 and black_player == "agent"):
                            mcts = MonteCarloTreeSearch(self.env, self.actor_critic,
                                                        self.num_simulations, heuristic_weight=0)
                            root = mcts.run(current_state, state_player, self.env.player)
                            action = root.select_action(temperature=0)
                        else:
                            action = random.sample(actions_index, 1)[0]
                        new_state, reward, done, actions_index = self.env.step(action)
                        current_state = new_state
                        state_player = self.env.refactor_state(new_state, self.env.player, self.env.move_counter)
                    if reward == 1:
                        if (red_player == "agent" and self.env.player == 1) or \
                                (black_player == "agent" and self.env.player == -1):
                            test_log["loss"] += 1
                            test_log["reward"] += -reward
                        else:
                            test_log["win"] += 1
                            test_log["reward"] += reward
                    else:
                        test_log["draw"] += 1
                        test_log["reward"] += reward

                if len(losses["loss"]) == 0:
                    mean_loss = 0
                else:
                    mean_loss = sum(losses["loss"]) / len(losses["loss"])

                # UPDATE AND PRINT LOG
                self.logger.update_log(1, episode + 1, total_steps,
                                       test_log["loss"] / self.test_games,
                                       test_log["win"] / self.test_games,
                                       mean_loss)
                self.logger.print_log()
                losses = {"loss": []}
