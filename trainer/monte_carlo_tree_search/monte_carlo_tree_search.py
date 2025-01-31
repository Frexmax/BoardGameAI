import time

import numpy as np

from trainer.monte_carlo_tree_search.node.node import Node
from utils.utils import apply_dirichlet_noise, normalize_action


class MonteCarloTreeSearch:
    """

    """


    def __init__(self, env, model, budget, heuristic_weight, alpha=1, epsilon=0.25):
        """
        TODO: explain-constructor-MCTS

        :param env: the environment providing state and actions
        :param model: neural networks for action probability and state value calculation
        :param budget: the maximum number of rollouts to perform
        :param heuristic_weight: weight provided to domain specific information
        :param alpha: alpha parameter of dirichlet noise
        :param epsilon: epsilon parameter of dirichlet noise
        """

        self.env = env
        self.model = model
        self.budget = budget
        self.alpha = alpha
        self.epsilon = epsilon
        self.heuristic_weight = heuristic_weight

    @staticmethod
    def create_root(player):
        """

        :param player:
        :return:
        """

        return Node(0, player)

    def run(self, state, state_player, player, root=None):
        """

        :param state:
        :param state_player:
        :param player:
        :param root:
        :return:
        """

        # Initialize the budget tracking, determining the termination of the search
        num_rollouts = 0
        start_time = time.time()
        time_taken = 0

        # TODO explain-start-player
        start_player = player

        if root is None or len(root.children) == 0:
            # SET ROOT
            root = Node(0, player)
            action_probs, _ = self.model.predict(state_player)
            action_probs = action_probs.numpy()[0]
            valid_moves = self.env.find_moves(state, player, *self.env.find_positions(state))

            # EXPAND ROOT TODO explain-root
            noised_action_probs = apply_dirichlet_noise(action_probs, self.alpha, self.epsilon, self.env.action_space)
            action_probs = normalize_action(self.env.action_space, np.array(valid_moves), noised_action_probs)
            root.expand(state, player, action_probs)

        # Run the loop until the number of rollouts has been reached
        # or the allocated time has elapsed
        # TODO check-time-tracking
        while self.budget > num_rollouts and self.budget > time_taken:
            node = root
            search_path = [node]

            # SELECT TODO explain-select
            while node.expanded():
                action, node = node.select_child(self.env, len(search_path), start_player)
                search_path.append(node)
            parent = search_path[-2]
            state = parent.state
            next_state = self.env.make_move(state, action, node.player)
            next_state_enemy = self.env.refactor_state(next_state, -node.player, len(search_path))
            value, game_end, valid_moves = self.env.state_reward(next_state, node.player, len(search_path))


            if not game_end:
                # If the game hasn't ended then expand this node

                # Get the action probabilities in the node state, and the node state value
                # from the neural network
                action_probs, network_value = self.model.predict(next_state_enemy)
                action_probs = action_probs.numpy()[0]
                network_value = network_value.numpy()[0][0]

                if self.heuristic_weight != 0:
                    # Add heuristics based on domain-specific knowledge
                    heuristics_value = self.env.add_heuristics(next_state, -node.player, valid_moves)
                else:
                    heuristics_value = 0
                value = self.heuristic_weight * heuristics_value + (1 - self.heuristic_weight) * network_value

                # Apply dirichlet noise and normalize action probabilities to range 0-1, with sum equal to 1
                noised_action_probs = apply_dirichlet_noise(action_probs, self.alpha, self.epsilon,
                                                            self.env.action_space)
                action_probs = normalize_action(self.env.action_space, np.array(valid_moves), noised_action_probs)

                # Expand this node, i.e. create child nodes
                node.expand(next_state, -parent.player, action_probs)

            # Backpropagate the search result to parent nodes
            self.backpropagate(search_path, value, parent.player * -1)

            time_taken = time.time() - start_time
            num_rollouts += 1
        return root

    @staticmethod
    def backpropagate(search_path, value, player):
        """

        :param search_path:
        :param value:
        :param player:
        :return:
        """

        for node in reversed(search_path):
            if node.player == player:
                node.value_sum += -value
            else:
                node.value_sum += value
            node.visit_count += 1
