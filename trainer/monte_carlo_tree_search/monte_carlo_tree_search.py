import time

import numpy as np

from trainer.monte_carlo_tree_search.node.node import Node
from trainer.monte_carlo_tree_search.utils.utils import apply_dirichlet_noise, normalize_action


class MonteCarloTreeSearch:
    """
    Class implementing the Monte Carlo Tree Search algorithm.
    This algorithm relies on intelligent search, by prioritizing some paths,
    depending on their game value. This game value is determined by a neural network
    trained via reinforcement learning, via domain-specific knowledge, or environment feedback.
    """

    def __init__(self, env, model, budget, heuristic_weight, alpha=1, epsilon=0.25):
        """
        Initialize the parameters of the Monte Carlo Tree,
        and the dirichlet noise used in it.

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

    
    def run(self, state, state_player, player, root=None):
        """
        Run the search algorithm, as long as the budget allows, from the provided state.

        :param state: the environment state from which the tree search should be run
        :param state_player: the environment state from the player's perspective
        :param player: the player who is about make a move
        :param root: root node - the start of the tree
        :return: the root node
        """

        # Initialize the budget tracking, determining the termination of the search
        num_rollouts = 0
        start_time = time.time()
        time_taken = 0

        # TODO explain-start-player
        start_player = player

        if root is None or len(root.children) == 0:
            # Create a root node, meaning a root with no ancestors - the start of the tree
            root = Node(0, player)
            action_probs, _ = self.model.predict(state_player)
            action_probs = action_probs.numpy()[0]
            valid_moves = self.env.find_moves(state, player, *self.env.find_positions(state))

            # Create children of the root node
            noised_action_probs = apply_dirichlet_noise(action_probs, self.alpha, self.epsilon, self.env.action_space)
            action_probs = normalize_action(self.env.action_space, np.array(valid_moves), noised_action_probs)
            root.expand(state, player, action_probs)

        # Run the loop until the number of rollouts has been reached
        # or the allocated time has elapsed
        # TODO check-time-tracking
        while self.budget > num_rollouts and self.budget > time_taken:
            node = root
            search_path = [node]

            # Select the node to simulate next, and the action needed to reach that node
            while node.expanded():
                action, node = node.select_child(self.env, len(search_path), start_player)
                search_path.append(node)

            # Get the state of the child node by making the action
            parent = search_path[-2]
            state = parent.state
            next_state = self.env.make_move(state, action, node.player)
            next_state_enemy = self.env.refactor_state(next_state, -node.player, len(search_path))

            # Calculate the value of the child node state, and get the valid moves in that state
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
        Backpropagate the value from the environment termination to all its ancestors.

        :param search_path: list of nodes leading to the termination
        :param value: value at the termination
        :param player: player who was to make the move at the end (i.e. the one who lost)
        """

        for node in reversed(search_path):
            if node.player == player:
                node.value_sum += -value
            else:
                node.value_sum += value
            node.visit_count += 1
