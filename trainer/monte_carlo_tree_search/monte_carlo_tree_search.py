import time
import numpy as np
from numba import jit
from node import node


def apply_dirichlet_noise(action_probs, alpha, epsilon, action_space):
    return (1 - epsilon) * action_probs + epsilon * np.random.dirichlet([alpha]*action_space)


class MonteCarloTreeSearch:
    def __init__(self, env, model, budget, heuristic_weight, alpha=1, epsilon=0.25):
        self.env = env
        self.model = model
        self.budget = budget
        self.alpha = alpha
        self.epsilon = epsilon
        self.heuristic_weight = heuristic_weight

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def normalize_action(action_space, valid_moves, action_probs):
        for i in range(action_space):
            if i not in valid_moves:
                action_probs[i] = 0
        if np.sum(action_probs) == 0:
            for i in valid_moves:
                action_probs[i] = 1 / len(valid_moves)
        else:
            action_probs /= np.sum(action_probs)  # NORMALIZE PROBS TO SUM EQUAL 1
        return action_probs

    @staticmethod
    def create_root(player):
        return Node(0, player)

    def run(self, state, state_player, player, root=None):
        # INITIALIZE BUDGET COUNT
        num_rollouts = 0
        start_time = time.time()
        time_taken = 0
        start_player = player
        if root is None or len(root.children) == 0:
            # SET ROOT
            root = Node(0, player)
            action_probs, _ = self.model.predict(state_player)
            action_probs = action_probs.numpy()[0]
            valid_moves = self.env.find_moves(state, player, *self.env.find_positions(state))

            # EXPAND ROOT
            noised_action_probs = apply_dirichlet_noise(action_probs, self.alpha, self.epsilon, self.env.action_space)
            action_probs = self.normalize_action(self.env.action_space, np.array(valid_moves), noised_action_probs)
            root.expand(state, player, action_probs)

        while self.budget > num_rollouts and self.budget > time_taken:
            node = root
            search_path = [node]
            # SELECT
            while node.expanded():
                action, node = node.select_child(self.env, len(search_path), start_player)
                search_path.append(node)
            parent = search_path[-2]
            state = parent.state
            next_state = self.env.make_move(state, action, node.player)
            next_state_enemy = self.env.refactor_state(next_state, -node.player, len(search_path))
            value, game_end, valid_moves = self.env.state_reward(next_state, node.player, len(search_path))
            if not game_end:  # IF THE GAME HAS NOT ENDED => EXPAND
                action_probs, network_value = self.model.predict(next_state_enemy)
                action_probs = action_probs.numpy()[0]
                network_value = network_value.numpy()[0][0]

                # ADD HEURISTICS
                if self.heuristic_weight != 0:
                    heuristics_value = self.env.add_heuristics(next_state, -node.player, valid_moves)
                else:
                    heuristics_value = 0
                value = self.heuristic_weight * heuristics_value + (1 - self.heuristic_weight) * network_value

                # APPLY NOISE AND NORMALIZE
                noised_action_probs = apply_dirichlet_noise(action_probs, self.alpha, self.epsilon,
                                                            self.env.action_space)
                action_probs = self.normalize_action(self.env.action_space, np.array(valid_moves), noised_action_probs)
                node.expand(next_state, -parent.player, action_probs)
            self.backpropagate(search_path, value, parent.player * -1)
            time_taken = time.time() - start_time
            num_rollouts += 1
        return root

    @staticmethod
    def backpropagate(search_path, value, player):
        for node in reversed(search_path):
            if node.player == player:
                node.value_sum += -value
            else:
                node.value_sum += value
            node.visit_count += 1
