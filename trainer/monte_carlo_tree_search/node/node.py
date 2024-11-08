import numpy as np

from trainer.monte_carlo_tree_search.score.score import ucb_score


class Node:
    def __init__(self, player, prior):
        self.visit_count = 0
        self.player = player
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self, temperature):
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)
        return action

    @staticmethod
    def check_winning_moves(env, action, state, player):
        next_state = env.make_move(state, action, player)
        value, game_end, valid_moves = env.state_reward(next_state, player, move_counter=0)
        if value == 1:
            return action
        return None

    @staticmethod
    def select_dict_action(env, state, path_length, start_player, current_player):
        move_count = env.move_counter + path_length - 1
        if move_count <= env.optimal_move_count and start_player == current_player:
            hashed_state = env.hash_state(state)
            if hashed_state in env.optimal_start_moves:
                return env.optimal_start_moves[hashed_state]
        return None

    def select_child(self, env, path_length, start_player):
        c = 4
        best_score = -np.inf
        best_action = -1
        best_child = None
        dict_action = self.select_dict_action(env, self.state, path_length, start_player, -self.player)
        dict_action = False  # CHANGE DUE TO HASH COLLISIONS
        if dict_action:
            best_action = dict_action
            best_child = self.children[dict_action]
        else:
            for action, child in self.children.items():
                winning_action = self.check_winning_moves(env, action, self.state, child.player)
                if winning_action is not None:
                    return winning_action, child
                score = ucb_score(c, self.visit_count, child.prior, child.value(), child.visit_count)
                if score > best_score:
                    best_score = score
                    best_action = action
                    best_child = child
        return best_action, best_child

    def expand(self, state, player, action_probs):
        self.player = player
        self.state = np.copy(state)
        for a, prob in enumerate(action_probs):
            if prob != 0:
                self.children[a] = Node(prior=prob, player=self.player)
