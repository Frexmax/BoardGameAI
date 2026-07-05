import numpy as np
from numba import jit

@jit(nopython=True, fastmath=True)
def apply_dirichlet_noise(action_probs, alpha, epsilon, action_space):
    """
    Add Dirichlet noise to the action probabilities of the root node, to encourage exploration
    during Monte-Carlo-Tree Search self-play

    :param action_probs: action probabilities predicted by the model
    :param alpha: concentration parameter of the Dirichlet distribution
    :param epsilon: weight given to the noise, relative to the original action probabilities
    :param action_space: number of possible actions
    :return: action probabilities with noise applied
    """

    return (1 - epsilon) * action_probs + epsilon * np.random.dirichlet([alpha]*action_space)

# @jit(nopython=True, fastmath=True)
def normalize_action(action_space, valid_moves, action_probs):
    """
    # TODO document-normalize

    :param action_space:
    :param valid_moves:
    :param action_probs:
    :return:
    """

    print(action_space)
    print(valid_moves)

    for i in range(action_space):
        if i not in valid_moves:
            action_probs[i] = 0
    if np.sum(action_probs) == 0:
        for i in valid_moves:
            action_probs[i] = 1 / len(valid_moves)
    else:
        action_probs /= np.sum(action_probs)
    return action_probs