import numpy as np
from numba import jit

@jit(nopython=True, fastmath=True)
def apply_dirichlet_noise(action_probs, alpha, epsilon, action_space):
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