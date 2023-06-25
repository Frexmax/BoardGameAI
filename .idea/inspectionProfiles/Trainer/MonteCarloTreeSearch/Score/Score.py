import numpy as np
from numba import jit


@jit(nopython=True)
def ucb_score(c, parent_visit_count, child_prior, child_value, child_visit_count):
    prior_score = c * child_prior * np.sqrt(parent_visit_count) / (child_visit_count + 1)
    if child_visit_count > 0:
        value_score = -child_value
    else:
        value_score = 0
    return value_score + prior_score
