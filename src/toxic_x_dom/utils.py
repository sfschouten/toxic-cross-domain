import numpy as np


def list_of_lists_to_numpy(list_of_lists, fill_value=None):
    max_len = max(map(len, list_of_lists))
    return np.array([xi + [fill_value] * (max_len - len(xi)) for xi in list_of_lists])

