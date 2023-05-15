import numpy as np


def bare_kuhn_length(kappa, l_b: float):
    return l_b * (
            (2 * kappa + np.exp(-2 * kappa) - 1)
            /  # -------------------------------
            (1 - np.exp(-2 * kappa) * (2 * kappa + 1))
    )
