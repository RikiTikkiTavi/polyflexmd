import numpy as np


def rouse_R_sq_avg(t: np.ndarray, N_b: int, l_b: float) -> np.ndarray:
    return np.repeat(N_b * l_b ** 2, t.shape)


def rouse_R_autocorr(t: np.ndarray, p_max: int, N_b: int, l_b: float, tau_R: float) -> np.ndarray:
    result = 0
    for p in range(1, p_max):
        if p % 2 == 0:
            continue
        result += 1 / p ** 2 * np.exp(-t * p ** 2 / tau_R)
    return 8 * N_b * l_b ** 2 / np.pi ** 2 * result


def rouse_g_4(t: np.ndarray, tau_R: float, p_max: int, N_b: int, l_b: float) -> np.ndarray:
    s = 0
    for p in range(1, p_max + 1):
        if p % 2 == 0:
            continue
        s += 1 / p ** 2 * np.exp(-t * p ** 2 / tau_R)
    return 2 * N_b * l_b ** 2 * (1 - 8 / np.pi ** 2 * s)


def rouse_relaxation_time(N: int, l: float, zeta: float, T: float, k_B: float) -> float:
    return N ** 2 * l ** 2 * zeta / (3 * np.pi ** 2 * k_B * T)


def relaxation_time_bead(tau_R: float, N: int) -> float:
    return 3 * np.pi ** 2 * tau_R / N ** 2
