"""DD-domain detectors.

lmmse_detect : exact LMMSE over the full effective channel matrix. This is
    the reference detector — O((MN)^3) but fine at SW-model grid sizes.
mp_detect : Gaussian-approximation message passing over the sparse effective
    channel (Raviteja et al. style). Low-complexity path intended to mirror
    what a real-time receiver would run; exposed for comparison.
"""

import numpy as np


def lmmse_detect(H_d: np.ndarray, y: np.ndarray, noise_var: float) -> np.ndarray:
    """x_hat = (H^H H + sigma^2 I)^-1 H^H y for unit-energy symbols."""
    G = H_d.conj().T @ H_d
    G[np.diag_indices_from(G)] += noise_var
    return np.linalg.solve(G, H_d.conj().T @ y)


def mp_detect(H_d: np.ndarray, y: np.ndarray, noise_var: float,
              constellation: np.ndarray, n_iter: int = 20,
              damping: float = 0.6, sparsity_thresh: float = 1e-3) -> np.ndarray:
    """Message-passing detection with Gaussian interference approximation.

    Works on the sparse support of H_d (entries below sparsity_thresh *
    max|H| are dropped). Returns soft symbol means; hard-demap outside.
    """
    n_obs, n_var = H_d.shape
    A = constellation
    n_sym = A.size

    mag = np.abs(H_d)
    keep = mag > sparsity_thresh * mag.max()
    obs_i, var_i = np.nonzero(keep)
    h = H_d[obs_i, var_i]                      # edge gains
    n_edge = h.size

    # p[e, a]: prob of symbol a on the variable of edge e (var -> obs msg)
    p = np.full((n_edge, n_sym), 1.0 / n_sym)

    h_abs2 = np.abs(h) ** 2
    ea = np.abs(A) ** 2

    for _ in range(n_iter):
        # --- means/vars per edge from var->obs messages
        mean_e = p @ A                          # E[x] per edge
        pow_e = p @ ea                          # E[|x|^2] per edge
        var_e = pow_e - np.abs(mean_e) ** 2

        # totals per observation node
        mu_tot = np.zeros(n_obs, dtype=complex)
        np.add.at(mu_tot, obs_i, h * mean_e)
        v_tot = np.full(n_obs, noise_var)
        np.add.at(v_tot, obs_i, h_abs2 * var_e)

        # obs -> var: exclude own edge
        mu_ex = mu_tot[obs_i] - h * mean_e
        v_ex = v_tot[obs_i] - h_abs2 * var_e
        v_ex = np.maximum(v_ex, 1e-12)

        # log-likelihood of each constellation point on each edge
        resid = y[obs_i, None] - mu_ex[:, None] - h[:, None] * A[None, :]
        ll = -np.abs(resid) ** 2 / v_ex[:, None]

        # var-node update: product of incoming obs->var msgs, excl. own edge
        ll_var = np.zeros((n_var, n_sym))
        np.add.at(ll_var, var_i, ll)
        ll_ex = ll_var[var_i] - ll
        ll_ex -= ll_ex.max(axis=1, keepdims=True)
        p_new = np.exp(ll_ex)
        p_new /= p_new.sum(axis=1, keepdims=True)
        p = damping * p_new + (1 - damping) * p

    # final marginals per variable (all incoming messages)
    mean_e = p @ A
    pow_e = p @ ea
    var_e = pow_e - np.abs(mean_e) ** 2
    mu_tot = np.zeros(n_obs, dtype=complex)
    np.add.at(mu_tot, obs_i, h * mean_e)
    v_tot = np.full(n_obs, noise_var)
    np.add.at(v_tot, obs_i, h_abs2 * var_e)
    mu_ex = mu_tot[obs_i] - h * mean_e
    v_ex = np.maximum(v_tot[obs_i] - h_abs2 * var_e, 1e-12)
    resid = y[obs_i, None] - mu_ex[:, None] - h[:, None] * A[None, :]
    ll = -np.abs(resid) ** 2 / v_ex[:, None]
    ll_var = np.zeros((n_var, n_sym))
    np.add.at(ll_var, var_i, ll)
    best = ll_var.argmax(axis=1)
    return A[best]
