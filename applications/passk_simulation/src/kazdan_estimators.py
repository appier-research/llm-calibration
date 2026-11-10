"""
pass@k estimators for comparing probe-based prediction with Kazdan et al. (2025).

Reference: Kazdan et al., "Efficient Prediction of Pass@k Scaling in Large Language Models"
(https://arxiv.org/abs/2510.05197).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln


def beta_passk(alpha, beta, k):
    """pass@k = 1 - E[(1 - p)^k] for p ~ Beta(alpha, beta) (Kazdan et al., Eq. 17)."""
    return 1.0 - np.exp(betaln(alpha, beta + k) - betaln(alpha, beta))


def probe_plugin_passk(confidence, k):
    """pass@k = 1 - (1 - q)^k, treating the probe confidence q as the true success rate."""
    return 1.0 - (1.0 - confidence) ** k


def probe_bayes_passk(confidence, successes, attempts, prior_strength, k, eps=1e-4):
    """
    Posterior-predictive pass@k with the probe confidence q as a per-question prior.

    Prior Beta(lambda * q, lambda * (1 - q)) is updated with the observed counts, giving
    posterior Beta(lambda * q + s, lambda * (1 - q) + n - s).
    """
    q = np.clip(confidence, eps, 1 - eps)
    alpha = prior_strength * q + successes
    beta = prior_strength * (1 - q) + attempts - successes
    return beta_passk(alpha, beta, k)


def fit_beta_binomial(successes, attempts):
    """Maximum-likelihood Beta(alpha, beta) from per-question binomial counts (Kazdan et al., Eq. 16)."""
    def neg_log_likelihood(log_params):
        alpha, beta = np.exp(log_params)
        return -np.sum(betaln(successes + alpha, attempts - successes + beta) - betaln(alpha, beta))

    mean = np.clip(successes.sum() / attempts.sum(), 1e-3, 1 - 1e-3)
    starts = [np.log([c * mean, c * (1 - mean)]) for c in (0.5, 2.0, 10.0)]
    fits = [minimize(neg_log_likelihood, x0, method='L-BFGS-B', bounds=[(-10, 10)] * 2) for x0 in starts]
    alpha, beta = np.exp(min(fits, key=lambda fit: fit.fun).x)
    return alpha, beta


def dynamic_sampling(pool, total_budget, rng):
    """
    Kazdan et al.'s dynamic sampler (Algorithm 1), capped by each question's pool size.

    Each step samples the question with the fewest successes, breaking ties by fewest attempts and
    then uniformly at random. Question i's j-th attempt reveals pool[i, j]; questions whose pool is
    exhausted are no longer sampled.

    Args:
        pool: (m, pool_size) array of pre-shuffled 0/1 outcomes.
        total_budget: Total number of samples to allocate across all questions.
        rng: numpy Generator for tie-breaking.

    Returns:
        successes, attempts: (m,) integer arrays.
    """
    m, cap = pool.shape
    successes = np.zeros(m, dtype=int)
    attempts = np.zeros(m, dtype=int)

    for _ in range(min(total_budget, m * cap)):
        available = attempts < cap
        hardest = np.flatnonzero(available & (successes == successes[available].min()))
        hardest = hardest[attempts[hardest] == attempts[hardest].min()]
        i = rng.choice(hardest)
        successes[i] += pool[i, attempts[i]]
        attempts[i] += 1

    return successes, attempts
