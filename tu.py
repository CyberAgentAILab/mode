import time

import numpy as np

from market import Market


def tu(
    market: Market,
    beta: float = 1.0,
    maxit: int = 10000,
    tol: float = 1e-9,
    verbose: bool = False,
) -> dict:
    """TU Matching algorithm based on Tomita et al. (2023).

    Args:
        market (Market): The market instance containing preferences and examination probabilities.
        beta (float): The temperature parameter.
        maxit (int): Maximum number of iterations.
        tol (float): Tolerance level for convergence.
        verbose (bool): Whether to print detailed information about the algorithm's performance.

    Returns:
        dict: A dictionary containing the recommendation policy, social welfare, match probability matrix, suboptimality, and execution time.
    """
    start_time = time.time()
    K = np.exp((market.pref_cand + market.pref_job.T) / (2.0 * beta))
    A = np.ones(market.num_cand)
    B = np.ones(market.num_job)

    for it in range(maxit):
        KB = (K @ B) / 2.0
        new_A = np.sqrt(np.ones(market.num_cand) + KB * KB) - KB
        update = np.max(np.abs(new_A - A))
        A = new_A

        KA = (K.T @ A) / 2.0
        new_B = np.sqrt(np.ones(market.num_job) + KA * KA) - KA
        update = max(update, np.max(np.abs(new_B - B)))
        B = new_B

        if update < tol:
            if verbose:
                print(f"Converged at iteration {it}")
            break
    else:
        if verbose:
            print(f"Reached maximum iterations {maxit} without convergence")

    mu = (
        K
        * np.tile(A.reshape(market.num_cand, 1), reps=(1, market.num_job))
        * np.tile(B.reshape(1, market.num_job), reps=(market.num_cand, 1))
    )
    rec = [[j for j in range(market.num_job)] for i in range(market.num_cand)]

    for i in range(market.num_cand):
        rec[i].sort(key=lambda j: mu[i, j], reverse=True)
        rec[i] = rec[i][: market.rec_cand_slot]

    rec = np.array(rec)

    end_time = time.time()
    match_probability = market.compute_match_probability(rec)

    if verbose:
        print(
            f"TU Matching: SW = {match_probability.sum():.5f}, Time = {end_time - start_time:.5f} [s]"
        )

    result = {
        "rec": rec,
        "sw": match_probability.sum(),
        "match_probability": match_probability,
        "suboptimality": market.compute_direct_effect_suboptimality(
            rec=rec, match_probability=match_probability, eps=0.0
        ),
        "time": end_time - start_time,
    }

    return result
