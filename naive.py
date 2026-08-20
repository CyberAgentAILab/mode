import time

import numpy as np

from market import Market


def naive(market: Market, verbose: bool = False) -> dict:
    """Naive matching algorithm that ranks jobs for each candidate based solely on the candidate's preferences.

    Args:
        market (Market): The market instance containing preferences and examination probabilities.
        verbose (bool): Whether to print detailed information about the algorithm's performance.

    Returns:
        dict: A dictionary containing the recommendation policy, social welfare, match probability matrix, suboptimality, and execution time.
    """
    start_time = time.time()
    rec = np.array(
        [
            sorted(
                [j for j in range(market.num_job)],
                key=lambda j: market.pref_cand[i, j],
                reverse=True,
            )[: market.rec_cand_slot]
            for i in range(market.num_cand)
        ]
    )
    end_time = time.time()
    match_probability = market.compute_match_probability(rec)

    if verbose:
        print(
            f"Naive: SW = {match_probability.sum():.5f}, Time = {end_time - start_time:.5f} [s]"
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
