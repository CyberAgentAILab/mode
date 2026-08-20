import time

import numpy as np

from market import Market


def mode(
    market: Market,
    max_iter: int = 1000,
    initial_rec: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """MODE algorithm that iteratively compute the optimal recommendation in direct effects for each candidate.

    Args:
        market (Market): The market instance containing preferences and examination probabilities.
        max_iter (int): Maximum number of iterations.
        initial_rec (np.ndarray | None): Initial recommendation policy. If None, a uniform random policy is used.
        verbose (bool): Whether to print detailed information about the algorithm's performance.

    Returns:
        dict: A dictionary containing the recommendation policy, social welfare, match probability matrix, suboptimality, execution time, and convergence history.
    """
    start_time = time.time()
    if initial_rec is None:
        rec = (
            np.ones(
                shape=(market.num_cand, market.num_job, market.rec_cand_slot),
                dtype=float,
            )
            / market.num_job
        )
    else:
        rec = initial_rec.copy()

    history = set()
    best_rec = rec.copy()
    best_sw = market.compute_match_probability(rec).sum()
    sw_history = [
        best_sw,
    ]
    candidates = [c for c in range(market.num_cand)]

    for _ in range(max_iter):
        utility = np.zeros((market.num_cand, market.num_job))
        if rec.shape == (market.num_cand, market.num_job, market.rec_cand_slot):
            for j in range(market.num_job):
                application_dist = np.zeros(market.rec_job_slot)
                application_dist[0] = 1.0
                candidates.sort(key=lambda i: market.pref_job[j, i], reverse=True)
                for i in candidates:
                    utility[i, j] = (
                        market.pref_cand[i, j]
                        * market.pref_job[j, i]
                        * (application_dist @ market.exam_job[j])
                    )
                    apply_prob = market.pref_cand[i, j] * (
                        rec[i, j] @ market.exam_cand[i]
                    )
                    application_dist = (
                        1 - apply_prob
                    ) * application_dist + apply_prob * np.concatenate(
                        [np.zeros(1), application_dist[:-1]]
                    )
        elif rec.shape == (market.num_cand, market.rec_cand_slot):
            tmp = -np.ones(shape=(market.num_cand, market.num_job), dtype=int)
            for i in range(market.num_cand):
                for k in range(market.rec_cand_slot):
                    tmp[i][rec[i, k]] = k
            for j in range(market.num_job):
                application_dist = np.zeros(market.rec_job_slot)
                application_dist[0] = 1.0
                candidates.sort(key=lambda x: market.pref_job[j, x], reverse=True)
                for i in candidates:
                    utility[i, j] = (
                        market.pref_cand[i, j]
                        * market.pref_job[j, i]
                        * (application_dist @ market.exam_job[j])
                    )
                    if tmp[i][j] >= 0:
                        apply_prob = (
                            market.pref_cand[i, j] * market.exam_cand[i, tmp[i][j]]
                        )
                        application_dist = (
                            1 - apply_prob
                        ) * application_dist + apply_prob * np.concatenate(
                            [np.zeros(1), application_dist[:-1]]
                        )
        else:
            raise ValueError(f"Invalid initial recommendation shape: {rec.shape}")

        new_rec = np.zeros(shape=(market.num_cand, market.rec_cand_slot), dtype=int)
        jobs = [j for j in range(market.num_job)]
        for i in range(market.num_cand):
            jobs.sort(key=lambda j: utility[i, j], reverse=True)
            for k in range(market.rec_cand_slot):
                new_rec[i, k] = jobs[k]
        new_rec_sw = market.compute_match_probability(new_rec).sum()
        if new_rec_sw > best_sw:
            best_sw = new_rec_sw
            best_rec = new_rec.copy()

        if verbose:
            print(
                f"Iteration {_ + 1}: SW = {new_rec_sw:.5f}, Time = {time.time() - start_time:.5f} [s]"
            )
        sw_history.append(new_rec_sw)

        if np.array_equal(rec, new_rec):
            if verbose:
                print("Converged.")
            status = "converged"
            break
        elif tuple(map(tuple, new_rec)) in history:
            if verbose:
                print("Cycle detected. Returning best recommendation found.")
            rec = best_rec
            status = "cycle_detected"
            break

        rec = new_rec
        history.add(tuple(map(tuple, rec)))
    else:
        status = "max_iter_reached"
        if verbose:
            print("Reached maximum iterations without convergence.")

    end_time = time.time()
    match_probability = market.compute_match_probability(rec)

    result = {
        "rec": rec,
        "sw": match_probability.sum(),
        "match_probability": match_probability,
        "suboptimality": market.compute_direct_effect_suboptimality(
            rec=rec, match_probability=match_probability, eps=0.0
        ),
        "time": end_time - start_time,
        "sw_history": sw_history,
        "status": status,
    }

    if verbose:
        print(
            f"Mode: SW: {result['sw']:.5f}, Time: {result['time']:.5f} [s], Status: {result['status']}"
        )

    return result
