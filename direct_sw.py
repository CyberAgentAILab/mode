import time

import cvxpy as cp
import numpy as np
import torch

from market import Market


def direct_sw(
    market: Market,
    lr: float = 0.2,
    maxit: int = 100,
    tol: float = 1e-3,
    verbose: bool = False,
    cpsolver: str = "SCS",
) -> dict:
    """Direct social welfare maximization method via Frank-Wolfe algorithm.

    Args:
        market (Market): The market instance containing preferences and examination probabilities.
        lr (float): Learning rate for the gradient update.
        maxit (int): Maximum number of iterations.
        tol (float): Tolerance level for convergence.
        verbose (bool): Whether to print detailed information about the algorithm's performance.
        cpsolver (str): The solver to use for the convex optimization problem in CVXPY.

    Returns:
        dict: A dictionary containing the recommendation policy, social welfare, match probability matrix, suboptimality, execution time, and convergence history.
    """
    start_time = time.time()
    rec = torch.tensor(
        np.ones(shape=(market.num_cand, market.num_job, market.rec_cand_slot))
        / market.num_job,
        requires_grad=True,
    )

    prev_sw = 0.0
    sw_history = []

    candidates = [[i for i in range(market.num_cand)] for _ in range(market.num_job)]
    for j in range(market.num_job):
        candidates[j].sort(key=lambda i: market.pref_job[j, i], reverse=True)

    for it in range(maxit):
        match_probability = torch.zeros(market.num_cand, market.num_job)
        for j in range(market.num_job):
            application_dist = torch.zeros(market.rec_job_slot, dtype=torch.float64)
            application_dist[0] = 1.0
            for i in candidates[j]:
                apply_prob = market.pref_cand[i, j] * torch.dot(
                    torch.tensor(market.exam_cand[i]), rec[i, j]
                )
                match_probability[i, j] = (
                    apply_prob
                    * market.pref_job[j, i]
                    * torch.dot(torch.tensor(market.exam_job[j]), application_dist)
                )
                application_dist = (
                    1 - apply_prob
                ) * application_dist + apply_prob * torch.concatenate(
                    [torch.zeros(1), application_dist[:-1]]
                )

        sw = torch.sum(match_probability)

        if verbose:
            print(
                f"Iteration {it}: SW = {sw.item():.4f}, Change = {sw.item() - prev_sw:.4f}, Time = {time.time() - start_time:.5f} [s]"
            )
            sw_history.append(sw.item())
        if abs(sw.item() - prev_sw) < tol:
            status = "converged"
            if verbose:
                print(f"Convergence achieved at iteration {it}.")
            break

        prev_sw = sw.item()
        sw.backward()

        x = cp.Variable(shape=(market.num_cand * market.num_job, market.rec_cand_slot))
        objective = cp.Maximize(
            cp.sum(
                cp.multiply(
                    x,
                    rec.grad.numpy().reshape(
                        market.num_cand * market.num_job, market.rec_cand_slot
                    ),
                )
            )
        )
        constraints = [x >= 0, cp.sum(x, axis=1) <= 1] + [
            cp.sum(x[i * market.num_job : (i + 1) * market.num_job, :], axis=0) == 1
            for i in range(market.num_cand)
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cpsolver, verbose=False)

        rec.data = (1 - lr) * rec.data + lr * x.value.reshape(
            market.num_cand, market.num_job, market.rec_cand_slot
        )
        rec.grad.zero_()
    else:
        status = "maxit_reached"
        if verbose:
            print("Maximum iterations reached without convergence.")

    end_time = time.time()
    rec = rec.detach().numpy()
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
            f"Direct SW: SW: {result['sw']:.5f}, Time: {result['time']:.5f} [s], Status: {result['status']}"
        )

    return result
