import time

import cvxpy as cp
import numpy as np
import torch

from market import Market


def approx_sw(
    market: Market,
    lr: float = 0.2,
    maxit: int = 100,
    tol: float = 1e-3,
    verbose: bool = False,
    cpsolver: str = "SCS",
) -> dict:
    """Approximate social welfare maximization method via Frank-Wolfe algorithm (Su et al. 2022).

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
    approx_sw_history = []

    for it in range(maxit):
        apply_prob = torch.zeros(market.num_cand, market.num_job)
        expected_rank = torch.zeros(market.num_cand, market.num_job)
        tmp_expected_rank = torch.zeros(market.num_job)
        candidates = [i for i in range(market.num_cand)]

        for j in range(market.num_job):
            candidates.sort(key=lambda i: market.pref_job[j, i], reverse=True)
            for i in candidates:
                expected_rank[i, j] = tmp_expected_rank[j]
                apply_prob[i, j] = market.pref_cand[i, j] * torch.dot(
                    torch.tensor(market.exam_cand[i]), rec[i, j]
                )
                tmp_expected_rank[j] += apply_prob[i, j]

        accept_prob = torch.zeros(market.num_cand, market.num_job)
        for i in range(market.num_cand):
            for j in range(market.num_job):
                if expected_rank[i, j] >= market.rec_job_slot:
                    continue
                if market.exam_job_type == "log":
                    accept_prob[i, j] = market.pref_job[j, i] / torch.log2(
                        expected_rank[i, j] + 2
                    )
                elif market.exam_job_type == "inv":
                    accept_prob[i, j] = market.pref_job[j, i] / (
                        expected_rank[i, j] + 1
                    )
                elif market.exam_job_type == "exp":
                    accept_prob[i, j] = market.pref_job[j, i] * torch.exp(
                        -(expected_rank[i, j])
                    )
                elif market.exam_job_type is None:
                    raise ValueError(
                        "exam_job_type must be specified in ApproxSW method."
                    )
                else:
                    raise ValueError(f"Invalid exam type: {market.exam_job_type}")

        sw = torch.sum(apply_prob * accept_prob)

        if verbose:
            print(
                f"Iteration {it}: SW = {sw.item():.5f}, Change = {sw.item() - prev_sw:.5f}, Time = {time.time() - start_time:.5f} [s]"
            )
            approx_sw_history.append(sw.item())
            sw_history.append(
                market.compute_match_probability(rec.detach().numpy()).sum()
            )

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
        "approx_sw_history": approx_sw_history,
        "sw_history": sw_history,
        "status": status,
    }

    if verbose:
        print(
            f"Approx SW: SW: {result['sw']:.5f}, Time: {result['time']:.5f} [s], Status: {result['status']}"
        )

    return result
