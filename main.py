import argparse
import json

import numpy as np

from approx_sw import approx_sw
from direct_sw import direct_sw
from market import Market
from mode import mode
from naive import naive
from reciprocal import reciprocal
from tu import tu


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main(
    n: int = 10,
    exam_type: str = "inv",
    pref_lambda: float = 0.8,
    seed: int | None = None,
    verbose: bool = True,
    result_file_path: str | None = None,
) -> None:
    result: dict = {}

    if verbose:
        print(
            f"Experiment with n={n}, exam_type={exam_type}, pref_lambda={pref_lambda}, seed={seed}"
        )
    market = Market(
        num_cand=int(1.5 * n),
        num_job=n,
        exam_cand_type=exam_type,
        exam_job_type=exam_type,
        pref_seed=seed,
        pref_lambda=pref_lambda,
    )

    if verbose:
        print("====== Naive ======")
    result["naive"] = naive(market, verbose=verbose)

    if verbose:
        print("====== Reciprocal ======")
    result["reciprocal"] = reciprocal(market, verbose=verbose)

    if verbose:
        print("====== TU ======")
    result["tu"] = tu(market, beta=1.0, maxit=10000, tol=1e-9, verbose=verbose)

    if verbose:
        print("====== ApproxSW ======")
    result["approx_sw"] = approx_sw(
        market, lr=0.2, maxit=100, tol=1e-3, verbose=verbose, cpsolver="SCS"
    )

    if verbose:
        print("====== DirectSW ======")
    result["direct_sw"] = direct_sw(
        market, lr=0.2, maxit=100, tol=1e-3, verbose=verbose, cpsolver="SCS"
    )

    if verbose:
        print("====== MODE ======")
    result["mode"] = mode(market, max_iter=1000, initial_rec=None, verbose=verbose)

    if result_file_path is None:
        result_file_path = (
            f"result_n{n}_exam{exam_type}_lambda{pref_lambda}_seed{seed}.json"
        )

    with open(result_file_path, "w") as f:
        json.dump(result, f, cls=NumpyEncoder, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n",
        "-n",
        type=int,
        default=10,
        help="Number of jobs (and approximately 1.5*n candidates)",
    )
    parser.add_argument(
        "--exam_type",
        "-e",
        type=str,
        default="inv",
        choices=["log", "inv", "exp"],
        help="Type of examination probability function",
    )
    parser.add_argument(
        "--pref_lambda",
        "-l",
        type=float,
        default=0.8,
        help="Lambda parameter for preference generation",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=0,
        help="Random seed for preference generation",
    )
    parser.add_argument(
        "--nonverbose", "-nv", action="store_false", help="Disable verbose output"
    )
    parser.add_argument(
        "--result_file_path",
        "-r",
        type=str,
        default=None,
        help="Path to save the result JSON file",
    )
    args = parser.parse_args()
    main(
        n=args.n,
        exam_type=args.exam_type,
        pref_lambda=args.pref_lambda,
        seed=args.seed,
        verbose=args.nonverbose,
        result_file_path=args.result_file_path,
    )
