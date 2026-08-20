# MODE: Mutual Optimality in Direct Effects of Reciprocal Recommendations in Matching Markets

This repository contains the implementation code for experiments with synthetic data in the paper "[MODE: Mutual Optimality in Direct Effects of Reciprocal Recommendations in Matching Markets](https://arxiv.org/abs/2608.01731)" by Yoji Tomita (CyberAgent, AI Lab), RecSys'26.


## Setup

If you can use `uv`, you can set up the environment with:
```shell
uv sync
```
Otherwise, you need following packages:
- Python 3.12
- Numpy 2.4.4
- CVXPY 1.8.2
- PyTorch 2.11.0


## Usage
You can run the main experiment script with `uv`:
```shell
uv run main.py --n 10 --exam_type inv --pref_lambda 0.8 --seed 0
```
or with your own python environment:
```shell
python main.py --n 10 --exam_type inv --pref_lambda 0.8 --seed 0
```
The options are:
- `--n` or `-n`: Number of jobs. Number of candidates is set to 1.5*n. Default is 10.
- `--exam_type` or `-e`: Type of examination probability vectors. Choices are "log", "inv", and "exp". Default is "inv".
- `--pref_lambda` or `-l`: Lambda parameter for preference generation. Default is 0.8.
- `--seed` or `-s`: Random seed for preference generation. Default is 0.
- `--nonverbose` or `-nv`: Disable verbose output. By default, verbose output is enabled.
- `--result_file_path` or `-r`: Path to save the result JSON file. If not provided, the result will be saved to a default file name based on the parameters.


## Files
- [`main.py`](main.py): Main script to run the experiment.
- [`market.py`](market.py): Contains the `Market` class which defines the matching market.
- [`naive.py`](naive.py): Contains the implementation of the Naive method.
- [`reciprocal.py`](reciprocal.py): Contains the implementation of the Reciprocal method.
- [`tu.py`](tu.py): Contains the implementation of the TU method.
- [`approx_sw.py`](approx_sw.py): Contains the implementation of the ApproxSW method.
- [`direct_sw.py`](direct_sw.py): Contains the implementation of the DirectSW method.
- [`mode.py`](mode.py): Contains the implementation of the MODE method.


## Citation

If you find our work useful in your research, please consider citing:
```bibtex
@inproceedings{tomita2026mode,
  title={MODE: Mutual Optimality in Direct Effects of Reciprocal Recommendations in Matching Markets},
  author={Yoji Tomita},
  booktitle={Proceedings of the 20th ACM Conference on Recommender Systems},
  year={2026}
}
```


## License

This repository is licensed under the MIT License.
