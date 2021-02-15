from ax.service.managed_loop import optimize
from train import train_evaluate
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='sgns', help="model name")
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--save_dir', type=str, default='./pts/', help="model directory path")
    parser.add_argument('--max_epoch', type=int, default=100, help="max number of epochs")
    parser.add_argument('--k', type=int, default=10, help="number of top ranked items")
    parser.add_argument('--conv_thresh', type=float, default=0.0001, help="threshold diff for convergence")
    parser.add_argument('--hrk_weight', type=float, default=0.5, help="weight to put on hrk metric value")
    parser.add_argument('--trials', type=int, default=10, help="number of trials ")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")

    return parser.parse_args()


def main():
    args = parse_args()
    best_parameters, values, experiment, cur_model = optimize(
        parameters=[
            {"name": "lr", "type": "range", "value_type": "float", "bounds": [1e-3, 0.4], "log_scale": True},
            {"name": "ss_t", "type": "range", "value_type": "float", "bounds": [1e-5, 3e-3], "log_scale": True},
            {"name": "e_dim", "type": "choice", "value_type": "int", "values": [80, 100, 150, 200, 220, 250]},
            {"name": "n_negs", "type": "choice", "value_type": "int", "values": [5, 7, 10, 15, 50]},
            {"name": "mini_batch", "type": "choice", "value_type": "int", "values": [5, 8, 16, 36]},
            {"name": "weights", "type": "choice", "value_type": "bool", "values": [True, True]},
            {"name": "max_epoch", "type": "fixed", "value_type": "int", "value": args.max_epoch},
            {"name": "k", "type": "fixed", "value_type": "int", "value": args.k},
            {"name": "conv_thresh", "type": "fixed", "value_type": "float", "value": args.conv_thresh},
            {"name": "hrk_weight", "type": "fixed", "value_type": "float", "value": args.hrk_weight},
            {"name": "cuda", "type": "fixed", "value": args.cuda},

        ],
        evaluation_function=train_evaluate,
        minimize=False,
        objective_name='0.5*hr_k + 0.5*mrr_k',
        total_trials=args.trials
    )


if __name__ == '__main__':
    main()