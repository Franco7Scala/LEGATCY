import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse arguments for experiments.")
    parser.add_argument("--debug", default="True", help="Enable debug mode.")
    parser.add_argument("--dataset-name", type=str, default="imdb", help="Name of the dataset to be used.")
    parser.add_argument("--result-directory", type=str, default=None, help="Name of the directory for the results.")
    parser.add_argument("--n-snapshot", type=int, default=4, help="Number of snapshots.")
    parser.add_argument("--times-first-snapshot", dest="times_first_snapshot", type=int, default=3, help="Multiplier for the first snapshot.")
    parser.add_argument("--metapaths-enabled", dest="metapaths_enabled", default="False", help="Enable metapaths.")
    parser.add_argument("--subgraph-hops", type=int, default=2, help="Number of hops for subgraph extraction.")
    parser.add_argument(
        "--training-strategy",
        type=str,
        choices=["DyHANE", "ActiveERS2", "FullRetraining", "OnlineTraining", "VotingStrategy"],
        default="FullRetraining",
        help="Training strategy name."
    )
    parser.add_argument(
        "--sampling-technique",
        type=str,
        choices=["RandomALTechnique", "LCSALTechnique", "EntropyALTechnique", "MarginALTechnique", "None"],
        default="LCSALTechnique",
        help="Sampling technique name or 'None'."
    )
    parser.add_argument("--k", type=int, default=500, help="Number of samples to select at each training stage (k).")
    parser.add_argument("--reduction-factor", type=float, default=0.5, help="Reduction factor for sampling.")
    parser.add_argument("--n-epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--max-lr", type=float, default=0.01, help="Maximum learning rate.")
    parser.add_argument("--min-lr", type=float, default=0.001, help="Minimum learning rate.")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of model layers.")
    parser.add_argument("--hidden-channels", type=int, default=64, help="Hidden channel size.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability.")
    return parser.parse_args()
