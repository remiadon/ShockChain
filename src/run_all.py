"""
Single entry point. Loads data, trains everything, prints/saves the report.

Usage:
    cd src && python run_all.py
"""

from dataset import load_data
from report import generate_report
from train import parse_args, set_seed, train_all_models


def main():
    args = parse_args()
    set_seed(args.seed)
    data = load_data(batch_size=args.batch_size)
    results = train_all_models(data, args=args, device="cpu")
    generate_report(results, indicators=data["indicators"])


if __name__ == "__main__":
    main()
