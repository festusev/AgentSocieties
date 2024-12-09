import manifold_api as mf
import argparse
import datetime

def create_benchmark(args: argparse.Namespace) -> :

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Creates scenarios from Manifold markets.")
    parser.add_argument("--limit", type=int, default=500, help="Max number of markets to include.")
    parser.add_argument("--start_date", type=lambda s: datetime.datetime.strptime(s, "%Y-%m-%d"), help="Earliest date to include.")
    parser.add_argument("--is_open", type=bool, default=True, help="Only include open markets if true, otherwise, only include closed markets.")
    args = parser.parse_args()

    benchmark = create_benchmark(args)
