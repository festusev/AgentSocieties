import json

from datasets import Dataset
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    args = parser.parse_args()

    scenarios = []
    for fname in os.listdir(args.folder):
        if not fname.endswith("json"):
            continue

        with open(os.path.join(args.folder, fname), "r") as f:
            scenarios.append(json.load(f))

    # breakpoint()
    ds = Dataset.from_dict({key: [d[key] for d in scenarios] for key in scenarios[0]})

    ds.push_to_hub("evanellis/manifold")
