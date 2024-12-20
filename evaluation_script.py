import os
import json
import re
import numpy as np
from scenario import Scenario
import argparse
import pydantic
from manifold_api import FullMarket
import traceback

def extract_probabilities(final_verdict: str):
    """Extracts ground truth probability and returns the last predicted probability."""
    matches = re.findall(r'(\d+)%', final_verdict)
    
    if matches:
        last_match = float(matches[-1]) / 100.0 
    else:
        last_match = None
    
    return last_match


def calculate_mae(ground_truth, predicted_prob):
    """Calculates the Mean Absolute Error (MAE)."""
    if ground_truth is None or predicted_prob is None:
        return None
    return abs(float(ground_truth) - float(predicted_prob))

def evaluate_scenarios(directory: str, ground_truths: dict[str, FullMarket], log_file: str):
    """Evaluates the MAE of scenarios in the given directory and logs results."""
    results = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    scenario = Scenario(file_path)
                    if scenario.config.id not in ground_truths.keys():
                        continue

                    final_verdict = scenario.run()
                    print("Final verdict: ", final_verdict)

                    ground_truth = ground_truths[scenario.config.id]
                    print("ground truth: ", ground_truth)

                    predicted_prob = extract_probabilities(final_verdict["Judge"])
                    print("predicted prob: ", final_verdict)
                    
                    mae = calculate_mae(ground_truth.p, predicted_prob)

                    if predicted_prob > 0.5 and ground_truth.resolution.lower() == "yes":
                        accuracy = 1
                    else:
                        accuracy = 0

                    print(f"Result for {ground_truth.question}: correct = {accuracy} mae = {mae}")
                    results.append({
                        "id": ground_truth.id,
                        "accuracy": accuracy,
                        "mae": mae,
                        "predicted_prob": predicted_prob,
                        "resolution": ground_truth.resolution,
                        "scenario_store": scenario.store
                    })


                    with open(log_file, 'w') as log:
                        json.dump(results, log, indent=4)
                except Exception:
                    print(f"Error processing file {file_path}: \n{traceback.format_exc()}")
                    breakpoint()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="config/manifold")
    parser.add_argument("--ground_truth", type=str, default="config/manifold/ground_truths.json")
    args = parser.parse_args()

    ta = pydantic.TypeAdapter(list[FullMarket])
    with open(os.path.join(args.dataset, "ground_truths.json"), 'r') as gt_file:
        ground_truth_list = ta.validate_json(gt_file.read())[-20:]

    # Pivot the ground truth data to be a dictionary of ids
    ground_truths = {market.id: market for market in ground_truth_list}

    directory = os.path.join(args.dataset, "scenarios")
    log_file = os.path.join(args.dataset, "results.json")

    evaluate_scenarios(directory, ground_truths, log_file)
