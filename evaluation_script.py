import json
import logging
import os
import re
import argparse
from typing import Optional
import glob
from scenario import Scenario  # Ensure this import points to your custom 'Scenario' class

logging.basicConfig(level=logging.INFO)


def extract_probability(final_verdict: str) -> Optional[float]:
    """
    Extract the probability from the final_verdict text.
    This function assumes the probability is in percentage form (e.g., "85%") and converts it to decimal form (e.g., 0.85).
    """
    try:
        probability_match = re.search(r"(\d+(\.\d+)?)%\s*", final_verdict)
        if probability_match:
            percentage = float(probability_match.group(1))
            return percentage / 100
        return None
    except Exception as e:
        logging.error(f"Error extracting probability: {e}")
        return None


def compare_probability_to_resolution(probability: float, resolution_probability: float) -> float:
    """
    Compare the extracted probability to the resolution_probability and return a score based on the comparison.
    A smaller difference means a higher score.
    """
    if probability is None:
        return 0.0

    difference = abs(probability - resolution_probability)

    return difference


def evaluate_verdict(final_verdict: str, resolution_probability: float) -> dict:
    """
    Evaluate the final verdict based on its probability and compare it to the resolution_probability.
    Returns a dictionary with the evaluation result.
    """
    probability = extract_probability(final_verdict)
    mae = compare_probability_to_resolution(probability, resolution_probability)

    return {
        "final_verdict": final_verdict,
        "probability": probability,
        "resolution_probability": resolution_probability,
        "mae": mae
    }


def calculate_average_mae(scenarios_path: str) -> float:
    """
    Go through each scenario file, calculate the MAE, and compute the average MAE.
    """
    total_mae = 0.0
    num_scenarios = 0

    scenario_files = glob.glob(os.path.join(scenarios_path, "*.json"))

    for scenario_file in scenario_files:
        try:
            # Load the scenario data
            with open(scenario_file, "r") as f:
                this_scenario = json.load(f)
            probability = this_scenario.get('probability', 0.0)

            # Create a Scenario object and check for the final verdict
            curr_scenario = Scenario(scenario_file)  # Ensure this is a valid class
            final_verdict = this_scenario.get('final_verdict', None)

            if not final_verdict:
                logging.warning(f"Missing final verdict in scenario: {scenario_file}")
                continue

            # Evaluate the verdict
            evaluation = evaluate_verdict(final_verdict, probability)
            mae = evaluation.get('mae', 0.0)
            total_mae += mae
            num_scenarios += 1

        except Exception as e:
            logging.error(f"Error processing scenario file {scenario_file}: {e}")

    if num_scenarios == 0:
        return 0.0
    return total_mae / num_scenarios


def main(scenarios_path: str) -> None:
    """
    Main function to evaluate the MAE across all scenarios in the specified path.
    """
    avg_mae = calculate_average_mae(scenarios_path)
    logging.info(f"Average MAE across all scenarios: {avg_mae}")
    print(f"Average MAE across all scenarios: {avg_mae}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MAE for all scenarios in the provided directory.")
    parser.add_argument('scenarios_path', type=str, help="Path to the directory containing scenario JSON files.")

    args = parser.parse_args()

    main(args.scenarios_path)
