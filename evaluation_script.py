import os
import json
import re
import numpy as np
from scenario import Scenario

def load_scenario(file_path: str):
    """Loads a scenario JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def extract_probabilities(scenario_data: dict):
    """Extracts ground truth probability and predicted probability."""
    final_verdict = scenario_data.get("final_verdict", "")
    
    match = re.search(r'(\d+)%', final_verdict)
    predicted_prob = float(match.group(1)) / 100.0 if match else None
    
    return predicted_prob

def calculate_mae(ground_truth, predicted_prob):
    """Calculates the Mean Absolute Error (MAE)."""
    if ground_truth is None or predicted_prob is None:
        return None
    return abs(ground_truth - predicted_prob)

def evaluate_scenarios(directory: str, ground_truths: dict, log_file: str):
    """Evaluates the MAE of scenarios in the given directory and logs results."""
    mae_list = []
    individual_maes = {}  
    num_scenarios = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    scenario_data = load_scenario(file_path)
                    scenario = Scenario(file_path)
                    final_verdict = scenario.run()
                    print("Final verdict: ", final_verdict)

                    question = scenario_data.get("root_question", "")

                    ground_truth = ground_truths.get(question, None)

                    predicted_prob = final_verdict
                    
                    mae = calculate_mae(ground_truth, predicted_prob)
                    
                    if mae is not None:
                        mae_list.append(mae)
                        individual_maes[question] = mae  
                        num_scenarios += 1
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    
    if num_scenarios > 0:
        average_mae = np.mean(mae_list)
        print(f"Average MAE: {average_mae:.4f}")
    else:
        average_mae = None
        print("No valid scenarios found for evaluation.")
    
    print("\nIndividual MAEs:")
    for question, mae in individual_maes.items():
        print(f"{question}: {mae:.4f}")
    
    log_data = {
        "individual_maes": individual_maes,
        "average_mae": average_mae
    }
    
    with open(log_file, 'w') as log:
        json.dump(log_data, log, indent=4)

if __name__ == "__main__":
    gt_file_path = "config/ground_truths.json" 
    with open(gt_file_path, 'r') as gt_file:
        ground_truths = json.load(gt_file)

    directory = "config/manifold"  
    log_file = "config/mae_log.json"  

    evaluate_scenarios(directory, ground_truths, log_file)
