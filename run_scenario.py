from scenario import Scenario
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a forecasting scenario.')
    parser.add_argument('--config', type=str, default="config/port_strike.json", help='The scenario config to use')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    scenario = Scenario(config_path=args.config)
    scenario.run()
