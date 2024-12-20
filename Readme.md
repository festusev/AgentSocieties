# Agent Societies
CS-194/294 Fa24 project

To install the requirements, use:
`pip install -r requirements.txt`

To run a single scenario, use:
`python run_scenario.py --config=path/to/config.json`

To create a dataset from Manifold Markets:
`python create_scenarios.py --limit=1000`

If you want to create a dataset of resolved markets, set the `--is_resolved` flag.

To evaluate from a local dataset:
`python evaluation_script.py --dataset=path/to/dataset`

This will output results into a results.json file.
