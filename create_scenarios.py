import manifold_api as mf
import argparse
import datetime
from scenario import ScenarioConfig, StepConfig, AgentConfig
import re
import os
import json

DEFAULT_AGENTS: dict[str, AgentConfig] = {
    "HeadAgent": AgentConfig(
            name="HeadAgent",
            system_message="You are the head agent responsible for gathering information and coordinating other agents.",
            capabilities={
                "web_retrieval": True,
                "llm": True
            }
    ),
    "UserAgent": AgentConfig(
            name="UserAgent",
            system_message="You are a user agent that initiates tasks and receives responses.",
            capabilities={
                "web_retrieval": False,
                "llm": False
            }
    ),
    "Judge": AgentConfig(
            name="Judge",
            system_message="You are a fair and impartial judge overseeing the trial.",
            capabilities={
                "web_retrieval": False,
                "llm": True
            }
    ),
    "Prosecutor": AgentConfig(
            name="Prosecutor",
            system_message="You argue for higher probability forecasts based on the evidence.",
            capabilities={
                "web_retrieval": False,
                "llm": True
            }
    ),
    "Defense": AgentConfig(
            name="Defense",
            system_message="You argue for lower probability forecasts based on the evidence.",
            capabilities={
                "web_retrieval": False,
                "llm": True
            }
    ),
    "Clerk": AgentConfig(
            name="Clerk",
            system_message="You are a court clerk responsible for jury selection. You ensure diverse representation while selecting jurors with relevant expertise for the case topic. Reply in JSON only, with no accompanying text. Use the second person to describe each juror.",
            capabilities={
                "web_retrieval": False,
                "llm": True
            },
            jury_selection={
                "requirements": [
                    "Ensure demographic diversity (age, ethnicity, geography, profession)",
                    "Include relevant expertise for the topic",
                    "Vary socioeconomic backgrounds",
                    "Mix urban/rural perspectives",
                    "Include different educational levels"
                ],
                "profile_fields": [
                    "Age and demographic details",
                    "Professional background",
                    "Relevant expertise/experience",
                    "Geographic location",
                    "Key perspective they bring"
                ],
                "prompt_template": "Generate {num_jurors} diverse jury profiles for a case about: {root_question}\n\nRequirements:\n{requirements}\n\nFor each juror, provide:\n{profile_fields}\n\nFormat as JSON with structure:\n{{\n    \"Juror1\": {{\n        \"name\": \"Juror1\",\n        \"profile\": \"Detailed profile in a few sentences in the second person...\"\n    }},\n    ...\n}}"
            }
    )
}

DEFAULT_STEPS: list[StepConfig] = [
    StepConfig(
        step="0",
        action_type="generate_jury",
        initiator="Clerk",
        receiver="UserAgent",
        num_jurors=11
    ),
    StepConfig(
        step="1",
        action_type="message",
        initiator="UserAgent",
        receiver="HeadAgent",
        message="Given the forecasting question '{root_question}', generate a search query to gather relevant news articles and information. Focus on current evidence and developments related to the question. Only output the search query and NOTHING ELSE.",
        store="search_query"
    ),
    StepConfig(
        step="2",
        action_type="web_search",
        agent="HeadAgent",
        query="{search_query}",
        num_results=20,
        store="articles"
    ),
    StepConfig(
        step="3",
        action_type="process_articles",
        agent="HeadAgent",
        criteria="relevance and objectivity",
        num_top_articles=6,
        store="top_articles"
    ),
    StepConfig(
        step="4",
        action_type="message",
        initiator="UserAgent",
        receiver="HeadAgent",
        message="Based on these articles, provide a concise summary of the key evidence regarding {root_question}. Focus on factual information and current developments, pay attention to the date of the article.\n\nArticles:\n{top_articles}",
        store="evidence_summary"
    ),
    StepConfig(
        step="5",
        action_type="message",
        initiator="UserAgent",
        receiver="Judge",
        message="We will be evaluating {root_question}. Here is the evidence summary: {evidence_summary}. Please establish the trial rules and procedures.",
        store="trial_rules"
    ),
    StepConfig(
        step="6",
        action_type="message",
        initiator="UserAgent",
        receiver="Prosecutor",
        message="Based on the evidence summary: {evidence_summary}, present your opening argument for why the probability should be high. Include specific evidence and reasoning.",
        store="prosecution_opening"
    ),
    StepConfig(
        step="7",
        action_type="message",
        initiator="UserAgent",
        receiver="Defense",
        message="Based on the evidence summary: {evidence_summary}, present your opening argument for why the probability should be low. Include specific evidence and reasoning.",
        store="defense_opening"
    ),
    StepConfig(
        step="8",
        action_type="message",
        initiator="UserAgent",
        receiver="Jurors",
        message="Consider the following: Evidence Summary: {evidence_summary}\nProsecution Opening: {prosecution_opening}\nDefense Opening: {defense_opening}\n\nBased on this information, what probability would you assign to the question: {root_question}? Provide your probability estimate (0-100%) and brief reasoning.",
        store="jury_forecasts"
    ),
    StepConfig(
        step="9",
        action_type="message",
        initiator="UserAgent",
        receiver="Judge",
        message="Based on the jury forecasts: {jury_forecasts}, please provide a final summary and the median probability forecast.",
        store="final_verdict"
    )
]

def create_scenarios(args: argparse.Namespace) -> list[ScenarioConfig]:
    unfiltered_lite_markets = mf.get_markets(
        limit=args.limit,
    )

    # Filter the markets to match the parameters
    lite_markets = []
    for market in unfiltered_lite_markets:
        if market.isResolved != args.is_resolved or market.outcomeType != "BINARY":
            continue
        lite_markets.append(market)

    print(f"Found {len(lite_markets)} matching binary markets")
    markets = [mf.get_full_market(market.id) for market in lite_markets]

    scenarios = []
    ground_truths = {}  # Dictionary to store probabilities
    for market in markets:
        question = market.question
        probability = market.probability  # Get the probability value for the market

        if probability is None:
            print(f"Skipping market {question} due to missing probability.")
            continue  # Skip scenario if probability is None

        scenario = ScenarioConfig(
                root_question=question,
                agents=DEFAULT_AGENTS,
                scenario=DEFAULT_STEPS
        )

        # Add the question-probability pair to the ground_truths dictionary
        ground_truths[question] = probability

        # Save scenario to disk
        fname = scenario.root_question.lower()
        fname = re.sub(r'[^a-z0-9_\s]', '', fname)
        fname = '_'.join(fname.split()[-4:])
        file_path = os.path.join(args.out, fname + ".json")

        # Save to the folder if the scenario is valid
        print(f"Saving scenario to {file_path}")
        os.makedirs(args.out, exist_ok=True)
        with open(file_path, "w") as fb:
            fb.write(scenario.model_dump_json())

    # Save the ground truth probabilities to a separate JSON file
    ground_truths_file = os.path.join(args.out, "ground_truths.json")
    print(f"Saving ground truths to {ground_truths_file}")
    with open(ground_truths_file, "w") as f:
        json.dump(ground_truths, f, indent=4)

    return scenarios

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Creates scenarios from Manifold markets.")
    parser.add_argument("--limit", type=int, default=500, help="Max number of markets to include.")
    parser.add_argument("--is_resolved", type=bool, default=False, help="Only include resolved markets if True, otherwise, only include unresolved markets.")
    parser.add_argument("--out", type=str, default="config/manifold/", help="Output directory.")
    args = parser.parse_args()

    scenarios = create_scenarios(args)