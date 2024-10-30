"""
Forecasting Trial System using LLM Agents

This script implements a mock trial system for making probability forecasts using LLM agents.
We use DuckDuckGo for web search since it doesn't require an API key, unlike Google.

The system follows a structured trial process:
1. Setup and Imports - Configure environment and import dependencies
2. Agent Creation (T=0) - Initialize head agent and user agent
3. Article Collection (T=1) - Gather relevant news articles via search
4. Article Analysis (T=2) - Generate search queries and rank articles
5. Trial Setup (T=3,4) - Create judge, jurors, attorneys and set rules
6. Evidence Selection (T=5,7) - Both sides select supporting evidence
7. Opening Statements (T=6,8) - Each side presents initial arguments
"""

# --- 1. IMPORTS AND CONFIG ---
from duckduckgo_search import DDGS  # DuckDuckGo search API - no key needed
import requests
from bs4 import BeautifulSoup
import autogen
import json
import os

# Using gpt-4o-mini to optimize for cost
config_list = [{"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}]


# --- 2. AGENT CREATION (T=0) ---
# Head agent coordinates other agents and manages information flow
head_agent = autogen.ConversableAgent(
    name="Head_Agent",
    llm_config={"config_list": config_list},
    system_message="You are the head agent responsible for gathering information and coordinating other agents.",
)

# User agent initiates tasks but doesn't use LLM directly
user_agent = autogen.ConversableAgent(
    name="User_Agent",
    human_input_mode="NEVER",
    llm_config=False,  # No LLM needed for this agent
    system_message="You are a user agent that initiates tasks and receives responses.",
)


# --- 3. ARTICLE COLLECTION (T=1) ---
def fetch_news_articles(query, num_results=20):
    """
    Fetch and validate news articles from DuckDuckGo search.
    
    Args:
        query (str): Search query to find relevant articles
        num_results (int): Number of articles to fetch (default: 20)
        
    Returns:
        list: List of dictionaries containing article data with keys:
            - url (str): Article URL
            - content (str): Article text content
            
    Raises:
        AssertionError: If expected number of results not found
        Exception: If article extraction fails
    """
    articles = []
    search_results = DDGS().text(query, max_results=num_results)
    
    # Validate we got expected number of results
    assert (
        len(search_results) == num_results
    ), f"Expected {num_results} articles, but got {len(search_results)}. Might need to reduce number of articles."
    
    # Extract content from results
    for result in search_results:
        articles.append({"url": result["href"], "content": result["body"]})
        
    # Final validation check
    if len(articles) != num_results:
        raise Exception(
            f"Error fetching articles. Expected {num_results} articles, but got {len(articles)}"
        )
    return articles


# --- 4. ARTICLE ANALYSIS (T=2) ---
# Root question we're trying to forecast
root_question = "Will an asteroid larger than 50m diameter hit the earth in 2024?"

# T = 0: Generate search query using LLM
search_query_prompt = f"""Given the forecasting question '{root_question}', generate a search query to Google that will help gather relevant news articles and information. 
The query should focus on current evidence and developments related to the question. Only output the search query and NOTHING ELSE"""

# Get and process search query
user_agent.initiate_chat(
    head_agent, message=search_query_prompt, silent=False, max_turns=1
)
search_query = head_agent.last_message()["content"]

# Normalize query: lowercase and remove punctuation
search_query = (
    search_query.lower()
    .replace(",", "")
    .replace(".", "")
    .replace("?", "")
    .replace("!", "")
    .replace("'", "")
    .replace('"', "")
)

# Fetch articles using processed query
article_bank = fetch_news_articles(search_query)


def rank_articles(articles, ranking_criteria, agent, num_articles=3):
    """
    Generic function to rank articles based on specified criteria using an LLM agent.
    
    Args:
        articles (list): List of articles to rank
        ranking_criteria (str): Prompt template for ranking criteria
        agent (ConversableAgent): Agent to use for ranking
        num_articles (int): Number of top articles to return (default: 3)
        
    Returns:
        list: Top ranked articles sorted by score
    """
    ranked_articles = []
    for article in articles:
        ranking_prompt = f"{ranking_criteria}\n\nArticle content:\n{article['content']}"

        user_agent.initiate_chat(
            agent, message=ranking_prompt, silent=False, max_turns=1
        )
        try:
            score1, score2 = map(
                int, agent.last_message()["content"].strip().split(",")
            )
            score = (score1 + score2) / 2
        except:
            score = 0

        ranked_articles.append({"article": article, "score": score})

    return [
        x["article"]
        for x in sorted(ranked_articles, key=lambda x: x["score"], reverse=True)[
            :num_articles
        ]
    ]


def rank_articles_unbiased(articles, num_articles=3):
    """
    Rank articles based on relevance and objectivity.
    
    Args:
        articles (list): List of articles to rank
        num_articles (int): Number of top articles to return
        
    Returns:
        list: Top ranked articles
    """
    criteria = f"""Rate this article on two criteria:
    1. Relevance to '{search_query}' (1-10)
    2. Objectivity and lack of bias (1-10)
    
    Only respond with two numbers separated by a comma (e.g. "8,7")"""
    
    return rank_articles(articles, criteria, head_agent, num_articles)


top_articles = rank_articles_unbiased(article_bank)

# T = 2: Generate summary prompt using LLM
summary_prompt_template = f"""Given the forecasting question '{root_question}', please create a summary of the key information from the following passages:"""

# Format articles for summary
articles_str = "\n".join(
    [f"{i+1}. {article['content']}\n-----" for i, article in enumerate(top_articles)]
)
summary_prompt = f"""{summary_prompt_template}

Articles:
{articles_str}
"""

# Get case summary
user_agent.initiate_chat(head_agent, message=summary_prompt, silent=False, max_turns=1)
summary = head_agent.last_message()["content"]
print("Case Summary:")
print(summary)


# --- 5. TRIAL SETUP (T=3,4) ---
# T = 3: Initialize judge, jurors, prosecutor and defense attorney
judge = autogen.ConversableAgent(
    name="Judge",
    system_message="You are a fair and impartial judge overseeing a forecasting trial. Your role is to ensure proper procedure and maintain order.",
    llm_config={"config_list": config_list},
)

juror1 = autogen.ConversableAgent(
    name="Juror1",
    system_message="You are a 45-year-old African American high school teacher from Atlanta, Georgia. As an educator with 20 years of experience, you bring a thoughtful and analytical perspective to discussions. You're active in your local church and community organizations.",
    llm_config={"config_list": config_list},
)

juror2 = autogen.ConversableAgent(
    name="Juror2", 
    system_message="You are a 32-year-old Hispanic construction foreman from Phoenix, Arizona. First-generation American, bilingual, and proud of your Mexican heritage. You're pragmatic and detail-oriented, having worked your way up from entry-level positions. You also have a PhD in astrophysics and are an expert on meteorite risk.",
    llm_config={"config_list": config_list},
)

juror3 = autogen.ConversableAgent(
    name="Juror3",
    system_message="You are a 68-year-old retired white factory worker from Detroit, Michigan. You've witnessed your city's economic ups and downs, making you particularly interested in political and economic issues. You're a grandfather of four and volunteer at the local veterans' center. You also have a PhD in astrophysics and are an expert on meteorite risk.",
    llm_config={"config_list": config_list},
)

juror4 = autogen.ConversableAgent(
    name="Juror4",
    system_message="You are a 29-year-old Asian American software developer from Seattle, Washington. Child of Korean immigrants, you're tech-savvy and data-driven in your decision-making. You're passionate about environmental issues and local politics. You also have a PhD in astrophysics and are an expert on meteorite risk.",
    llm_config={"config_list": config_list},
)

juror5 = autogen.ConversableAgent(
    name="Juror5",
    system_message="You are a 51-year-old white small business owner from Des Moines, Iowa. Running a family-owned hardware store has given you strong views on economic policies and regulations. You're active in your local Chamber of Commerce. You also have a PhD in astrophysics and are an expert on meteorite risk.",
    llm_config={"config_list": config_list},
)

juror6 = autogen.ConversableAgent(
    name="Juror6",
    system_message="You are a 38-year-old Native American nurse from Albuquerque, New Mexico. Your experience in healthcare has shaped your views on social policies. You're a member of the Navajo Nation and advocate for indigenous rights. You also have a PhD in astrophysics and are an expert on meteorite risk.",
    llm_config={"config_list": config_list},
)

juror7 = autogen.ConversableAgent(
    name="Juror7",
    system_message="You are a 42-year-old white farmer from rural Nebraska. A fourth-generation farmer, you're concerned about agricultural policies and rural development. You have a degree in agricultural science and are tech-savvy in modern farming practices. You also have a PhD in astrophysics and are an expert on meteorite risk.",
    llm_config={"config_list": config_list},
)

juror8 = autogen.ConversableAgent(
    name="Juror8",
    system_message="You are a 35-year-old African American military veteran from Virginia Beach. Your service in the Navy has given you a global perspective. Now working in cybersecurity, you're detail-oriented and security-minded. You also have a PhD in astrophysics and are an expert on meteorite risk.",
    llm_config={"config_list": config_list},
)

juror9 = autogen.ConversableAgent(
    name="Juror9",
    system_message="You are a 55-year-old white suburban mom from Minneapolis, Minnesota. With a background in accounting and active involvement in local school boards, you bring both financial acumen and community perspective to discussions. You also have a PhD in astrophysics and are an expert on meteorite risk.",
    llm_config={"config_list": config_list},
)

juror10 = autogen.ConversableAgent(
    name="Juror10",
    system_message="You are a 26-year-old second-generation Indian American journalist from Houston, Texas. Your work in local news has given you deep insight into community issues. You're data-oriented but also value human interest stories. You also have a PhD in astrophysics and are an expert on meteorite risk.",
    llm_config={"config_list": config_list},
)

jurors = [juror1, juror2, juror3, juror4, juror5, juror6, juror7, juror8, juror9, juror10]

prosecutor = autogen.ConversableAgent(
    name="Prosecutor",
    system_message="You are a prosecutor who argues for higher probability forecasts based on the evidence.",
    llm_config={"config_list": config_list},
)

defense = autogen.ConversableAgent(
    name="Defense",
    system_message="You are a defense attorney who argues for lower probability forecasts based on the evidence.",
    llm_config={"config_list": config_list},
)

# T = 4: Judge sets trial rules
trial_rules = f"""We will establish the following trial rules:
1. The prosecution will present their forecast and supporting arguments first
2. The defense will present their forecast and supporting arguments second 
3. Each juror will then independently submit their probability forecast
4. The final forecast will be the average of the two juror forecasts

The question being forecast is: {root_question}
The evidence summary is: {summary}"""

user_agent.initiate_chat(judge, message=trial_rules, silent=False, max_turns=1)


def rank_prosecution_articles(articles, num_articles=3):
    """
    Rank articles based on relevance and support for prosecution's case.
    
    Args:
        articles (list): List of articles to rank
        num_articles (int): Number of top articles to return
        
    Returns:
        list: Top ranked articles supporting prosecution
    """
    criteria = f"""Rate this article on two criteria regarding the question: '{root_question}'

    1. Relevance to the forecasting question (1-10)
    2. How strongly it supports a higher probability forecast (1-10)
    
    Only respond with two numbers separated by a comma (e.g. "8,7")"""
    
    return rank_articles(articles, criteria, prosecutor, num_articles)


top_prosecution_articles = rank_prosecution_articles(article_bank)

# T = 6: Prosecutor prepares opening statement
opening_statement_prompt = f"""Based on the following evidence, prepare a compelling opening statement arguing for a high probability forecast for the question '{root_question}'.
Focus on the strongest evidence and most convincing arguments. The statement should be clear, persuasive, and well-structured.

Evidence:
{chr(10).join([f"{i+1}. {article['content']}" for i, article in enumerate(top_prosecution_articles)])}"""

user_agent.initiate_chat(
    prosecutor, message=opening_statement_prompt, silent=False, max_turns=1
)
prosecution_opening = prosecutor.last_message()["content"]
print("\nProsecution Opening Statement:")
print(prosecution_opening)


def rank_defense_articles(articles, num_articles=3):
    """
    Rank articles based on relevance and support for defense's case.
    
    Args:
        articles (list): List of articles to rank
        num_articles (int): Number of top articles to return
        
    Returns:
        list: Top ranked articles supporting defense
    """
    criteria = f"""Rate this article on two criteria regarding the question: '{root_question}'

    1. Relevance to the forecasting question (1-10)
    2. How strongly it supports a lower probability forecast (1-10)
    
    Only respond with two numbers separated by a comma (e.g. "8,7")"""
    
    return rank_articles(articles, criteria, defense, num_articles)


top_defense_articles = rank_defense_articles(article_bank)

# T = 8: Defense prepares opening statement
opening_statement_prompt = f"""Based on the following evidence, prepare a compelling opening statement arguing for a low probability forecast for the question '{root_question}'.
Focus on the strongest evidence and most convincing arguments. The statement should be clear, persuasive, and well-structured.

Evidence:
{chr(10).join([f"{i+1}. {article['content']}" for i, article in enumerate(top_defense_articles)])}"""

user_agent.initiate_chat(
    defense, message=opening_statement_prompt, silent=False, max_turns=1
)
defense_opening = defense.last_message()["content"]
print("\nDefense Opening Statement:")
print(defense_opening)

# --- 6. JURY DELIBERATION AND VOTING ---
def get_juror_forecast(juror, evidence_summary, prosecution_case, defense_case):
    """
    Have a juror evaluate evidence and submit their probability forecast.
    
    Args:
        juror (ConversableAgent): The juror agent
        evidence_summary (str): Summary of all evidence
        prosecution_case (str): Prosecution's opening statement
        defense_case (str): Defense's opening statement
        
    Returns:
        float: Probability forecast from 0-100
    """
    deliberation_prompt = f"""Based on the following information, provide your probability forecast (0-100%) for the question: '{root_question}'

Evidence Summary:
{evidence_summary}

Prosecution's Case:
{prosecution_case}

Defense's Case:
{defense_case}

Please carefully consider all evidence and arguments presented. Respond ONLY with a number between 0 and 100."""

    user_agent.initiate_chat(juror, message=deliberation_prompt, silent=False, max_turns=1)
    
    try:
        # Extract number from response and validate
        forecast = float(juror.last_message()["content"].strip().replace('%', ''))
        if not 0 <= forecast <= 100:
            raise ValueError
        return forecast
    except:
        print(f"Warning: Invalid forecast from {juror.name}, defaulting to 50")
        return 50.0

# Get forecasts from both jurors
juror_forecasts = [get_juror_forecast(juror, summary, prosecution_opening, defense_opening) for juror in jurors]

# Calculate final forecast
final_forecast = sum(juror_forecasts) / len(juror_forecasts)

# Have judge announce the verdict
verdict_announcement = f"""The jury has reached a decision.

{chr(10).join(f'Juror {i+1} Forecast: {forecast}%' for i, forecast in enumerate(juror_forecasts))}

The final probability forecast for the question '{root_question}' is: {final_forecast}%"""

user_agent.initiate_chat(judge, message=verdict_announcement, silent=False, max_turns=1)
