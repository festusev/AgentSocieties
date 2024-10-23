from googlesearch import search
import requests
from bs4 import BeautifulSoup
import autogen
import json
import os

# we're using gpt-4o-mini to save money
config_list = [
    {
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY")
    }
]


# Create the head agent as a ConversableAgent
head_agent = autogen.ConversableAgent(
    name="Head_Agent",
    llm_config={"config_list": config_list},
    system_message="You are the head agent responsible for gathering information and coordinating other agents."
)

# Create a user agent as a ConversableAgent
user_agent = autogen.ConversableAgent(
    name="User_Agent",
    human_input_mode="NEVER",
    llm_config=False,  # This agent doesn't use an LLM
    system_message="You are a user agent that initiates tasks and receives responses."
)

# Function to fetch and parse news articles
def fetch_news_articles(query, num_results=5):
    articles = []
    for url in search(query, num_results=num_results):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text from paragraphs
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        articles.append({
            'url': url,
            'content': text
        })
    if len(articles) != num_results: # you might need to re-run again or use smaller number of articles if the extraction isn't working.
        raise Exception(f"Error fetching articles. Expected {num_results} articles, but got {len(articles)}")
    return articles

# Root forecasting question
root_question = "Will Eric Adams be indicted in 2024?"

# T = 0: Generate search query using LLM
search_query_prompt = f"""Given the forecasting question '{root_question}', generate a search query to Google that will help gather relevant news articles and information. 
The query should focus on current evidence and developments related to the question. Only output the search query and NOTHING ELSE."""

# Get search query from head agent
user_agent.initiate_chat(head_agent, message=search_query_prompt, silent=False, max_turns=1)
search_query = head_agent.last_message()["content"]

# Fetch articles using generated query. Sometimes this fails; if so, reduce the number of articles we're querying.
articles = fetch_news_articles(search_query)

# T = 1: Rank articles and select top using LLM
def rank_articles(articles, num_articles=3):
    # Use head agent to rank each article's relevance
    ranked_articles = []
    for article in articles:
        ranking_prompt = f"""On a scale of 1-10, rate how relevant this article is to the search query '{search_query}'. 
        Only respond with a number 1-10.
        
        Article content:
        {article['content']}"""
        
        user_agent.initiate_chat(head_agent, message=ranking_prompt, silent=False, max_turns=1)
        try:
            score = int(head_agent.last_message()["content"].strip())
        except:
            score = 0
            
        ranked_articles.append({
            'article': article,
            'score': score
        })
    
    # Sort by score and return top original article objects
    return [x['article'] for x in sorted(ranked_articles, key=lambda x: x['score'], reverse=True)[:num_articles]]

top_articles = rank_articles(articles)

# T = 2: Generate summary prompt using LLM
summary_prompt_template = f"""Given the forecasting question '{root_question}', create a prompt that will help analyze and summarize the key information from the following passages:"""

# Combine generated prompt with articles
articles_str = "\n".join([f"{i+1}. {article['content']}\n-----" for i, article in enumerate(top_articles)])
summary_prompt = f"""{summary_prompt_template}

Articles:
{articles_str}
"""

# Initiate a chat between user_agent and head_agent for final summary
user_agent.initiate_chat(head_agent, message=summary_prompt, silent=False, max_turns=1)

# Extract the summary from the last message in the conversation
summary = head_agent.last_message()["content"]
print("Case Summary:")
print(summary)

# T = 3: Initialize judge, jurors, prosecutor and defense attorney
judge = autogen.ConversableAgent(
    name="Judge",
    system_message="You are a fair and impartial judge overseeing a forecasting trial. Your role is to ensure proper procedure and maintain order.",
    llm_config={"config_list": config_list}
)

juror1 = autogen.ConversableAgent(
    name="Juror1", 
    system_message="You are an intelligent and analytical juror. Your role is to carefully evaluate evidence and arguments to make probability forecasts.",
    llm_config={"config_list": config_list}
)

juror2 = autogen.ConversableAgent(
    name="Juror2",
    system_message="You are an intelligent and analytical juror. Your role is to carefully evaluate evidence and arguments to make probability forecasts.", 
    llm_config={"config_list": config_list}
)

prosecutor = autogen.ConversableAgent(
    name="Prosecutor",
    system_message="You are a prosecutor who argues for higher probability forecasts based on the evidence.",
    llm_config={"config_list": config_list}
)

defense = autogen.ConversableAgent(
    name="Defense",
    system_message="You are a defense attorney who argues for lower probability forecasts based on the evidence.",
    llm_config={"config_list": config_list}
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