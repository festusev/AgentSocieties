import json
import os
import re
from duckduckgo_search import DDGS
import autogen
from datetime import datetime
import logging
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import tiktoken
from typing import List, Dict
from pydantic import BaseModel

class Scenario:
    """
    A modular Scenario class that executes steps based on a JSON configuration.

    This class initializes agents and performs actions such as messaging and
    web searches as specified in the config.
    """
    def __init__(self, config_path):
        """
        Initialize the Scenario with agents and configuration.

        Args:
            config_path (str): Path to the JSON config file.
        """
        # Load configuration from JSON file
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Setup logging
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create log filename from config path and timestamp
        config_name = Path(config_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"{config_name}_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Initialize the variable store
        self.store = {}

        # Store the root question
        self.root_question = self.config.get('root_question', '')
        self.config_list = [{"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}]

        # Initialize agents
        self._initialize_agents()

    def _get_default_jury_profiles(self, num_jurors):
        """
        Get default jury profiles for the case topic
        """
        jury_profiles = {}
        for i in range(num_jurors):
            profile = {
                "name": f"Juror{i+1}",
                "system_message": f"You are a juror tasked with evaluating evidence and making probability forecasts.",
                "capabilities": {
                    "web_retrieval": False,
                    "llm": True
                }
            }
            jury_profiles[f"Juror{i+1}"] = profile
            self.jurors.append(f"Juror{i+1}")

        return jury_profiles

    def _generate_jurors(self, num_jurors):
        """Generate diverse jury profiles based on the case topic"""        
        clerk = self._get_agent('Clerk')
        user_proxy = self._get_agent('UserAgent')
        
        clerk_config = self.config['agents']['Clerk']['jury_selection']
        # Prompt for the clerk to generate profiles
        prompt = clerk_config['prompt_template'].format(
            num_jurors=num_jurors,
            root_question=self.root_question,
            requirements="\n".join(f"{i+1}. {r}" for i, r in enumerate(clerk_config['requirements'])),
            profile_fields="\n".join(f"- {f}" for f in clerk_config['profile_fields'])
        )

        user_proxy.initiate_chat(clerk, message=prompt, silent=False, max_turns=1)
        try:
            message_content = clerk.last_message()["content"]
            if "```json" in message_content:
                json_content = message_content.split("```json")[1].split("```")[0].strip()
                profiles = json.loads(json_content)
            else:
                profiles = json.loads(message_content)

            for profile in profiles:
                profiles[profile]["capabilities"] = {
                    "web_retrieval": False,
                    "llm": True
                }
                self.jurors.append(profile)
            return profiles
        except Exception as e:
            logging.error(f"Failed to parse jury profiles JSON: {e}")
            return self._get_default_jury_profiles(num_jurors)

    def _initialize_agents(self):
        """
        Initialize all agents as specified in the configuration.
        """        
        # Initialize agents
        self.agents = {}
        for agent_name, agent_config in self.config['agents'].items():
            agent = self._create_agent(agent_config)
            self.agents[agent_name] = agent

        # Automatically generate jury profiles if enabled
        # Otherwise, Jurors were already specified in the config
        self.jurors = [] # If "Jurors" is the recipient, these names are used
        if self.config.get('generate_jury', False):
            num_jurors = self.config.get('num_jurors', 3)
            jury_profiles = self._generate_jurors(num_jurors)

            for juror_name, juror_config in jury_profiles.items():
                juror = self._create_agent(juror_config)
                self.agents[juror_name] = juror

    def _create_agent(self, agent_config):
        """
        Create an agent based on the provided configuration.

        Args:
            agent_config (dict): Configuration for the agent.

        Returns:
            autogen.ConversableAgent: Initialized agent.
        """
        # Using gpt-4o-mini to optimize for cost
        llm_enabled = agent_config['capabilities'].get('llm', False)
        llm_config = {"config_list": self.config_list} if llm_enabled else False

        return autogen.ConversableAgent(
            name=agent_config.get('name', ''),
            system_message=agent_config.get('system_message', ''),
            llm_config=llm_config,
            human_input_mode="NEVER",
        )

    def _resolve_placeholders(self, text):
        """
        Resolve placeholders in the text using the variable store.

        Args:
            text (str): The text containing placeholders.

        Returns:
            str: The text with placeholders resolved.
        """
        def replacer(match):
            key = match.group(1)
            # First check if the key is in the store
            if key in self.store:
                return str(self.store[key])
            # Then check if it's an attribute of self
            elif hasattr(self, key):
                return str(getattr(self, key))
            else:
                return ''

        return re.sub(r'\{([^}]+)\}', replacer, text)

    def _get_agent(self, name):
        """
        Retrieve an agent or list of agents by name.

        Args:
            name (str or list): The name of the agent or list of agent names.

        Returns:
            autogen.ConversableAgent or list[autogen.ConversableAgent]: The agent object(s).
        """
        if isinstance(name, list):
            # If name is a list, return list of agents
            return [self.agents[agent_name] for agent_name in name]
        # Otherwise return single agent
        return self.agents[name]

    def _execute_action(self, step):
        """
        Execute a single action as specified in the scenario steps.

        Args:
            step (dict): Configuration for the step/action.
        """
        action_type = step['action_type']
        if action_type == 'message':
            self._action_message(step)
        elif action_type == 'web_search':
            self._action_web_search(step)
        elif action_type == 'process_articles':
            self._action_process_articles(step)
        else:
            print(f"Unknown action type: {action_type}")

    def _action_message(self, step):
        """
        Updated message action with logging
        """
        initiator_name = step['initiator']
        receiver_name = step['receiver']
        message_template = step['message']
        store_variable = step.get('store', None)
        max_turns = step.get('max_turns', 1)
        silent = step.get('silent', False)

        # Resolve placeholders in the message
        message = self._resolve_placeholders(message_template)

        # Get initiator and receiver agents
        initiator = self._get_agent(initiator_name)
        if receiver_name == "Jurors":
            receiver = self._get_agent(self.jurors)
        else:
            receiver = self._get_agent(receiver_name)

        # Log the prompt
        logging.info(f"PROMPT - From: {initiator_name}, To: {receiver_name}\n{message}\n")

        if isinstance(receiver, list):
            # Broadcasting message to multiple agents
            responses = {}
            for agent in receiver:
                initiator.initiate_chat(agent, message=message, silent=silent, max_turns=max_turns)
                response = agent.last_message()['content']
                responses[agent.name] = response
                logging.info(f"RESPONSE - From: {agent.name}\n{response}\n")
            
            if store_variable:
                self.store[store_variable] = responses
        else:
            # One-to-one message
            initiator.initiate_chat(receiver, message=message, silent=silent, max_turns=max_turns)
            response = receiver.last_message()['content']
            if store_variable:
                self.store[store_variable] = response
            logging.info(f"RESPONSE - From: {receiver_name}\n{response}\n")

    def _format_article(self, article):
        pass
        
    def _action_web_search(self, step):
        """
        Perform a web search action.
        """
        query_template = step['query']
        num_results = step.get('num_results', 10)
        store_variable = step.get('store', None)

        # Resolve placeholders in the query
        query = self._normalize_query(self._resolve_placeholders(query_template))
        logging.info(f"Normalized search query: {query}")
        
        # Perform search using DuckDuckGo API
        search_results = self._fetch_news_articles(query, num_results)
        
        # Store the articles if needed
        if store_variable:
            self.store[store_variable] = search_results
            logging.info(f"Stored {len(search_results)} articles in variable: {store_variable}")

    
    def _normalize_query(self, query):
        """
        Normalize the query by converting to lowercase and removing punctuation.
        """
        return query.lower().replace(",", "").replace(".", "").replace("?", "").replace("!", "").replace("'", "").replace('"', "")

    def _fetch_news_articles(self, query, num_results=20):
        """
        Fetch and validate news articles from DuckDuckGo search, including full content.
        """
        articles = []
        logging.info(f"Starting web search with query: {query}")
        logging.info(f"Requested number of results: {num_results}")
        
        search_results = DDGS().text(query, max_results=num_results)
        
        # Extract content from results
        for result in search_results:
            url = result["href"]
            try:
                # Fetch full article content
                full_content = self._fetch_article_content(url)
                if full_content:
                    articles.append({
                        "url": url,
                        "title": result.get("title", ""),
                        "snippet": result["body"],
                        "content": full_content
                    })
                    logging.info(f"Successfully fetched article from: {url}")
                else:
                    logging.warning(f"No content extracted from: {url}")
            except Exception as e:
                logging.error(f"Error fetching article from {url}: {str(e)}")
        
        # Process and summarize articles
        processed_articles = self._process_article_contents(articles)
        logging.info(f"Search completed. Retrieved and processed {len(processed_articles)} articles")
        return processed_articles

    def _fetch_article_content(self, url: str) -> str:
        """
        Fetch and extract main content from an article URL.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'iframe']):
                element.decompose()
            
            # Extract main content (adjust selectors based on common article structures)
            content = ""
            main_content = soup.find('article') or soup.find(class_=['article', 'post', 'content', 'main-content'])
            
            if main_content:
                content = main_content.get_text(separator=' ', strip=True)
            else:
                # Fallback to body content if no article container found
                content = soup.body.get_text(separator=' ', strip=True)
            
            return content
        except Exception as e:
            logging.error(f"Error fetching article content: {str(e)}")
            return ""

    def _process_article_contents(self, articles: List[Dict]) -> List[Dict]:
        """
        Process and summarize article contents using chunks, limited to 3 chunks per article.
        """
        encoder = tiktoken.encoding_for_model("gpt-4")
        chunk_size = 2000
        max_chunks = 3
        user_agent = self._get_agent('UserAgent')
        summarizer_agent = self._get_agent('HeadAgent')
        
        for article in articles:
            content = article['content']
            tokens = encoder.encode(content)
            
            if len(tokens) > chunk_size:
                # Split into chunks and limit to max_chunks
                chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)][:max_chunks]
                summaries = []
                
                logging.info(f"\nProcessing article from: {article.get('url', 'Unknown URL')}")
                logging.info(f"Total chunks to process: {len(chunks)}")
                
                for i, chunk in enumerate(chunks, 1):
                    chunk_text = encoder.decode(chunk)
                    prompt = f"Please provide a concise summary of this text chunk, focusing on the most important information:\n\n{chunk_text}"
                    
                    user_agent.initiate_chat(
                        summarizer_agent,
                        message=prompt,
                        silent=True, 
                        max_turns=1
                    )
                    
                    chunk_summary = summarizer_agent.last_message()['content']
                    summaries.append(chunk_summary)
                    logging.info(f"\nChunk {i} Summary:\n{chunk_summary}")
                
                # Combine chunk summaries
                final_summary_prompt = f"Please combine these {len(summaries)} summaries into a coherent article, focusing on the key points:\n\n{' '.join(summaries)}"
                user_agent.initiate_chat(
                    summarizer_agent,
                    message=final_summary_prompt,
                    silent=True, 
                    max_turns=1
                )
                
                final_summary = summarizer_agent.last_message()['content']
                article['content'] = final_summary
                logging.info(f"\nFinal Combined Summary:\n{final_summary}\n")
                logging.info("=" * 80 + "\n")  # Separator between articles
            else:
                logging.info(f"\nArticle from {article.get('url', 'Unknown URL')} was short enough to use without chunking\n")
                logging.info("=" * 80 + "\n")
        
        return articles

    def _action_process_articles(self, step):
        """
        Process and rank articles based on specified criteria.

        Args:
            step (dict): Configuration for the process_articles action containing:
                - agent: Name of the agent to process articles
                - articles: List of articles to process
                - criteria: Ranking criteria
                - num_top_articles: Number of top articles to keep
                - store: Variable name to store results
        """
        # Get parameters from step config
        agent_name = step['agent']
        articles = self.store.get('articles', [])
        criteria = step.get('criteria', 'relevance and objectivity')
        num_top_articles = step.get('num_top_articles', 6)
        store_variable = step.get('store', None)

        # Get the agent
        agent = self._get_agent(agent_name)
        

        # Rank each article
        ranked_articles = []
        for article in articles:
            ranking_prompt = f"""Rate this article on two criteria:
1. Relevance to the question '{self.root_question}' (1-10)
2. {criteria} (1-10)

Only respond with two numbers separated by a comma (e.g. "8,7")

Article content:
{article['content']}"""

            # Get ranking from agent
            user_agent = self._get_agent('UserAgent')

            user_agent.initiate_chat(agent, message=ranking_prompt, max_turns=1)

            
            try:
                # Parse scores from response
                score1, score2 = map(int, agent.last_message()['content'].strip().split(','))
                score = (score1 + score2) / 2
            except:
                score = 0
            
            ranked_articles.append({'article': article, 'score': score})

        # Sort and get top articles
        # import pdb; pdb.set_trace()
        top_articles = sorted(ranked_articles, key=lambda x: x['score'], reverse=True)[:num_top_articles]
        top_article_contents = [article['article']['content'] for article in top_articles]

        # Store results if needed
        if store_variable:
            self.store[store_variable] = top_article_contents

    def run(self):
        """
        Execute the scenario steps as defined in the configuration.
        """
        logging.info(f"Starting scenario: {self.root_question}\n")
        steps = self.config.get('scenario', [])
        for step in steps:
            logging.info(f"Executing step {step['step']}: {step['action_type']}")
            self._execute_action(step)

        final_verdict = self.store.get('final_verdict', '')
        logging.info(f"\nFinal Verdict:\n{final_verdict}")

if __name__ == "__main__":
    scenario = Scenario(config_path='config/election_2024.json')
    scenario.run()