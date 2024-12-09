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
from typing import Literal, Optional
from pydantic import BaseModel

class AgentConfig(BaseModel):
    name: str = ''
    system_message: str = ''
    capabilities: dict[Literal["web_retrieval", "llm"], bool]
    jury_selection: Optional[dict[
        Literal["requirements", "profile_fields", "prompt_template"], list[str] | str
    ]] = None

class StepConfig(BaseModel):
    step: str
    action_type: str
    initiator: Optional[str] = None
    receiver: Optional[str | list[str]] = None
    message: Optional[str] = None
    store: str = ''
    num_jurors: int = 3
    max_turns: int = 1
    silent: bool = False
    agent: Optional[str] = None # Consider combining this with initiator
    query: Optional[str] = None
    criteria: str = 'relevance and objectivity'
    num_results: int = 10
    num_top_articles: int = 6


class ScenarioConfig(BaseModel):
    root_question: str = ''
    agents: dict[str, AgentConfig]
    scenario: list[StepConfig]


class Scenario:
    """
    A modular Scenario class that executes steps based on a JSON configuration.

    This class initializes agents and performs actions such as messaging and
    web searches as specified in the config.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initialize the Scenario with agents and configuration.

        Args:
            config_path (str): Path to the JSON config file.
        """
        # Load configuration from JSON file
        with open(config_path, 'r') as f:

            self.config: ScenarioConfig = ScenarioConfig.model_validate_json(f.read())

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
        self.root_question = self.config.root_question
        self.config_list = [{"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}]

        # Initialize agents
        self._initialize_agents()

    def _initialize_agents(self) -> None:
        """
        Initialize all agents as specified in the configuration.
        """
        # Initialize agents
        self.agents = {}
        for agent_name, agent_config in self.config.agents.items():
            agent = self._create_agent(agent_config)
            self.agents[agent_name] = agent

        # Automatically generate jury profiles if enabled
        # Otherwise, Jurors were already specified in the config
        self.jurors = []  # If "Jurors" is the recipient, these names are used

    def _create_agent(self, agent_config: AgentConfig) -> autogen.ConversableAgent:
        """
        Create an agent based on the provided configuration.

        Args:
            agent_config (AgentConfig): Configuration for the agent.

        Returns:
            autogen.ConversableAgent: Initialized agent.
        """
        # Using gpt-4o-mini to optimize for cost
        llm_enabled = agent_config.capabilities.get('llm', False)
        llm_config = {"config_list": self.config_list} if llm_enabled else False

        return autogen.ConversableAgent(
                name=agent_config.name,
                system_message=agent_config.system_message,
                llm_config=llm_config,
                human_input_mode="NEVER",
        )

    def _resolve_placeholders(self, text: str) -> str:
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

    def _execute_action(self, step: StepConfig) -> None:
        """
        Execute a single action as specified in the scenario steps.

        Args:
            step (StepConfig): Configuration for the step/action.
        """
        action_type = step.action_type
        if action_type == 'generate_jury':
            self._action_generate_jury(step)
        elif action_type == 'message':
            self._action_message(step)
        elif action_type == 'web_search':
            self._action_web_search(step)
        elif action_type == 'process_articles':
            self._action_process_articles(step)
        else:
            print(f"Unknown action type: {action_type}")

    def _get_default_jury_profiles(self, step: StepConfig) -> dict[str, AgentConfig]:
        """
        Get default jury profiles for the case topic
        """
        num_jurors = step.num_jurors
        jury_profiles = {}
        for i in range(num_jurors):
            profile = AgentConfig(
                name=f"Juror{i+1}",
                system_message="You are a juror tasked with evaluating evidence and making probability forecasts.",
                capabilities={
                    "web_retrieval": False,
                    "llm": True
                }
            )
            jury_profiles[f"Juror{i + 1}"] = profile
            self.jurors.append(f"Juror{i + 1}")

            juror = self._create_agent(profile)
            self.agents[profile.name] = juror

        return jury_profiles

    def _action_generate_jury(self, step: StepConfig) -> dict[str, AgentConfig]:
        """Generate diverse jury profiles based on the case topic"""
        num_jurors = step.num_jurors
        clerk = self._get_agent(step.initiator)
        user_proxy = self._get_agent(step.receiver)

        clerk_config = self.config.agents[step.initiator].jury_selection
        # Prompt for the clerk to generate profiles
        prompt = clerk_config['prompt_template'].format(
                num_jurors=num_jurors,
                root_question=self.root_question,
                requirements="\n".join(f"{i + 1}. {r}" for i, r in enumerate(clerk_config['requirements'])),
                profile_fields="\n".join(f"- {f}" for f in clerk_config['profile_fields'])
        )

        user_proxy.initiate_chat(clerk, message=prompt, silent=False, max_turns=1)
        try:
            message_content = clerk.last_message()["content"]
            if "```json" in message_content:
                message_content = message_content.split("```json")[1].split("```")[0].strip()
            profiles = json.loads(message_content)

            for profile in profiles:
                profiles[profile]["capabilities"] = {
                    "web_retrieval": False,
                    "llm": True
                }
                profiles[profile][
                    "system_message"] = f"""{profiles[profile]["profile"]} You are a juror tasked with evaluating evidence and making probability forecasts with your best judgement."""
                self.jurors.append(profile)

                juror = self._create_agent(profiles[profile])
                self.agents[profile] = juror
            return profiles
        except Exception as e:
            logging.error(f"Failed to parse jury profiles JSON: {e}")
            return self._get_default_jury_profiles(step)

    def _action_message(self, step: StepConfig) -> None:
        """
        Updated message action with logging
        """
        initiator_name = step.initiator
        receiver_name = step.receiver
        message_template = step.message
        store_variable = step.store

        # Resolve placeholders in the message
        message = self._resolve_placeholders(message_template)

        # Get initiator and receiver agents
        initiator = self._get_agent(initiator_name)

        if receiver_name == "Jurors":
            receiver_name = self.jurors

        if not isinstance(receiver_name, list):
            receiver_name = [receiver_name]

        receiver = self._get_agent(receiver_name)

        # Log the prompt
        logging.info(f"PROMPT - From: {initiator_name}, To: {receiver_name}\n{message}\n")

        # Broadcast the message
        responses = {}
        for agent in receiver:
            initiator.initiate_chat(agent, message=message, silent=step.silent, max_turns=step.max_turns)
            response = agent.last_message()['content']
            responses[agent.name] = response
            logging.info(f"RESPONSE - From: {agent.name}\n{response}\n")

        if store_variable:
            self.store[store_variable] = responses

    def _format_article(self, article):
        pass

    def _action_web_search(self, step: StepConfig) -> None:
        """
        Perform a web search action.
        """
        query_template = step.query
        num_results = step.num_results
        store_variable = step.store

        # Resolve placeholders in the query
        query = self._normalize_query(self._resolve_placeholders(query_template))
        logging.info(f"Normalized search query: {query}")

        # Perform search using DuckDuckGo API
        search_results = self._fetch_news_articles(query, num_results)

        # Store the articles if needed
        if store_variable:
            self.store[store_variable] = search_results
            logging.info(f"Stored {len(search_results)} articles in variable: {store_variable}")

    def _normalize_query(self, query: str) -> str:
        """
        Normalize the query by converting to lowercase and removing punctuation.
        """
        normalized = re.sub(r"[,.?!'\"]", "", query.lower())
        return normalized

    def _fetch_news_articles(self, query: str, num_results=20) -> list[dict]:
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

    def _process_article_contents(self, articles: list[dict]) -> list[dict]:
        """
        Process and summarize article contents using chunks, limited to 3 chunks per article.
        Returns a list of processed articles with metadata and summaries.
        """
        encoder = tiktoken.encoding_for_model("gpt-4")
        chunk_size = 2000
        max_chunks = 3
        user_agent = self._get_agent('UserAgent')
        summarizer_agent = self._get_agent('HeadAgent')

        def analyze_url(url: str) -> dict:
            """Analyze URL for source and date"""
            prompt = f"""Analyze only the URL to extract:
1. Source/publisher name from the domain:
   - apnews.com → AP News
   - cnn.com → CNN
   - cbsnews.com → CBS News
   - reuters.com → Reuters
2. Publication date from URL path:
   - /2024/10/01/ → 2024-10-01
   - /article/2024-09-16/ → 2024-09-16
   - /news/2024/03/ → 2024-03

URL: {url}

Format response as JSON with ONLY these fields:
{{
    "date": "YYYY-MM-DD if found in URL, otherwise 'unknown'",
    "source": "source if domain recognized, otherwise 'unknown'"
}}"""

            user_agent.initiate_chat(
                    summarizer_agent,
                    message=prompt,
                    silent=True,
                    max_turns=1
            )

            try:
                return json.loads(summarizer_agent.last_message()['content'])
            except json.JSONDecodeError:
                return {"date": "unknown", "source": "unknown"}

        def analyze_content(content: str, metadata: dict) -> dict:
            """Analyze content for title, summary, and missing metadata"""
            prompt = f"""Analyze this content to extract:
1. Title of the article
2. A summary of the key points
3. Publication date (ONLY if found in article AND current date is unknown)
4. Source/publisher (ONLY if found in article AND current source is unknown)

Current metadata:
Date: {metadata['date']}
Source: {metadata['source']}

Content:
{content}

Return ONLY these fields in JSON:
{{
    "title": "actual title",
    "summary": "actual summary",
    "date": "{metadata['date']}" or "YYYY-MM-DD" if new date found,
    "source": "{metadata['source']}" or "actual source" if new source found
}}"""

            user_agent.initiate_chat(
                    summarizer_agent,
                    message=prompt,
                    silent=True,
                    max_turns=1
            )

            try:
                response = summarizer_agent.last_message()['content']

                # Check if response contains nested JSON
                if '```json' in response:
                    json_content = response.split('```json')[1].split('```')[0].strip()
                    content_data = json.loads(json_content)
                else:
                    content_data = json.loads(response)

                # Validate and clean the data
                if content_data.get('date', '').startswith('YYYY-MM-DD') or 'if found' in content_data.get('date', ''):
                    content_data['date'] = metadata['date']

                if content_data.get('source', '') == 'actual source':
                    content_data['source'] = metadata['source']

                # Only update date/source if they were unknown and new values are valid
                if metadata['date'] != "unknown":
                    content_data['date'] = metadata['date']
                if metadata['source'] != "unknown":
                    content_data['source'] = metadata['source']

                # Ensure summary doesn't contain JSON formatting
                if isinstance(content_data.get('summary'), str) and '```json' in content_data['summary']:
                    try:
                        nested_json = json.loads(content_data['summary'].split('```json')[1].split('```')[0].strip())
                        content_data.update(nested_json)
                    except (json.JSONDecodeError, IndexError):
                        pass

                return content_data
            except json.JSONDecodeError:
                return {
                    "title": "unknown",
                    "summary": summarizer_agent.last_message()['content'],
                    "date": metadata['date'],
                    "source": metadata['source']
                }

        processed_articles = []
        for article in articles:
            content = article['content']
            url = article.get('url', '')

            # First analyze URL for metadata
            metadata = analyze_url(url)

            # Then analyze content and update metadata
            article_data = analyze_content(content, metadata)

            # If title is still unknown, use original article title
            if article_data["title"] == "unknown":
                article_data["title"] = article.get("title", "unknown")

            processed_articles.append(article_data)
            logging.info(f"\nProcessed Article:\n{json.dumps(article_data, indent=2)}\n")
            logging.info("=" * 80 + "\n")

        return processed_articles

    def _action_process_articles(self, step: StepConfig) -> None:
        """
        Process and rank articles based on specified criteria.
        """
        agent_name = step.agent
        articles = self.store.get('articles', [])
        criteria = step.criteria
        num_top_articles = step.num_top_articles

        agent = self._get_agent(agent_name)
        user_agent = self._get_agent('UserAgent')

        # Rank each article
        ranked_articles = []
        for article in articles:
            # Extract the actual article data if it's nested in JSON
            if isinstance(article['summary'], str) and '```json' in article['summary']:
                try:
                    json_content = article['summary'].split('```json')[1].split('```')[0].strip()
                    nested_data = json.loads(json_content)
                    article.update(nested_data)
                except (json.JSONDecodeError, IndexError):
                    pass

            ranking_prompt = f"""Rate this article on two criteria:
1. Relevance to the question '{self.root_question}' (1-10)
2. {criteria} (1-10)

Only respond with two numbers separated by a comma (e.g. "8,7")

Article Details:
Title: {article['title']}
Date: {article['date']}
Source: {article['source']}
Summary: {article['summary']}"""

            user_agent.initiate_chat(agent, message=ranking_prompt, max_turns=1)

            try:
                score1, score2 = map(int, agent.last_message()['content'].strip().split(','))
                score = (score1 + score2) / 2
            except:
                score = 0

            ranked_articles.append({'article': article, 'score': score})

        # Sort and get top articles
        top_articles = sorted(ranked_articles, key=lambda x: x['score'], reverse=True)[:num_top_articles]

        # Format the top articles with their metadata
        formatted_articles = []
        for ranked_article in top_articles:
            article = ranked_article['article']
            formatted_article = f"""Title: {article['title']}
Date: {article['date']}
Source: {article['source']}
Summary: {article['summary']}
---"""
            formatted_articles.append(formatted_article)

        # Store results if needed
        if step.store:
            self.store[step.store] = formatted_articles

    def run(self) -> None:
        """
        Execute the scenario steps as defined in the configuration.
        """
        logging.info(f"Starting scenario: {self.root_question}\n")
        for step in self.config.scenario:
            logging.info(f"Executing step {step.step}: {step.action_type}")
            self._execute_action(step)

        final_verdict = self.store.get('final_verdict', '')
        logging.info(f"\nFinal Verdict:\n{final_verdict}")
