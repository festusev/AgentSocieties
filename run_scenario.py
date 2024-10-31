import json
import os
import re
from duckduckgo_search import DDGS
import autogen

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

        # Initialize the variable store
        self.store = {}

        # Store the root question
        self.root_question = self.config.get('root_question', '')

        # Initialize agents
        self._initialize_agents()

    def _initialize_agents(self):
        """
        Initialize all agents as specified in the configuration.
        """
        self.agents = {}
        for agent_name, agent_config in self.config['agents'].items():
            agent = self._create_agent(agent_config)
            self.agents[agent_name] = agent

    def _create_agent(self, agent_config):
        """
        Create an agent based on the provided configuration.

        Args:
            agent_config (dict): Configuration for the agent.

        Returns:
            autogen.ConversableAgent: Initialized agent.
        """
        # Using gpt-4o-mini to optimize for cost
        config_list = [{"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")}]

        llm_enabled = agent_config['capabilities'].get('llm', False)
        llm_config = {"config_list": config_list} if llm_enabled else False

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
        Perform a message action between agents.

        Args:
            step (dict): Configuration for the message action.
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
        receiver = self._get_agent(receiver_name)

        if isinstance(receiver, list):
            # Broadcasting message to multiple agents (e.g., Jurors)
            responses = {}
            for agent in receiver:
                initiator.initiate_chat(agent, message=message, silent=silent, max_turns=max_turns)
                responses[agent.name] = agent.last_message()['content']
            if store_variable:
                self.store[store_variable] = responses
        else:
            # One-to-one message
            initiator.initiate_chat(receiver, message=message, silent=silent, max_turns=max_turns)
            if store_variable:
                self.store[store_variable] = receiver.last_message()['content']

    
    def _format_article(self, article):
        pass
        
    def _action_web_search(self, step):
        """
        Perform a web search action.

        Args:
            step (dict): Configuration for the web search action.
        """
        query_template = step['query']
        num_results = step.get('num_results', 10)
        store_variable = step.get('store', None)


        # Resolve placeholders in the query
        query = self._normalize_query(self._resolve_placeholders(query_template))
        # Perform search using DuckDuckGo API
        search_results = self._fetch_news_articles(query, num_results)
        
        # Store the articles if needed
        if store_variable:
            self.store[store_variable] = search_results

    
    def _normalize_query(self, query):
        """
        Normalize the query by converting to lowercase and removing punctuation.
        """
        return query.lower().replace(",", "").replace(".", "").replace("?", "").replace("!", "").replace("'", "").replace('"', "")

    def _fetch_news_articles(self, query, num_results=20):
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
        print(f"Starting scenario: {self.root_question}\n")
        steps = self.config.get('scenario', [])
        for step in steps:
            print(f"Executing step {step['step']}: {step['action_type']}")
            self._execute_action(step)

        # After running all steps, you can output or process the final results
        final_verdict = self.store.get('final_verdict', '')
        print("\nFinal Verdict:")
        print(final_verdict)

if __name__ == "__main__":
    scenario = Scenario(config_path='config/eric_adams_indictment.json')
    scenario.run()