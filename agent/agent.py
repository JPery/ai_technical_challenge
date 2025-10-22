import logging
import os
from typing import List, Dict
from openai import OpenAI
import nltk
from agent.constants import SYSTEM_PROMPT, DEFAULT_LANG, DEFAULT_TOP_K, CONTEXT_PROMPT, OPENAI_API_KEY, \
    OPENAI_API_URL, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DATA_FOLDER, LLM_MODEL_NAME, RETRIEVER_DIR
from agent.retrievers import HybridRetriever
from agent.utils import load_retriever, save_retriever, load_and_preprocess_data

LOGGER = logging.getLogger("airline-agent:agent")

def setup_agent():
    """
    Setup the agent
    :return: The instantiated agent
    """
    # Downloads needed resources
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    # Loads an existing retriever or creates a new one
    if os.path.exists(RETRIEVER_DIR):
        LOGGER.info(f"Loading retriever from {RETRIEVER_DIR}")
        retriever = load_retriever()
    else:
        LOGGER.info(f"Creating new retriever")
        # Carga el dataset
        dataset = load_and_preprocess_data(DATA_FOLDER)
        retriever = HybridRetriever()
        retriever.build_index(dataset, lang=DEFAULT_LANG)
        LOGGER.info(f"Saving retriever in {RETRIEVER_DIR}")
        save_retriever(retriever)
        retriever = retriever

    # Initializes agent
    return Agent(retriever)

class Agent:
    def __init__(self, retriever):
        self.retriever = retriever
        self.conversation_history = []
        self.relevant_docs = None
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_URL,
        )

    def generate_prompt(self, query: str, context_documents: List[str]) -> List[List[Dict]]:
        """
        Generates full prompt for the given query
        :param query: Query to generate prompt for
        :param context_documents: List of context documents
        :return: List of full prompt for the given query
        """
        context = ""
        for i, doc in enumerate(context_documents):
            context += f"\n{doc}\n"
        system_prompt = f"{SYSTEM_PROMPT}\n\n{CONTEXT_PROMPT}\n{context}"
        prompt = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}, ]
                },
                *self.conversation_history,
                {
                    "role": "user",
                    "content": [{"type": "text", "text": query}, ]
                },
            ],
        ]
        return prompt

    def chat_request(self, prompt):
        """
        Perform a chat request into the model
        :param prompt: Generated prompt for the given query
        :return: Text of the response
        """
        return self.client.chat.completions.create(
            messages=prompt[0],
            model=LLM_MODEL_NAME,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            stream=True
        )

    def chat(self, user_input: str, lang: str = DEFAULT_LANG, top_k: int = DEFAULT_TOP_K):
        """
        Create a chat request into the model. First retrieves relevant documents, then creates a prompt for the given query
        :param user_input: Given query
        :param lang: language of the documents
        :param top_k: number of documents to use as context
        :return: a completion of the chat request
        """
        search_query = "\n".join([x['content'][0]['text'] for x in filter(lambda x: x['role'] == 'user', self.conversation_history)]) + '\n' + user_input
        relevant_docs = self.retriever.search_documents(search_query, top_k=top_k, lang=lang)
        prompt = self.generate_prompt(user_input, relevant_docs)
        return self.chat_request(prompt)

    def save_new_message(self, user_input, response):
        """
        Save a new message to the model
        :param user_input: Given query by the user
        :param response: Response from the chat request
        """
        self.conversation_history.extend([
            {
                "role": "user",
                "content": [{"type": "text", "text": user_input}, ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}, ]
            },
        ])

    def clear_history(self):
        """
        Clear the history of the chat request
        """
        self.conversation_history = []
        self.relevant_docs = None

