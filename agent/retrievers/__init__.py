from abc import ABC, abstractmethod
from typing import List, Tuple
from nltk import word_tokenize
from nltk.corpus import stopwords
from agent.constants import DEFAULT_LANG, DEFAULT_TOP_K

namespace = "airline-namespace"

class TextPreprocessor:

    @classmethod
    def preprocess(cls, text: str, lang=DEFAULT_LANG) -> str:
        if isinstance(text, tuple): text = ' '.join(text)
        text = text.lower()
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words(lang))
        tokens = [t for t in tokens if t not in stop_words]
        return ' '.join(tokens)


class Retriever(ABC):

    def __init__(self, name='abstract_retriever'):
        self.name = name

    def get_name(self):
        return self.name

    @abstractmethod
    def build_index(self, documents: List[str], lang: str = DEFAULT_LANG):
        """
        This method receives a set of documents and indexes them in order to perform searches
        :param documents: List of documents
        :param lang: Language of the documents
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = DEFAULT_TOP_K, lang: str = DEFAULT_LANG) -> List[Tuple[int, float]]:
        """
        This method returns the top k results for a given query in the index
        :param query: Query to search for
        :param top_k: Number of results to return
        :param lang: Language of the documents
        :return: A list of tuples containing the top k results in the index
        """
        pass

    def search_documents(self, query: str, top_k: int = DEFAULT_TOP_K, lang: str = DEFAULT_LANG) -> List[str]:
        """
        This method returns the top k results for a given query in the documents
        :param query: Query to search for
        :param top_k: Number of results to return
        :param lang: Language of the documents
        :return: A list of tuples containing the top k results in the documents
        """
        relevant_documents = self.search(query, top_k, lang)
        return [text for text, _, _ in relevant_documents]

