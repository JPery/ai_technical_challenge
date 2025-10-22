from abc import ABC, abstractmethod
from typing import List, Tuple
from nltk import word_tokenize
from nltk.corpus import stopwords
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import SentenceTransformer
from agent.constants import DEFAULT_LANG, DEFAULT_TOP_K, SENTENCE_TRANSFORMER_MODEL, MIN_HYBRID_RETRIEVER_SCORE, PINECONE_API_KEY

pc = Pinecone(api_key=PINECONE_API_KEY)
sparse_index_name = "airline-sparse"
dense_index_name = "airline-dense"
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


class SparseRetriever(Retriever):
    def __init__(self):
        super().__init__('sparse_retriever')
        self.model = BM25Encoder()

    def build_index(self, documents: List[str], lang: str = DEFAULT_LANG):
        processed_docs = [TextPreprocessor.preprocess(doc, DEFAULT_LANG) for doc in documents]
        self.model.fit(processed_docs)
        if not pc.has_index(sparse_index_name):
            embeddings = self.model.encode_documents(processed_docs)
            pc.create_index(
                name=sparse_index_name,
                vector_type="sparse",
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            pc.Index(sparse_index_name).upsert([
                {
                    "id": "Doc-" + str(i),
                    "sparse_values": emb,
                    "metadata": {"text": text},
                } for i, (text, emb) in enumerate(zip(documents, embeddings))],
                namespace=namespace
            )

    def search(self, query: str, top_k: int = DEFAULT_TOP_K, lang: str = DEFAULT_LANG) -> List[Tuple[str, float, str]]:
        processed_query = TextPreprocessor.preprocess(query, lang)
        vector = self.model.encode_queries(processed_query)
        sparse_index = pc.Index(sparse_index_name)
        a = sparse_index.query(namespace=namespace,
                               sparse_vector=vector,
                               top_k=top_k,
                               include_metadata=True,
                               include_values=False
            )
        return [(
            x.metadata['text'],
            x.score,
            x.id
        ) for x in a.matches]


class DenseRetriever(Retriever):

    def __init__(self, model=SENTENCE_TRANSFORMER_MODEL):
        super().__init__('dense_retriever' + model)
        self.model = SentenceTransformer(model)

    def build_index(self, documents: List[str], lang: str = DEFAULT_LANG):
        if not pc.has_index(dense_index_name):
            processed_docs = [TextPreprocessor.preprocess(doc, DEFAULT_LANG) for doc in documents]
            embeddings = self.model.encode(processed_docs, show_progress_bar=True)
            pc.create_index(
                name=dense_index_name,
                vector_type="dense",
                dimension=embeddings.shape[-1],
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ),
            )
            pc.Index(dense_index_name).upsert([
                {
                    "id": "Doc-" + str(i),
                    "values": emb,
                    "metadata": {"text": text},
                } for i, (text, emb) in enumerate(zip(documents, embeddings))],
                namespace=namespace
            )

    def search(self, query: str, top_k: int = DEFAULT_TOP_K, lang: str = DEFAULT_LANG) -> List[Tuple[str, float, str]]:
        processed_query = TextPreprocessor.preprocess(query, lang)
        vector = self.model.encode(processed_query)
        dense_index = pc.Index(dense_index_name)
        a = dense_index.query(namespace=namespace,
                              vector=vector.tolist(),
                              top_k=top_k,
                              include_metadata=True,
                              include_values=False
                              )
        return [(
            x.metadata['text'],
            x.score,
            x.id
        ) for x in a.matches]

class HybridRetriever(Retriever):

    def __init__(self, weight_sparse: float = 0.3,
                 weight_dense: float = 0.7, model=SENTENCE_TRANSFORMER_MODEL):
        super().__init__('hybrid_retriever' + model)
        self.model = model
        self.weight_sparse = weight_sparse
        self.weight_dense = weight_dense
        self.sparse_retriever = SparseRetriever()
        self.dense_retriever = DenseRetriever(self.model)

    def build_index(self, documents: List[str], lang: str = DEFAULT_LANG):
        self.sparse_retriever.build_index(documents)
        self.dense_retriever.build_index(documents)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K, lang: str = DEFAULT_LANG) -> List[Tuple[str, float, str]]:
        sparse_results = self.sparse_retriever.search(query, top_k=top_k, lang=lang)
        dense_results = self.dense_retriever.search(query, top_k=top_k, lang=lang)
        combined_scores = {}
        for text, score, _id in sparse_results:
            combined_scores[_id] = {
                'score': score * self.weight_sparse,
                'text': text
            }

        for text, score, _id in dense_results:
            if _id in combined_scores:
                combined_scores[_id]['score'] += score * self.weight_dense
            else:
                combined_scores[_id] = {
                    'score': score * self.weight_sparse,
                    'text': text
                }
        sorted_results = sorted(combined_scores.items(),
                                key=lambda x: x[1]['score'],
                                reverse=True)[:top_k]
        return [(x['text'], x['score'], _id) for _id, x in sorted_results if x['score'] > MIN_HYBRID_RETRIEVER_SCORE]

