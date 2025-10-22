from typing import List, Tuple
from agent.constants import SENTENCE_TRANSFORMER_MODEL, DEFAULT_LANG, DEFAULT_TOP_K, MIN_HYBRID_RETRIEVER_SCORE
from agent.retrievers import Retriever
from agent.retrievers.dense_retriever import DenseRetriever
from agent.retrievers.sparse_retriever import SparseRetriever


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
