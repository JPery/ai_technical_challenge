import logging
import os
import pickle
from typing import List, Tuple
from agent.constants import SENTENCE_TRANSFORMER_MODEL, DEFAULT_LANG, DEFAULT_TOP_K, MIN_HYBRID_RETRIEVER_SCORE, \
    RETRIEVER_DIR, DATA_FOLDER
from agent.retrievers import Retriever, load_and_preprocess_data
from agent.retrievers.dense_retriever import DenseRetriever
from agent.retrievers.sparse_retriever import SparseRetriever

SPARSE_RETRIEVER_PICKLE = 'sparse_bm25.pkl'
CONFIG_PICKLE = 'config.pkl'


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

    def save(self, save_dir=RETRIEVER_DIR):
        """
        Saves the retriever to be loaded later
        :param retriever: retriever to be saved
        :param save_dir: directory to save the retriever
        :return:
        """
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save sparse retriever data
        with open(os.path.join(save_dir, SPARSE_RETRIEVER_PICKLE), 'wb') as f:
            pickle.dump(self.sparse_retriever.model, f)

        # Save model name and weights
        config = {
            'model_name': self.model,
            'weight_sparse': self.weight_sparse,
            'weight_dense': self.weight_dense,
        }
        with open(os.path.join(save_dir, CONFIG_PICKLE), 'wb') as f:
            pickle.dump(config, f)

        print(f"Retriever saved in {save_dir}")

    def load(self, save_dir=RETRIEVER_DIR):
        """
        Loads the retriever previously saved
        :param save_dir: directory where retriever is saved
        :return: Retriever previously saved
        """
        # Load retriever config
        with open(os.path.join(save_dir, CONFIG_PICKLE), 'rb') as f:
            config = pickle.load(f)

        # Initialize retriever with saved config
        self.sparse_retriever = SparseRetriever()
        self.dense_retriever = DenseRetriever(config['model_name'])

        # Initialize index if does not exist the path
        if not os.path.exists(os.path.join(save_dir, SPARSE_RETRIEVER_PICKLE)):
            self.build_index(load_and_preprocess_data(DATA_FOLDER), lang=DEFAULT_LANG)
        else:
            # Load sparse retriever data
            with open(os.path.join(save_dir, SPARSE_RETRIEVER_PICKLE), 'rb') as f:
                self.sparse_retriever.model = pickle.load(f)

            logging.getLogger("airline-agent:utils").info(f"Retriever loaded from {save_dir}")
