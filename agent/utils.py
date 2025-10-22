import logging
import os
import glob
import pickle
from typing import List

from agent.constants import DATA_FOLDER, DEFAULT_LANG, RETRIEVER_DIR
from agent.retrievers import HybridRetriever, SparseRetriever, DenseRetriever, Retriever

SPARSE_RETRIEVER_PICKLE = 'sparse_bm25.pkl'
CONFIG_PICKLE = 'config.pkl'

def load_and_preprocess_data(path: str) -> List[str]:
    """
    Loads parsed data and preprocesses it to feed a retriever.
    :param path: path to parsed data
    :return: list of texts of the parsed data
    """
    texts = []
    for item in sorted(glob.glob(f"{path}/*.txt")):
        with open(item, "r", encoding="utf8") as f:
            text = f.read()
            texts.append(text)
    return texts


def save_retriever(retriever: HybridRetriever, save_dir=RETRIEVER_DIR):
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
        pickle.dump(retriever.sparse_retriever.model, f)

    # Save model name and weights
    config = {
        'model_name': retriever.model,
        'weight_sparse': retriever.weight_sparse,
        'weight_dense': retriever.weight_dense,
    }
    with open(os.path.join(save_dir, CONFIG_PICKLE), 'wb') as f:
        pickle.dump(config, f)

    print(f"Retriever saved in {save_dir}")


def load_retriever(save_dir=RETRIEVER_DIR) -> Retriever:
    """
    Loads the retriever previously saved
    :param save_dir: directory where retriever is saved
    :return: Retriever previously saved
    """
    # Load retriever config
    with open(os.path.join(save_dir, CONFIG_PICKLE), 'rb') as f:
        config = pickle.load(f)

    # Initialize retriever with saved config
    retriever = HybridRetriever(
        weight_sparse=config['weight_sparse'],
        weight_dense=config['weight_dense'],
        model=config['model_name']
    )

    retriever.sparse_retriever = SparseRetriever()
    retriever.dense_retriever = DenseRetriever(config['model_name'])

    # Initialize index if does not exist the path
    if not os.path.exists(os.path.join(save_dir, SPARSE_RETRIEVER_PICKLE)):
        retriever.build_index(load_and_preprocess_data(DATA_FOLDER), lang=DEFAULT_LANG)
    else:
        # Load sparse retriever data
        with open(os.path.join(save_dir, SPARSE_RETRIEVER_PICKLE), 'rb') as f:
            retriever.sparse_retriever.model = pickle.load(f)

        logging.getLogger("airline-agent:utils").info(f"Retriever loaded from {save_dir}")
    return retriever
