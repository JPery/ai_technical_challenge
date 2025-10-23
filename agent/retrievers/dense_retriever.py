from typing import List, Tuple
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from agent.constants import SENTENCE_TRANSFORMER_MODEL, DEFAULT_LANG, PINECONE_API_KEY, DEFAULT_TOP_K
from agent.retrievers import Retriever, TextPreprocessor, namespace

dense_index_name = "airline-dense"

class DenseRetriever(Retriever):

    def __init__(self, model=SENTENCE_TRANSFORMER_MODEL):
        super().__init__('dense_retriever' + model)
        self.model = SentenceTransformer(model)

    def build_index(self, documents: List[str], lang: str = DEFAULT_LANG):
        pc = Pinecone(api_key=PINECONE_API_KEY)
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
        dense_index = Pinecone(api_key=PINECONE_API_KEY).Index(dense_index_name)
        query_result = dense_index.query(namespace=namespace,
                              vector=vector.tolist(),
                              top_k=top_k,
                              include_metadata=True,
                              include_values=False
                              )
        return [(
            x.metadata['text'],
            x.score,
            x.id
        ) for x in query_result.matches]
