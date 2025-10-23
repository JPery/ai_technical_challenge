from typing import List, Tuple
from pinecone import ServerlessSpec, Pinecone
from pinecone_text.sparse import BM25Encoder
from agent.constants import DEFAULT_LANG, PINECONE_API_KEY, DEFAULT_TOP_K
from agent.retrievers import Retriever, TextPreprocessor, namespace

sparse_index_name = "airline-sparse"

class SparseRetriever(Retriever):
    def __init__(self):
        super().__init__('sparse_retriever')
        self.model = BM25Encoder()

    def build_index(self, documents: List[str], lang: str = DEFAULT_LANG):
        processed_docs = [TextPreprocessor.preprocess(doc, DEFAULT_LANG) for doc in documents]
        self.model.fit(processed_docs)
        pc = Pinecone(api_key=PINECONE_API_KEY)
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
        sparse_index = Pinecone(api_key=PINECONE_API_KEY).Index(sparse_index_name)
        query_result = sparse_index.query(namespace=namespace,
                               sparse_vector=vector,
                               top_k=top_k,
                               include_metadata=True,
                               include_values=False
            )
        return [(
            x.metadata['text'],
            x.score,
            x.id
        ) for x in query_result.matches]