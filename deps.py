import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


from typing import Optional


_qdrant_client = None
_vector_store = None
_embeddings = None
_llm = None


def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        url = os.environ.get("QDRANT_URL")
        api_key = os.environ.get("QDRANT_API_KEY")
        _qdrant_client = QdrantClient(url=url, api_key=api_key)
    return _qdrant_client


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        model = os.environ.get("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        _embeddings = HuggingFaceEmbeddings(model_name=model)
    return _embeddings


def get_vector_store(collection_name: Optional[str] = None):
    global _vector_store
    if _vector_store is None:
        client = get_qdrant_client()
        emb = get_embeddings()
        collection_name = collection_name or os.environ.get("QDRANT_COLLECTION", "my_json_collection")
        _vector_store = QdrantVectorStore(client=client, embedding=emb, collection_name=collection_name)
    return _vector_store


def get_llm():
    global _llm
    if _llm is None:
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")
        temperature = float(os.environ.get("AZURE_TEMPERATURE", "0"))
        _llm = AzureChatOpenAI(
            azure_deployment=deployment,
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            temperature=temperature,
        )
    return _llm