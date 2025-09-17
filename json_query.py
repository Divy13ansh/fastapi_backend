
import json
import os
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

def get_qdrant_client():
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

def get_embeddings():
    model_name = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model_name)

def get_vector_store(client, embeddings):
    collection_name = os.getenv("QDRANT_COLLECTION", "my_json_collection")
    return QdrantVectorStore(
        client=client,
        embedding=embeddings,
        collection_name=collection_name
    )

def get_llm():
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
        temperature=float(os.getenv("AZURE_OPENAI_TEMPERATURE", "0")),
    )

def get_qa_chain(llm, vector_store):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
    )

def query_loop():
    client = get_qdrant_client()
    embeddings = get_embeddings()
    vector_store = get_vector_store(client, embeddings)
    llm = get_llm()
    qa_chain = get_qa_chain(llm, vector_store)

    print("✅ Qdrant + RAG pipeline ready. Ask me anything (type 'exit' to quit).")

    while True:
        query = input("\nEnter your query: ")
        if query.lower() in ["exit", "quit", "q"]:
            break

        # Step 1: check retriever scores
        results = vector_store.similarity_search_with_score(query, k=3)
        print(max(score for _, score in results))
        print()
        if not results or max(score for _, score in results) < 0.37:
            print("❌ I don’t know based on the stored documents.")
            continue

        # Step 2: run through RAG with strict prompt
        result = qa_chain({"query": query})
        print("\n📌 Answer:", result["result"])
        doc = result["source_documents"][0]
        print("\n📖 Source Pages:", [doc.metadata.get("pages")])

if __name__ == "__main__":
    query_loop()
