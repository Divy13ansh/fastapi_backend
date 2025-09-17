
import uuid
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from sqlmodel import Session, select


from db import ChatSession, get_db, init_db, Message
import db as db
from json_query import get_qdrant_client, get_embeddings, get_vector_store, get_llm, get_qa_chain

app = FastAPI(title="RAG Chat API — End-to-end")

db.init_db()


# Request/Response schemas
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    query: str


class Source(BaseModel):
    page: Optional[str]
    snippet: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[Source]




@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    session_id = req.session_id or str(uuid.uuid4())

    # Save session row if new
    existing = db.exec(select(ChatSession).where(ChatSession.session_id == session_id)).first()
    if not existing:
        db.add(ChatSession(session_id=session_id))
        db.commit()

    # Use json_query.py logic
    client = get_qdrant_client()
    embeddings = get_embeddings()
    vector_store = get_vector_store(client, embeddings)
    llm = get_llm()
    qa_chain = get_qa_chain(llm, vector_store)

    results = vector_store.similarity_search_with_score(req.query, k=3)
    if not results or max(score for _, score in results) < 0.37:
        answer = "I don’t know based on the stored documents."
        sources = []
    else:
        out = qa_chain({"query": req.query})
        answer = out.get("result") or out.get("answer") or ""
        sources = []
        for doc in out.get("source_documents", []):
            page = doc.metadata.get("pages", "N/A")
            if isinstance(page, list):
                page = ", ".join(str(p) for p in page)
            sources.append({
                "page": page,
                "snippet": doc.page_content[:400]
            })

    # persist messages
    db.add(Message(session_id=session_id, role="user", content=req.query))
    db.add(Message(session_id=session_id, role="assistant", content=answer))
    db.commit()

    return ChatResponse(session_id=session_id, answer=answer, sources=sources)