import uuid
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from sqlmodel import Session, select
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from db import ChatSession, get_db, init_db, Message
import db as db
from json_query import get_qdrant_client, get_embeddings, get_vector_store, get_llm, get_qa_chain

class AppState:
    def __init__(self):
        self.client = None
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize resources
    try:
        app_state.client = get_qdrant_client()
        app_state.embeddings = get_embeddings()
        app_state.vector_store = get_vector_store(app_state.client, app_state.embeddings)
        app_state.llm = get_llm()
        app_state.qa_chain = get_qa_chain(app_state.llm, app_state.vector_store)
        print("✅ All resources initialized successfully")
        yield
    except Exception as e:
        print(f"❌ Error initializing resources: {str(e)}")
        raise
    finally:
        # Shutdown: Clean up resources
        if app_state.client:
            app_state.client.close()

app = FastAPI(title="RAG Chat API — End-to-end", lifespan=lifespan)
app_state = AppState()

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

    try:
        # Use the global instances instead of creating new ones
        results = app_state.vector_store.similarity_search_with_score(req.query, k=3)
        if not results or max(score for _, score in results) < 0.37:
            answer = "I don't know based on the stored documents."
            sources = []
        else:
            out = app_state.qa_chain.invoke({"query": req.query})
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
