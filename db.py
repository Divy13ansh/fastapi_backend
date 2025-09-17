from sqlmodel import SQLModel, Field, create_engine, Session
from typing import Optional
from datetime import datetime


DATABASE_URL = "sqlite:///./chat_history.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


class ChatSession(SQLModel, table=True):
	id: Optional[int] = Field(default=None, primary_key=True)
	session_id: str
	created_at: datetime = Field(default_factory=datetime.utcnow)


class Message(SQLModel, table=True):
	id: Optional[int] = Field(default=None, primary_key=True)
	session_id: str
	role: str # 'user' or 'assistant'
	content: str
	created_at: datetime = Field(default_factory=datetime.utcnow)


def init_db():
	SQLModel.metadata.create_all(engine)


def get_db():
	with Session(engine) as session:
		yield session