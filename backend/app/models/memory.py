"""
Memory models — stored in PostgreSQL.

Tables:
  chat_messages   — short-term turn-by-turn history (last N turns)
  chat_summaries  — rolling compressed summary per notebook
"""
import uuid
from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from app.db.database import Base


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    notebook_id = Column(UUID(as_uuid=True), ForeignKey("notebooks.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    conversation_id = Column(String, nullable=False, index=True)  # groups turns in one session
    role = Column(String, nullable=False)   # "user" | "assistant"
    content = Column(Text, nullable=False)
    turn_index = Column(Integer, nullable=False, default=0)  # monotonically increasing per conversation
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ChatSummary(Base):
    __tablename__ = "chat_summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    notebook_id = Column(UUID(as_uuid=True), ForeignKey("notebooks.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    conversation_id = Column(String, nullable=False, index=True)
    summary = Column(Text, nullable=False)        # compressed rolling summary
    turns_covered = Column(Integer, nullable=False, default=0)  # how many turns are summarised
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
