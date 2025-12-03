"""
FastAPI server module for the Telegram RAG chatbot.

This module implements the main application server with RAG (Retrieval-Augmented
Generation) logic, conversation management, and Telegram webhook integration.
"""

import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import requests
import re

from .embeddings import SentenceTransformerEmbeddingFunction


load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docs")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")
DB_PATH = os.getenv("DB_PATH", "./conversations.db")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY is not set. Set it in your .env file.")
    genai_client = None
else:
    genai.configure(api_key=GEMINI_API_KEY)


class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"


class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    reply: str
    session_id: str


def init_db() -> None:
    """Initialize SQLite database for conversation storage."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            timestamp REAL NOT NULL,
            sources TEXT
        )
    """)
    conn.commit()
    conn.close()


def get_conversation_history(session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Get recent conversation history for a session."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT question, answer, timestamp 
        FROM conversations 
        WHERE session_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (session_id, limit))
    
    history = []
    for row in cursor.fetchall():
        history.append({
            "question": row[0],
            "answer": row[1],
            "timestamp": row[2]
        })
    
    conn.close()
    return list(reversed(history))  # Return in chronological order


def save_conversation(session_id: str, question: str, answer: str, sources: List[Dict[str, Any]]) -> None:
    """Save conversation to database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO conversations (session_id, question, answer, timestamp, sources)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, question, answer, time.time(), str(sources)))
    conn.commit()
    conn.close()


def build_app() -> FastAPI:
    app = FastAPI(title="Community Assistant Chatbot (Console)", version="0.2.2")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize database
    init_db()

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    embedder = SentenceTransformerEmbeddingFunction()

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    def _answer(question: str, session_id: str = "default") -> AskResponse:
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on server")

        # Check for social media questions and provide direct response
        question_lower = question.lower()
        if any(keyword in question_lower for keyword in ['social media', 'linkedin', 'facebook', 'instagram', 'follow', 'social']):
            social_media_response = (
                "Hei! 👋 Great question! You can follow us on:\n\n"
                "• [LinkedIn](https://www.linkedin.com/company/witasoy/posts/?feedView=all)\n"
                "• [Facebook](https://www.facebook.com/witasoy/)\n"
                "• [Instagram](https://www.instagram.com/witasoy/)\n\n"
                "We share updates about events, news, and opportunities for entrepreneurs in Viitasaari! "
                "Would you like to subscribe to our newsletter as well? 📧"
            )
            return AskResponse(
                answer=social_media_response,
                sources=[{"source": "general-info.txt", "distance": 0.0, "urls": []}],
                session_id=session_id
            )

        # Get conversation history
        history = get_conversation_history(session_id, limit=5)
        
        query_embedding = embedder([question])[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        context_parts: List[str] = []
        sources: List[Dict[str, Any]] = []
        all_urls: List[str] = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            context_parts.append(doc)
            # Parse URLs from semicolon-separated string
            urls_str = meta.get("urls", "")
            source_urls = [url.strip() for url in urls_str.split(";") if url.strip()] if urls_str else []
            all_urls.extend(source_urls)
            sources.append({
                "source": meta.get("source", "unknown"),
                "distance": dist,
                "urls": source_urls,
            })

        context = "\n\n".join(context_parts[:4])
        
        # Build conversation history string
        history_str = ""
        if history:
            history_str = "\n\nPrevious conversation:\n"
            for h in history[-3:]:  # Last 3 exchanges
                history_str += f"User: {h['question']}\nAssistant: {h['answer']}\n"

        # Get unique URLs for reference
        unique_urls = list(set(all_urls))[:3]  # Limit to 3 most relevant URLs
        
        # Load general info for contact details
        general_info = ""
        try:
            with open("general-info.txt", "r", encoding="utf-8") as f:
                general_info = f.read()
        except FileNotFoundError:
            pass
        
        # Add social media links to context if not already present
        social_media_context = """
        Witas Oy Social Media Links:
        - LinkedIn: https://www.linkedin.com/company/witasoy/posts/?feedView=all
        - Facebook: https://www.facebook.com/witasoy/
        - Instagram: https://www.instagram.com/witasoy/
        - Newsletter: https://witas.fi/witas-oy/tilaa-uutiskirje
        """

        prompt = (
            "You are a friendly, professional customer support representative for Witas Oy, a development company serving entrepreneurs in Viitasaari, Finland.\n"
            "Your role is to provide personalized, helpful assistance with a warm, human touch.\n\n"
            "IMPORTANT GUIDELINES:\n"
            "- Keep responses concise (2-3 paragraphs max)\n"
            "- Use a conversational, friendly tone like a real person\n"
            "- Always end with a helpful follow-up question\n"
            "- For contact info, provide specific names, phone numbers, and booking links\n"
            "- Use emojis sparingly but effectively (📞, 📧, 🤝, 💼)\n"
            "- Make responses interactive and engaging\n"
            "- ALWAYS use the EXACT links from the contact information below\n"
            "- Format links as [text](url) for clickable links in Telegram\n\n"
            "ANTI-REDUNDANCY:\n"
            "- If there is any conversation history, DO NOT start with greetings (e.g., 'Hi', 'Hello', 'Hei'). Go straight to the answer.\n\n"
            "CONTACT INFORMATION REFERENCE (USE THESE EXACT LINKS):\n"
            f"{general_info}\n\n"
            f"CONVERSATION HISTORY:\n{history_str}\n\n"
            f"CONTEXT FROM WEBSITE:\n{context}\n\n"
            f"SOCIAL MEDIA AND CONTACT LINKS:\n{social_media_context}\n\n"
            f"AVAILABLE LINKS: {', '.join(unique_urls) if unique_urls else 'None'}\n\n"
            f"USER QUESTION: {question}\n\n"
            "MANDATORY: If the user asks about social media, you MUST respond with this EXACT format:\n"
            "'Hei! 👋 Great question! You can follow us on:\n"
            "• [LinkedIn](https://www.linkedin.com/company/witasoy/posts/?feedView=all)\n"
            "• [Facebook](https://www.facebook.com/witasoy/)\n"
            "• [Instagram](https://www.instagram.com/witasoy/)\n\n"
            "We share updates about events, news, and opportunities for entrepreneurs in Viitasaari! Would you like to subscribe to our newsletter as well?'\n\n"
            "DO NOT say you don't have the links. ALWAYS use the links above.\n\n"
            "RESPONSE FORMAT:\n"
            "1. Give a direct, helpful answer (1-2 paragraphs)\n"
            "2. Include specific contact details if relevant\n"
            "3. Use EXACT links from the contact information above\n"
            "4. Format all links as [text](url) for clickability\n"
            "5. End with a personalized follow-up question\n\n"
            "Remember: Be human, be helpful, be concise, and ALWAYS use the exact links provided!"
        )

        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        answer_text = getattr(response, "text", None) or "No answer generated."

        # If this is a continued conversation, strip leading greetings to avoid repetition
        if history:
            answer_text = re.sub(r"^\s*(?:hi|hello|hei|hey|moikka|terve|hola)[!,.\s\u200b\ud83d\ude4b\U0001F44B]*", "", answer_text, flags=re.IGNORECASE)

        # Save conversation to database
        save_conversation(session_id, question, answer_text.strip(), sources)

        return AskResponse(answer=answer_text.strip(), sources=sources, session_id=session_id)

    @app.get("/api/ask", response_model=AskResponse)
    def ask_get(q: Optional[str] = Query(default=None, alias="question"), session_id: Optional[str] = Query(default="default")) -> AskResponse:
        if q is None:
            raise HTTPException(status_code=400, detail="Provide question param, e.g. /api/ask?question=Hi")
        return _answer(q, session_id)

    @app.post("/api/ask", response_model=AskResponse)
    def ask_post(payload: AskRequest) -> AskResponse:
        return _answer(payload.question, payload.session_id)

    @app.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest) -> ChatResponse:
        resp = _answer(payload.message, payload.session_id)
        return ChatResponse(reply=resp.answer, session_id=resp.session_id)

    @app.post("/telegram/webhook/{secret}")
    async def telegram_webhook(secret: str, request: Request) -> Dict[str, Any]:
        if TELEGRAM_WEBHOOK_SECRET and secret != TELEGRAM_WEBHOOK_SECRET:
            raise HTTPException(status_code=401, detail="Invalid webhook secret")

        update = await request.json()

        message = update.get("message") or update.get("edited_message")
        if not message:
            # Ignore non-message updates (callbacks, joins, etc.)
            return {"ok": True}

        chat = message.get("chat", {})
        chat_id = chat.get("id")
        text = message.get("text", "").strip()

        if not chat_id or not text:
            return {"ok": True}

        # Use chat_id as session_id for continuity
        try:
            resp = _answer(text, str(chat_id))
            reply_text = resp.answer
        except HTTPException as e:
            reply_text = f"Error: {e.detail}"
        except Exception:
            reply_text = "Sorry, something went wrong. Please try again later."

        # Send reply back to Telegram
        if not TELEGRAM_BOT_TOKEN:
            # If token not configured, just acknowledge
            return {"ok": True, "note": "Bot token not configured; reply not sent."}

        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": reply_text,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": False,  # Allow link previews for better UX
                },
                timeout=15,
            )
        except requests.RequestException:
            # Swallow send errors to avoid retries from Telegram
            pass

        return {"ok": True}

    return app


def run() -> None:
    """Run the FastAPI server."""
    app = build_app()
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    run()
