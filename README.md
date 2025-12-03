# Telegram RAG Chatbot

An intelligent Telegram chatbot powered by Retrieval-Augmented Generation (RAG) that provides contextual responses by searching through a knowledge base of crawled website content. Built with FastAPI, ChromaDB, and Google Gemini AI.

## Overview

This project implements a production-ready chatbot system that:

- Crawls and indexes website content into a vector database
- Uses semantic search to retrieve relevant context
- Generates intelligent responses using Google's Gemini AI
- Maintains conversation history for context-aware interactions
- Integrates seamlessly with Telegram via webhooks

## Features

- **Web Crawling**: Automated crawling of websites with configurable depth and domain restrictions
- **Vector Search**: Semantic search using ChromaDB with sentence transformers for accurate context retrieval
- **RAG Architecture**: Combines retrieved context with LLM generation for accurate, source-backed responses
- **Conversation Memory**: SQLite-based conversation history for maintaining context across sessions
- **Telegram Integration**: Native Telegram bot support with webhook-based message handling
- **RESTful API**: FastAPI endpoints for programmatic access and testing
- **Production Ready**: Environment-based configuration, error handling, and scalable architecture

## Architecture

```
┌─────────────┐
│   Telegram  │
│     Bot     │
└──────┬──────┘
       │ Webhook
       ▼
┌─────────────────┐
│   FastAPI       │
│   Server        │
└──────┬──────────┘
       │
       ├──► ChromaDB (Vector Store)
       ├──► SQLite (Conversations)
       └──► Google Gemini API
```

### Components

1. **Crawler** (`app/crawl.py`): Extracts HTML content from websites
2. **Ingestion** (`app/ingest.py`): Processes documents, chunks text, and creates embeddings
3. **Embeddings** (`app/embeddings.py`): Thread-safe sentence transformer wrapper
4. **Server** (`app/server.py`): FastAPI application with RAG logic and Telegram webhook

## Prerequisites

- Python 3.11 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))
- Telegram Bot Token ([Create via @BotFather](https://core.telegram.org/bots/tutorial))

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd telegram-chat-bot
   ```
2. **Create a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
4. **Configure environment variables**

   ```bash
   cp ENV_EXAMPLE.txt .env
   ```

   Edit `.env` and set:

   - `GEMINI_API_KEY`: Your Google Gemini API key
   - `TELEGRAM_BOT_TOKEN`: Your Telegram bot token (optional, for Telegram integration)
   - `TELEGRAM_WEBHOOK_SECRET`: A random string for webhook security (optional)

## Usage

### Initial Setup

1. **Crawl the target website**

   ```bash
   python -m app.crawl
   ```

   This will crawl the website specified in `CRAWL_BASE_URL` (default: https://witas.fi/) and save HTML files to the data directory.
2. **Ingest content into vector database**

   ```bash
   python -m app.ingest
   ```

   This processes the crawled content, creates text chunks, generates embeddings, and stores them in ChromaDB.
3. **Reset and rebuild** (optional)

   ```bash
   python reset_and_rebuild.py
   ```

   This script clears all data stores and rebuilds from scratch.

### Running the Server

```bash
python main.py
```

The server will start on `http://0.0.0.0:8000` (or the port specified in your `.env`).

### API Endpoints

- `GET /health` - Health check endpoint
- `GET /api/ask?question=<text>&session_id=<id>` - Ask a question via GET
- `POST /api/ask` - Ask a question via POST
  ```json
  {
    "question": "What services do you offer?",
    "session_id": "user123"
  }
  ```
- `POST /chat` - Simplified chat endpoint
  ```json
  {
    "message": "Hello",
    "session_id": "user123"
  }
  ```
- `POST /telegram/webhook/<secret>` - Telegram webhook endpoint

### Telegram Integration

1. **Set up webhook** (replace with your domain and secret):

   ```bash
   curl -X POST "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/setWebhook" \
     -d "url=https://your-domain.com/telegram/webhook/<YOUR_SECRET>"
   ```
2. **Test locally** using a tool like [ngrok](https://ngrok.com/):

   ```bash
   ngrok http 8000
   # Use the ngrok URL in the webhook setup
   ```

## Configuration

All configuration is done via environment variables in `.env`:

| Variable                    | Description                        | Default                |
| --------------------------- | ---------------------------------- | ---------------------- |
| `GEMINI_API_KEY`          | Google Gemini API key              | Required               |
| `GEMINI_MODEL`            | Gemini model to use                | `gemini-pro`         |
| `TELEGRAM_BOT_TOKEN`      | Telegram bot token                 | Optional               |
| `TELEGRAM_WEBHOOK_SECRET` | Webhook path secret                | Optional               |
| `CHROMA_PATH`             | ChromaDB storage path              | `./chroma`           |
| `DATA_PATH`               | Data directory for crawled content | `./data`             |
| `DB_PATH`                 | SQLite database path               | `./conversations.db` |
| `COLLECTION_NAME`         | ChromaDB collection name           | `docs`               |
| `HOST`                    | Server host                        | `0.0.0.0`            |
| `PORT`                    | Server port                        | `8000`               |
| `CRAWL_BASE_URL`          | Base URL to crawl                  | `https://witas.fi/`  |
| `CRAWL_MAX_PAGES`         | Maximum pages to crawl             | `60`                 |
| `CRAWL_SAME_DOMAIN_ONLY`  | Only crawl same domain             | `true`               |
| `CRAWL_DELAY_SECONDS`     | Delay between requests             | `0.5`                |

## Project Structure

```
telegram-chat-bot/
├── app/
│   ├── __init__.py          # Package initialization
│   ├── crawl.py             # Web crawler module
│   ├── ingest.py            # Document ingestion and vectorization
│   ├── embeddings.py        # Embedding function wrapper
│   └── server.py            # FastAPI application and RAG logic
├── main.py                  # Application entry point
├── reset_and_rebuild.py     # Utility script for data reset
├── requirements.txt         # Python dependencies
├── ENV_EXAMPLE.txt          # Environment variable template
├── general-info.txt         # Static contact information
└── README.md                # This file
```

## How It Works

1. **Content Collection**: The crawler visits pages on the target website, extracts HTML, and saves it locally.
2. **Processing**: The ingestion module:

   - Extracts text from HTML (removes scripts, styles)
   - Splits content into semantic chunks (1200 words with 150-word overlap)
   - Generates embeddings using sentence transformers
   - Stores everything in ChromaDB with metadata (source URLs, file paths)
3. **Query Processing**: When a user asks a question:

   - The question is embedded using the same model
   - ChromaDB performs cosine similarity search to find relevant chunks
   - Top 5 most relevant chunks are retrieved
   - Conversation history is loaded from SQLite
   - A prompt is constructed with context, history, and system instructions
   - Google Gemini generates a response
   - The conversation is saved to SQLite
4. **Response Delivery**: The response is sent back via Telegram or returned via API.

## Technologies Used

- **FastAPI**: Modern, fast web framework for building APIs
- **ChromaDB**: Open-source vector database for embeddings
- **Sentence Transformers**: State-of-the-art sentence embeddings
- **Google Gemini**: Large language model for text generation
- **SQLite**: Lightweight database for conversation storage
- **BeautifulSoup4**: HTML parsing and text extraction
- **Uvicorn**: ASGI server for FastAPI

## Development

### Running Tests

Currently, the project uses manual testing via the API endpoints. You can test locally:

```bash
# Start the server
python main.py

# In another terminal, test the API
curl "http://localhost:8000/api/ask?question=What%20services%20do%20you%20offer?"
```

### Code Style

The project follows PEP 8 style guidelines. Consider using:

- `black` for code formatting
- `flake8` or `pylint` for linting
- `mypy` for type checking

## Troubleshooting

**Issue**: "GEMINI_API_KEY is not set"

- **Solution**: Ensure your `.env` file exists and contains a valid `GEMINI_API_KEY`

**Issue**: No documents found during ingestion

- **Solution**: Run the crawler first (`python -m app.crawl`) to populate the data directory

**Issue**: Telegram webhook not receiving messages

- **Solution**: Verify the webhook URL is correct and accessible. Check that `TELEGRAM_BOT_TOKEN` is set correctly.

**Issue**: Poor response quality

- **Solution**: Ensure the vector database is properly populated. Try resetting and rebuilding with `python reset_and_rebuild.py`

## Future Enhancements

- [ ] Add support for multiple knowledge bases
- [ ] Implement streaming responses for better UX
- [ ] Add authentication for API endpoints
- [ ] Support for file uploads (PDFs, docs) for knowledge base
- [ ] Admin dashboard for monitoring and analytics
- [ ] Multi-language support
- [ ] Rate limiting and usage tracking

## Author

Developed as part of a portfolio project demonstrating expertise in:

- RAG (Retrieval-Augmented Generation) systems
- Vector databases and semantic search
- API development with FastAPI
- Telegram bot integration
- Web scraping and data processing

---

For questions or issues, please open an issue on the repository.
