"""
Document ingestion module for processing and indexing content.

This module handles loading documents, chunking text, generating embeddings,
and storing them in ChromaDB for semantic search.
"""

import os
import uuid
from pathlib import Path
from typing import List, Tuple

import chromadb
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from .embeddings import SentenceTransformerEmbeddingFunction


load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma")
DATA_PATH = os.getenv("DATA_PATH", "./data")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docs")


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text("\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join([line for line in lines if line])


def _extract_urls_from_html(html: str, base_url: str = "") -> List[str]:
    """Extract URLs from HTML content."""
    soup = BeautifulSoup(html, "html.parser")
    urls = []
    
    # Extract links
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.startswith("http"):
            urls.append(href)
        elif href.startswith("/"):
            # Relative URL - construct full URL if base_url provided
            if base_url:
                full_url = base_url.rstrip("/") + href
                urls.append(full_url)
    
    return list(set(urls))  # Remove duplicates


def _load_documents(data_dir: Path) -> List[Tuple[str, str, List[str]]]:
    docs: List[Tuple[str, str, List[str]]] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".txt", ".md"}:
            content = _read_text_file(path)
            urls = []
        elif path.suffix.lower() in {".html", ".htm"}:
            html_content = _read_text_file(path)
            content = _extract_text_from_html(html_content)
            # Try to construct base URL from file path
            base_url = "https://witas.fi"  # Default base URL
            urls = _extract_urls_from_html(html_content, base_url)
        else:
            continue
        if content.strip():
            docs.append((str(path.relative_to(data_dir)), content, urls))
    return docs


def _chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 150) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def ingest() -> None:
    data_dir = Path(DATA_PATH)
    data_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    docs = _load_documents(data_dir)
    if not docs:
        print(f"No documents found in {data_dir}. Add .txt/.md/.html files and re-run.")
        return

    embedder = SentenceTransformerEmbeddingFunction()

    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[dict] = []
    embeddings: List[List[float]] = []

    for doc_id, content, urls in docs:
        chunks = _chunk_text(content)
        for chunk in chunks:
            ids.append(str(uuid.uuid4()))
            texts.append(chunk)
            metadatas.append({
                "source": doc_id,
                "urls": "; ".join(urls[:5]) if urls else ""  # Store up to 5 URLs as semicolon-separated string
            })

    embeddings = embedder(texts)

    batch_size = 512
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=texts[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
        )
        print(f"Indexed {min(i + batch_size, len(ids))}/{len(ids)} chunks")

    print("Ingestion complete. ChromaDB is ready.")


if __name__ == "__main__":
    ingest()
