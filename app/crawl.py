import os
import time
import urllib.parse
from collections import deque
from pathlib import Path
from typing import Set

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("CRAWL_BASE_URL", "https://witas.fi/")
OUTPUT_DIR = Path(os.getenv("CRAWL_OUTPUT_DIR", "./data/witas"))
MAX_PAGES = int(os.getenv("CRAWL_MAX_PAGES", "60"))
SAME_DOMAIN_ONLY = os.getenv("CRAWL_SAME_DOMAIN_ONLY", "true").lower() == "true"
DELAY_SECONDS = float(os.getenv("CRAWL_DELAY_SECONDS", "0.5"))


def normalize_url(url: str) -> str:
    url = urllib.parse.urldefrag(url)[0]
    if url.endswith("/"):
        url = url[:-1]
    return url


def is_same_domain(url: str, base_netloc: str) -> bool:
    return urllib.parse.urlparse(url).netloc == base_netloc


def fetch(url: str) -> str | None:
    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "WitasCrawler/0.1 (+github.com/example)"
        })
        if resp.status_code == 200 and "text/html" in resp.headers.get("Content-Type", ""):
            return resp.text
        return None
    except requests.RequestException:
        return None


def save_html(base_url: str, url: str, html: str) -> None:
    # Remove the base URL to get relative path
    if url.startswith(base_url):
        rel = url[len(base_url):].strip("/")
    else:
        rel = url.strip("/")
    
    # Handle root URL case
    if not rel or rel == "":
        rel = "index"
    
    # Clean up the path and ensure it's safe
    rel = rel.replace("//", "/").strip("/")
    if not rel:
        rel = "index"
    
    out_path = OUTPUT_DIR / f"{rel}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        out_path.write_text(html, encoding="utf-8", errors="ignore")
        print(f"Saved to: {out_path}")
    except Exception as e:
        print(f"Failed to save {url} to {out_path}: {e}")


def crawl() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parsed_base = urllib.parse.urlparse(BASE_URL)
    base_origin = f"{parsed_base.scheme}://{parsed_base.netloc}"

    queue: deque[str] = deque([normalize_url(BASE_URL)])
    seen: Set[str] = set()
    pages = 0

    while queue and pages < MAX_PAGES:
        url = queue.popleft()
        if url in seen:
            continue
        seen.add(url)

        html = fetch(url)
        if not html:
            continue

        save_html(base_origin, url, html)
        pages += 1
        print(f"Saved: {url}")
        time.sleep(DELAY_SECONDS)

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = urllib.parse.urljoin(url, a["href"])  # resolve relative
            href = normalize_url(href)
            if not href.startswith(base_origin):
                if SAME_DOMAIN_ONLY:
                    continue
            if any(href.endswith(ext) for ext in [".pdf", ".jpg", ".png", ".gif", ".zip", ".svg"]):
                continue
            if href not in seen:
                queue.append(href)

    print(f"Crawled {pages} pages -> {OUTPUT_DIR}")


if __name__ == "__main__":
    crawl()
