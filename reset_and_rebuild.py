import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

# Import project modules
from app.crawl import crawl
from app.ingest import ingest


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path, ignore_errors=True)


def main() -> None:
    load_dotenv()

    chroma_path = Path(os.getenv("CHROMA_PATH", "./chroma")).resolve()
    db_path = Path(os.getenv("DB_PATH", "./conversations.db")).resolve()
    data_path = Path(os.getenv("DATA_PATH", "./data")).resolve()
    crawl_output_dir = Path(os.getenv("CRAWL_OUTPUT_DIR", "./data/witas")).resolve()

    print("Resetting data stores...")
    print(f"- Deleting ChromaDB at: {chroma_path}")
    _remove_path(chroma_path)

    print(f"- Deleting conversations DB at: {db_path}")
    _remove_path(db_path)

    # Remove both generic DATA_PATH and the crawler output dir (in case they differ)
    if crawl_output_dir != data_path:
        print(f"- Deleting crawl output at: {crawl_output_dir}")
        _remove_path(crawl_output_dir)

    print(f"- Deleting data directory at: {data_path}")
    _remove_path(data_path)

    # Recreate base data directory to avoid downstream issues
    data_path.mkdir(parents=True, exist_ok=True)

    print("\nRe-crawling website...")
    crawl()

    print("\nRebuilding vector database (ingestion)...")
    ingest()

    print("\nAll done. You can now (re)start the server: python main.py")


if __name__ == "__main__":
    main()


