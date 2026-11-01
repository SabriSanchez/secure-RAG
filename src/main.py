# Main CLI entrypoint for RAG system tasks.

import os
import signal
import sys
import fire
from pathlib import Path
from typing import Optional
from src.cli.evaluator import run_evaluate
from src.cli.generator import run_answer, run_answer_dataset
from src.cli.indexer import run_index
from src.cli.searcher import run_search, run_search_dataset


class RAGCLI:
    """Main CLI orchestrating indexing, search, generation, and evaluation."""

    def __init__(
        self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"
    ) -> None:
        """Initializes default paths for the RAG CLI."""
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.bm25_path = str(self.processed_dir / "bm25_index")
        self.vector_path = str(self.processed_dir / "vector_db")

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingests data/raw/ and builds the indices in data/processed/."""
        run_index(self.raw_dir, self.processed_dir, max_chunk_size)

    def search(self, query: str, k: int = 5) -> None:
        """Returns the top-k retrieved results for a given query."""
        run_search(query, k, self.bm25_path, self.vector_path)

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: Optional[str] = None,
    ) -> None:
        """Executes search on a full dataset and saves to JSON."""
        run_search_dataset(
            dataset_path, k, save_directory, self.bm25_path, self.vector_path
        )

    def answer(self, query: str, k: int = 3) -> None:
        """Answers a single query using retrieved context and an LLM."""
        run_answer(query, k, self.bm25_path, self.vector_path)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: Optional[str] = None,
    ) -> None:
        """Generates answers for an entire dataset of retrieved results."""
        run_answer_dataset(student_search_results_path, save_directory)

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Calculates recall@k comparing results against ground truth."""
        run_evaluate(student_search_results_path, dataset_path)


def main() -> None:
    signal.signal(signal.SIGINT, lambda s, f: os._exit(130))
    try:
        fire.Fire(RAGCLI)
    except Exception:
        sys.exit(1)
