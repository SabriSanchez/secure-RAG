# CLI command logic for running hybrid retrieval
# queries and batch dataset search.

from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from src.cli.cli_utils import setup_retriever
from src.general_utils.models import MinimalSearchResults, MinimalSource
from src.general_utils.models import StudentSearchResults
from src.general_utils.parser import (
    parse_answered_questions,
    parse_unanswered_questions,
)


def run_search(
    query: str, k: int, bm25_path: str, vector_path: str
) -> None:
    """Execute a single search query using hybrid retrieval
    and print top results.

    Args:
        query (str): Search query or question to retrieve context for.
        k (int): Number of top search results to return.
        bm25_path (str): Path to the BM25 index directory.
        vector_path (str): Path to the database directory.

    Returns:
        None
    """
    try:
        if not query or not query.strip():
            raise ValueError("The query cannot be empty.")
        if int(k) <= 0:
            raise ValueError("The parameter k must be greater than 0.")

        retriever = setup_retriever(bm25_path, vector_path, use_reranker=False)
        results = retriever.retrieve(query, top_k=int(k))

        print(f"\n=== Top {len(results)} Results for: '{query}' ===")
        for rank, chunk in enumerate(results, 1):
            start = chunk.first_character_index
            end = chunk.last_character_index
            print(f"[{rank}] Archive: {chunk.file_path} ({start} -> {end})")
            snippet = chunk.text[:100].replace("\n", " ")
            print(f"    Snippet: {snippet}...\n")

    except Exception as e:
        print(f"Search error: {e}")


def run_search_dataset(
    dataset_path: str,
    k: int,
    save_directory: Optional[str],
    bm25_path: str,
    vector_path: str,
) -> None:
    """Retrieve search context for a dataset of questions and export to JSON.

    Args:
        dataset_path (str):
            Path to the input JSON file containing evaluation questions.
        k (int): Number of top context results to retrieve per question.
        save_directory (Optional[str]):
            Custom directory path to save the output file.
            If None, uses a default path based on the input file's directory.
        bm25_path (str): Path to the BM25 index directory.
        vector_path (str): Path to the database directory.

    Returns:
        None
    """
    try:
        if int(k) <= 0:
            raise ValueError("k must be greater than 0.")

        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        raw_content = path.read_text(encoding="utf-8")
        questions = parse_unanswered_questions(
            raw_content
        ) or parse_answered_questions(raw_content)

        if not questions:
            raise ValueError("No valid questions were found.")

        retriever = setup_retriever(bm25_path, vector_path, use_reranker=False)
        formatted_results = []
        failed = 0

        for q in tqdm(questions, desc=f"Searching ({path.name})"):
            sources: List[MinimalSource] = []
            try:
                if q.question.strip():
                    hits = retriever.retrieve(q.question, top_k=int(k))
                    sources = [
                        MinimalSource(
                            file_path=chunk.file_path,
                            first_character_index=chunk.first_character_index,
                            last_character_index=chunk.last_character_index,
                        )
                        for chunk in hits
                    ]
            except Exception as e:
                failed += 1
                tqdm.write(
                    f"Warning: search failed for question "
                    f"'{q.question_id}': {e}"
                )
            formatted_results.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=sources,
                )
            )

        out_dir = Path(
            save_directory or f"data/output/search_results/{path.parent.name}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / path.name

        output_data = StudentSearchResults(
            search_results=formatted_results, k=int(k)
        )
        out_file.write_text(
            output_data.model_dump_json(indent=2), encoding="utf-8"
        )
        if failed:
            print(f"Warning: {failed} question(s) failed and have no sources.")
        print(f"Results saved in: {out_file}")

    except Exception as e:
        print(f"Error processing search dataset: {e}")
