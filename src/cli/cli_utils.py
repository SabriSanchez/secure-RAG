# Utility functions and configurations for CLI retrieval
# setup and file filtering.

from pathlib import Path

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_retriever import VectorRetriever


VALID_EXTENSIONS = {".py", ".md", ".markdown", ".rst", ".txt"}
IGNORED_DIRS = {
    ".git",
    ".venv",
    ".uv-cache",
    ".hf-cache",
    "__pycache__",
    "node_modules",
}


def should_process_file(file_path: Path) -> bool:
    """Determine whether a given file path should be processed based on rules.

    Args:
        file_path (Path): The path of the file to evaluate.

    Returns:
        bool: True if the file has an allowed extension and is not within an
            ignored directory; False otherwise.
    """
    if any(p in IGNORED_DIRS or p.startswith(".") for p in file_path.parts):
        return False
    return file_path.suffix.lower() in VALID_EXTENSIONS


def setup_retriever(
    bm25_path: str, vector_path: str, use_reranker: bool = False
) -> HybridRetriever:
    """Initialize and combine BM25 and vector retrievers
    into a hybrid retriever.

    Args:
        bm25_path (str): Path to the directory containing the BM25 index.
        vector_path (str): Path to the vector database.
        use_reranker (bool, optional):
            Whether to enable cross-encoder reranking. Defaults to True.

    Returns:
        HybridRetriever:
            An instantiated hybrid retriever combining both search engines.

    Raises:
        FileNotFoundError:
            If the specified BM25 index directory does not exist.
    """
    if not Path(bm25_path).exists():
        raise FileNotFoundError(
            f"BM25 index not found at {bm25_path}. Run 'index' first."
        )

    bm25_ret = BM25Retriever(index_dir=bm25_path)
    vec_ret = VectorRetriever(index_dir=vector_path)

    return HybridRetriever(
        bm25_retriever=bm25_ret,
        vector_retriever=vec_ret,
        use_reranker=use_reranker,
    )
