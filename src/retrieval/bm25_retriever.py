# Performs lexical sparse retrieval over a persisted BM25 index.
import bm25s
from typing import List
from src.general_utils.models import DocumentChunk
from src.general_utils.text_utils import clean_code_text


class BM25Retriever:
    """Retrieves relevant document chunks using a sparse BM25 search index."""

    def __init__(self, index_dir: str = "data/processed/bm25_index"):
        """Initializes the BM25Retriever by loading a persisted index.

        Args:
            index_dir (str): Directory path containing the saved BM25 index
                and corpus. Defaults to "data/processed/bm25_index".
        """
        self.retriever = bm25s.BM25.load(index_dir, load_corpus=True)

    def retrieve(self, query: str, top_k: int = 20) -> List[DocumentChunk]:
        """Retrieves top-K matching document chunks for a given query string.

        Args:
            query (str): The search query text.
            top_k (int): Maximum number of matching chunks to return.
                Defaults to 20.

        Returns:
            List[DocumentChunk]: List of retrieved document chunks matching
                the query.
        """
        cleaned_query = clean_code_text(query)
        query_tokens = bm25s.tokenize(cleaned_query)
        k_val = min(top_k, len(self.retriever.corpus))
        results, _ = self.retriever.retrieve(query_tokens, k=k_val)

        chunk_objects = []
        for raw_item in results[0].tolist():
            c_id = raw_item.get("chunk_id") or (
                f"{raw_item['file_path']}_"
                f"{raw_item['first_character_index']}"
            )

            chunk = DocumentChunk(
                chunk_id=c_id,
                file_path=raw_item["file_path"],
                text=raw_item["text"],
                first_character_index=raw_item["first_character_index"],
                last_character_index=raw_item["last_character_index"],
            )
            chunk_objects.append(chunk)

        return chunk_objects
