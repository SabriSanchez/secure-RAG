import pickle
import torch
from pathlib import Path
from typing import List
from src.general_utils.models import DocumentChunk
from sentence_transformers import (
    SentenceTransformer
)


class VectorRetriever:
    """Retrieves relevant document chunks using vector embedding similarity."""
    def __init__(self, index_dir: str = "data/processed/vector_index") -> None:
        """Initialize the retriever with model, embeddings, and metadata.
        Args:
            index_dir: Path to the directory containing saved index files.
        """
        self.model = (
            SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        )
        self.model.max_seq_length = 128
        self.embeddings = torch.load(Path(index_dir) / "embeddings.pt")
        with open(Path(index_dir) / "chunks_metadata.pkl", "rb") as f:
            self.chunks = pickle.load(f)

    def retrieve(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """Retrieve the top-k most similar chunks for a given query.

        Args:
            query: The search query string.
            top_k: Number of top results to return.

        Returns:
            A list of the top-k matching chunk objects.
        """
        with torch.inference_mode():
            query_emb = self.model.encode(
                query,
                convert_to_tensor=True,
                normalize_embeddings=True
            )

            scores = torch.mv(self.embeddings, query_emb)
            top_results = torch.topk(scores, k=min(top_k, len(self.chunks)))

        return [self.chunks[idx] for idx in top_results.indices]
