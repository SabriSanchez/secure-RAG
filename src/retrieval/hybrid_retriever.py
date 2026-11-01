# Combines lexical and vector retrieval using RRF and cross-encoder re-ranking.

from typing import Dict, List
from sentence_transformers import CrossEncoder
from src.general_utils.models import DocumentChunk
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever


class HybridRetriever:
    """Combines BM25 and vector retrieval with RRF and re-ranking."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
        rrf_k: int = 60,
        use_reranker: bool = True,
    ):
        """Initializes the HybridRetriever with BM25 and vector retrievers.

        Args:
            bm25_retriever (BM25Retriever): Instance of BM25 retriever for
                lexical search.
            vector_retriever (VectorRetriever): Instance of vector retriever
                for semantic search.
            rrf_k (int): Reciprocal Rank Fusion constant parameter.
                Defaults to 60.
            use_reranker (bool): Whether to apply cross-encoder re-ranking.
                Defaults to True.
        """
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.rrf_k = rrf_k
        self.use_reranker = use_reranker

        if self.use_reranker:
            self.reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )

    def retrieve(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """Retrieves and re-ranks top document chunks for a given query.

        Args:
            query (str): The search query string.
            top_k (int): Maximum number of top results to return.
                Defaults to 5.

        Returns:
            List[DocumentChunk]: Re-ranked list of top document chunks.
        """
        candidate_k = max(top_k * 4, 20)

        bm25_results = self.bm25_retriever.retrieve(query, top_k=candidate_k)
        vector_results = self.vector_retriever.retrieve(
            query, top_k=candidate_k
        )

        rrf_candidates = self._apply_rrf(
            bm25_results, vector_results, top_k=candidate_k
        )

        if self.use_reranker and rrf_candidates:
            return self._rerank(query, rrf_candidates, top_k=top_k)

        return rrf_candidates[:top_k]

    def _apply_rrf(
        self,
        bm25_chunks: List[DocumentChunk],
        vector_chunks: List[DocumentChunk],
        top_k: int,
    ) -> List[DocumentChunk]:
        """Merges search results using Reciprocal Rank Fusion (RRF).

        Args:
            bm25_chunks (List[DocumentChunk]): Lexical search results.
            vector_chunks (List[DocumentChunk]): Vector search results.
            top_k (int): Maximum number of candidate chunks to select.

        Returns:
            List[DocumentChunk]: Ranked list of fused candidate chunks.
        """
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}

        for rank, chunk in enumerate(bm25_chunks):
            chunk_id = chunk.chunk_id
            scores[chunk_id] = (
                scores.get(chunk_id, 0.0) + 6.0 / (self.rrf_k + rank + 1)
            )
            chunk_map[chunk_id] = chunk

        for rank, chunk in enumerate(vector_chunks):
            chunk_id = chunk.chunk_id
            scores[chunk_id] = (
                scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            )
            chunk_map[chunk_id] = chunk

        sorted_ids = sorted(
            scores.keys(), key=lambda x: scores[x], reverse=True
        )
        return [chunk_map[cid] for cid in sorted_ids[:top_k]]

    def _rerank(
        self,
        query: str,
        candidates: List[DocumentChunk],
        top_k: int,
    ) -> List[DocumentChunk]:
        """Re-ranks candidates using a CrossEncoder model.

        Args:
            query (str): The search query string.
            candidates (List[DocumentChunk]): Candidate document chunks.
            top_k (int): Number of top re-ranked chunks to return.

        Returns:
            List[DocumentChunk]: Top-K re-ranked document chunks.
        """
        pairs = [[query, chunk.text] for chunk in candidates]
        scores = self.reranker.predict(pairs)

        scored_chunks = sorted(
            zip(scores, candidates), key=lambda x: x[0], reverse=True
        )
        return [chunk for _, chunk in scored_chunks[:top_k]]
