# CLI command logic for document ingestion, chunking, and index construction.

from pathlib import Path
from tqdm import tqdm

from src.cli.cli_utils import should_process_file
from src.indexing.bm25_indexer import BM25Indexer
from src.indexing.chunker import CodeAndDocChunker
from src.indexing.vector_indexer import VectorIndexer


def run_index(
    raw_dir: Path, processed_dir: Path, max_chunk_size: int
) -> None:
    """Ingest raw documents, split them into chunks, and build search indices.

    Args:
        raw_dir (Path): Path to the directory containing raw files to index.
        processed_dir (Path):
            Path to the output directory where generated indices
            (BM25 and ChromaDB) will be stored.
        max_chunk_size (int):
            Maximum allowable character count per text chunk.

    Returns:
        None
    """
    try:
        try:
            max_chunk_size = int(max_chunk_size)
        except (ValueError, TypeError):
            print(f"Error: max_chunk_size must be an integer,"
                  f"got '{max_chunk_size}'.")
            return
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be greater than 0.")

        files = [
            f for f in raw_dir.rglob("*")
            if f.is_file() and should_process_file(f)
        ]
        if not files:
            print(f"Error: No files were found in {raw_dir}")
            return

        chunker = CodeAndDocChunker(max_chunk_size=max_chunk_size)
        all_chunks = []

        for f in tqdm(files, desc="Generating chunks"):
            try:
                all_chunks.extend(chunker.chunk_file(f))
            except Exception:
                continue

        print(f"Indexing {len(all_chunks)} chunks...")

        bm25_path = str(processed_dir / "bm25_index")
        bm25_idx = BM25Indexer(save_dir=bm25_path)
        bm25_idx.build_index(all_chunks)

        vector_path = str(processed_dir / "vector_db")
        vec_idx = VectorIndexer(save_dir=vector_path)
        vec_idx.build_index(all_chunks)

        print(f"Indexing successfully completed on {processed_dir}")

    except Exception as e:
        print(f"Fatal error during indexing: {e}")
