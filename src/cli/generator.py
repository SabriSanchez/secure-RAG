# CLI command logic for generating LLM answers using retrieved search contexts.

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

from src.cli.cli_utils import setup_retriever
from src.indexing.security import SecuritySanitizer
from src.general_utils.models import MinimalAnswer, MinimalSource
from src.general_utils.models import StudentSearchResultsAndAnswer
from src.general_utils.models import UnansweredQuestion
from src.general_utils.text_utils import read_corpus_file

_MODEL: Any = None
_TOKENIZER: Any = None

_MAX_INPUT_TOKENS = 2000
_SAFETY_MARGIN = 16
_SEPARATOR = "\n\n---\n\n"
_SYSTEM_PROMPT = (
    "You are a technical assistant for the vLLM codebase. "
    "Answer the question using only the information in the context. "
    "If the context does not contain the answer, say that you do not "
    "know. Keep the answer short, clear and precise."
)
_USER_TEMPLATE = "Context:\n{context}\n\nQuestion: {query}"


def _load_qwen() -> Tuple[Any, Any]:
    """Lazily load and cache the Qwen LLM model and tokenizer.

    Returns:
        Tuple: A tuple containing the loaded language model
        and tokenizer instances.

    Raises:
        RuntimeError: If the model or tokenizer fails to load.
    """
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print("Loading model Qwen/Qwen3-0.6B...", file=sys.stderr)
            model_name = "Qwen/Qwen3-0.6B"
            _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            _MODEL = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f"Error loading the LLM: {e}")
    return _MODEL, _TOKENIZER


def _count_tokens(text: str, tokenizer: Any) -> int:
    """Count the tokens of a text without special tokens.

    Args:
        text (str): The text to measure.
        tokenizer (Any): Tokenizer of the generation model.

    Returns:
        int: Number of tokens in the text.
    """
    return len(tokenizer.encode(text, add_special_tokens=False))


def _render_prompt(query: str, context: str, tokenizer: Any) -> str:
    """Render the chat prompt with the system instructions.

    Thinking mode is disabled so that Qwen3 answers directly instead of
    writing its reasoning first.

    Args:
        query (str): The user question.
        context (str): The retrieved context, already sanitized.
        tokenizer (Any): Tokenizer of the generation model.

    Returns:
        str: The prompt text, ready to be tokenized.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _USER_TEMPLATE.format(context=context, query=query),
        },
    ]
    return str(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def _build_prompt(
    query: str, chunks: List[str], tokenizer: Any, max_tokens: int
) -> str:
    """Build a prompt whose context fits inside the token budget.

    The question and the template are always kept. Chunks are added in
    ranking order until the budget is full, so the best chunks are kept
    and the weakest ones are dropped. If even the first chunk does not
    fit, it is cut to the remaining budget.

    Args:
        query (str): The user question.
        chunks (List[str]): Sanitized chunk texts, best first.
        tokenizer (Any): Tokenizer of the generation model.
        max_tokens (int): Maximum number of tokens for the whole prompt.

    Returns:
        str: The final prompt.
    """
    fixed_cost = _count_tokens(_render_prompt(query, "", tokenizer), tokenizer)
    budget = max_tokens - fixed_cost - _SAFETY_MARGIN
    sep_cost = _count_tokens(_SEPARATOR, tokenizer)

    selected: List[str] = []
    used = 0
    for chunk in chunks:
        cost = _count_tokens(chunk, tokenizer)
        if selected:
            cost += sep_cost
        if used + cost > budget:
            if not selected and budget > 0:
                ids = tokenizer.encode(chunk, add_special_tokens=False)
                selected.append(tokenizer.decode(ids[:budget]))
            break
        selected.append(chunk)
        used += cost

    return _render_prompt(query, _SEPARATOR.join(selected), tokenizer)


def _generate_answer(
    model: Any, tokenizer: Any, prompt: str, max_new_tokens: int
) -> str:
    """Generate an answer for a rendered prompt.

    Args:
        model (Any): The generation model.
        tokenizer (Any): Tokenizer of the generation model.
        prompt (str): The rendered chat prompt.
        max_new_tokens (int): Maximum number of tokens to generate.

    Returns:
        str: The generated answer, without any thinking block.
    """
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False
    )
    input_len = inputs.input_ids.shape[-1]
    text = str(
        tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    )
    return text.split("</think>")[-1].strip()


def run_answer(
    query: str, k: int, bm25_path: str, vector_path: str
) -> None:
    """Retrieve context, generate an answer and print it as JSON.

    The output follows the MinimalAnswer model.

    Args:
        query (str): The search query or question to answer.
        k (int): Number of top context passages to retrieve.
        bm25_path (str): Path to the BM25 index directory.
        vector_path (str): Path to the database directory.

    Returns:
        None
    """
    try:
        if not query or not query.strip():
            raise ValueError("The query cannot be empty.")

        question = UnansweredQuestion(question=query)
        retriever = setup_retriever(bm25_path, vector_path)
        results = retriever.retrieve(query, top_k=int(k))

        sanitized_chunks = [
            SecuritySanitizer.sanitize(chunk.text)[0] for chunk in results
        ]
        model, tokenizer = _load_qwen()
        prompt = _build_prompt(
            query, sanitized_chunks, tokenizer, _MAX_INPUT_TOKENS
        )
        answer_text = _generate_answer(model, tokenizer, prompt, 150)

        answer = MinimalAnswer(
            question_id=question.question_id,
            question=query,
            retrieved_sources=[
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
                for chunk in results
            ],
            answer=answer_text,
        )
        print(answer.model_dump_json(indent=2))

    except Exception as e:
        print(f"Error generating response: {e}")


def read_source_text(source: Dict[str, Any]) -> str:
    """Read the text of a source from disk using its character range.

    Args:
        source (Dict[str, Any]): Source with file_path,
            first_character_index and last_character_index.

    Returns:
        str: The source text, or an empty string if it cannot be read.
    """
    try:
        content = read_corpus_file(source["file_path"])
        return content[source["first_character_index"]:
                       source["last_character_index"]]
    except (OSError, KeyError, TypeError):
        return ""


def run_answer_dataset(
    student_search_results_path: str,
    save_directory: Optional[str] = None,
) -> None:
    """Generate LLM answers for a dataset of search results
    and save output to JSON using pre-retrieved sources.

    Args:
        student_search_results_path (str):
            Path to the input JSON file with search results and questions.
        save_directory (Optional[str]):
            Custom directory path to save output file.
            If None, a default path based on input parent directory is used.

    Returns:
        None
    """
    try:
        results_path = Path(student_search_results_path)
        data = json.loads(results_path.read_text(encoding="utf-8"))
        search_results = data.get("search_results", [])
        k_val = data.get("k", 5)

        if not search_results:
            raise ValueError("There are no search results in the archive.")

        answered_results = []
        model, tokenizer = _load_qwen()

        for item in tqdm(search_results, desc="Generating Responses"):
            query = item.get("question", "")

            raw_sources = item.get("retrieved_sources", [])

            texts = [read_source_text(s) for s in raw_sources]
            sanitized_chunks = [
                SecuritySanitizer.sanitize(text)[0] for text in texts if text
            ]
            prompt = _build_prompt(
                query, sanitized_chunks, tokenizer, _MAX_INPUT_TOKENS
            )
            answer_text = _generate_answer(model, tokenizer, prompt, 100)

            sources = [MinimalSource(**s) for s in raw_sources]
            answered_results.append(
                MinimalAnswer(
                    question_id=item.get("question_id", ""),
                    question=query,
                    retrieved_sources=sources,
                    answer=answer_text,
                )
            )

        parent_name = results_path.parent.name
        out_dir = Path(
            save_directory
            or f"data/output/search_results_and_answer/{parent_name}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / results_path.name

        output_data = StudentSearchResultsAndAnswer(
            search_results=answered_results, k=k_val
        )
        out_file.write_text(
            output_data.model_dump_json(indent=2), encoding="utf-8"
        )
        print(f"Answers saved in: {out_file}")

    except Exception as e:
        print(f"Error processing the response dataset: {e}")
