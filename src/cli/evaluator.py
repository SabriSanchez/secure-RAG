# CLI execution logic for evaluating retrieval
# performance using Recall@k metrics.

import json
from pathlib import Path


def run_evaluate(student_path: str, dataset_path: str) -> None:
    """Evaluate retrieval results by calculating Recall@k
    against a ground truth dataset.

    Args:
        student_path (str):
            Path to the JSON file containing predicted search results.
        dataset_path (str):
            Path to the JSON file containing ground truth
            questions and reference sources.

    Returns:
        None
    """
    try:
        student_data = json.loads(
            Path(student_path).read_text(encoding="utf-8")
        )
        truth_data = json.loads(
            Path(dataset_path).read_text(encoding="utf-8")
        )

        student_dict = {
            item["question_id"]: item["retrieved_sources"]
            for item in student_data.get("search_results", [])
        }

        matches, evaluated = 0, 0

        for truth_q in truth_data.get("rag_questions", []):
            q_id = truth_q.get("question_id")
            truth_sources = truth_q.get("sources", [])

            if not truth_sources or q_id not in student_dict:
                continue

            evaluated += 1
            hit = False
            for ts in truth_sources:
                for ss in student_dict[q_id]:
                    if ts["file_path"] == ss["file_path"]:
                        start_max = max(
                            ts["first_character_index"],
                            ss["first_character_index"],
                        )
                        end_min = min(
                            ts["last_character_index"],
                            ss["last_character_index"],
                        )
                        if start_max < end_min:
                            hit = True
                            break
                if hit:
                    break

            if hit:
                matches += 1

        if evaluated == 0:
            print("No valid questions were found for evaluation.")
            return

        recall = matches / evaluated
        print("\nLocal Assessment (Mock Recall):")
        print(f"Questions: {evaluated}")
        print(f"Recall: {recall:.3f} ({(recall * 100):.1f}%)")

    except Exception as e:
        print(f"Error during evaluation: {e}")
