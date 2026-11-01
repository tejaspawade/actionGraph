
import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

from app.services.meeting_service import MeetingService


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "data" / "evaluations" / "evaluation_dataset.json"
SIMILARITY_THRESHOLD = 0.65

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def normalize_action(action: str) -> str:
    """Normalize action text before generating embeddings."""
    action = action.casefold()
    action = re.sub(r"[^\w\s]", "", action)
    return " ".join(action.split())


def match_actions(expected: list, predicted: list):
    """
    Match actions one-to-one using semantic similarity.

    This function must be called separately for each meeting.
    Returns:
        matches: (expected_index, predicted_index, similarity)
        unmatched_expected: indices of unmatched expected actions
        unmatched_predicted: indices of unmatched predictions
    """
    if not expected or not predicted:
        return (
            [],
            set(range(len(expected))),
            set(range(len(predicted))),
        )

    expected_texts = [
        normalize_action(item["action"]) for item in expected
    ]
    predicted_texts = [
        normalize_action(item.action) for item in predicted
    ]

    expected_embeddings = embedding_model.encode(
        expected_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    predicted_embeddings = embedding_model.encode(
        predicted_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    similarities = util.cos_sim(
        expected_embeddings, predicted_embeddings
    )

    # Keep only candidate pairs that meet the similarity threshold.
    candidates = []
    for expected_index in range(len(expected)):
        for predicted_index in range(len(predicted)):
            score = similarities[expected_index, predicted_index].item()

            if score >= SIMILARITY_THRESHOLD:
                candidates.append(
                    (score, expected_index, predicted_index)
                )

    # Greedy one-to-one matching: highest similarity first.
    candidates.sort(key=lambda item: item[0], reverse=True)

    matches = []
    matched_expected = set()
    matched_predicted = set()

    for score, expected_index, predicted_index in candidates:
        if expected_index in matched_expected:
            continue
        if predicted_index in matched_predicted:
            continue

        matches.append((expected_index, predicted_index, score))
        matched_expected.add(expected_index)
        matched_predicted.add(predicted_index)

    unmatched_expected = (
        set(range(len(expected))) - matched_expected
    )
    unmatched_predicted = (
        set(range(len(predicted))) - matched_predicted
    )

    return matches, unmatched_expected, unmatched_predicted


def calculate_metrics(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(f"True positives:  {metrics['tp']}")
    print(f"False positives: {metrics['fp']}")
    print(f"False negatives: {metrics['fn']}")
    print(f"Precision:       {metrics['precision']:.2%}")
    print(f"Recall:          {metrics['recall']:.2%}")
    print(f"F1-score:        {metrics['f1']:.2%}")


def main():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    meeting_service = MeetingService()

    overall_tp = overall_fp = overall_fn = 0
    explicit_tp = explicit_fp = explicit_fn = 0
    implicit_tp = implicit_fp = implicit_fn = 0

    type_correct = 0
    type_incorrect = 0

    for meeting in dataset:
        meeting_id = meeting["meeting_id"]
        transcript = meeting["transcript"]
        expected = meeting["expected_actions"]

        print(f"\n{'=' * 12} {meeting_id} {'=' * 12}")

        result = meeting_service.extract_action_items(transcript)
        predicted = result.action_items

        print(f"Expected actions: {len(expected)}")
        print(f"Predicted actions: {len(predicted)}")

        for item in predicted:
            print(f"  - [{item.type}] {item.action}")

        # Overall matching happens within this meeting only.
        matches, unmatched_expected, unmatched_predicted = (
            match_actions(expected, predicted)
        )

        print("\nSemantic matches:")
        for expected_index, predicted_index, score in matches:
            expected_item = expected[expected_index]
            predicted_item = predicted[predicted_index]

            print(
                f"  MATCH ({score:.3f}): "
                f"{expected_item['action']} <-> "
                f"{predicted_item.action}"
            )

            if expected_item["type"] == predicted_item.type:
                type_correct += 1
            else:
                type_incorrect += 1
                print(
                    "    TYPE MISMATCH: expected "
                    f"{expected_item['type']}, predicted "
                    f"{predicted_item.type}"
                )

        for index in unmatched_expected:
            print(f"  MISSED: {expected[index]['action']}")

        for index in unmatched_predicted:
            print(f"  EXTRA: {predicted[index].action}")

        overall_tp += len(matches)
        overall_fp += len(unmatched_predicted)
        overall_fn += len(unmatched_expected)

        # Class-specific metrics require both semantic and type agreement.
        for action_type in ("explicit", "implicit"):
            typed_expected = [
                item for item in expected
                if item["type"] == action_type
            ]
            typed_predicted = [
                item for item in predicted
                if item.type == action_type
            ]

            typed_matches, typed_unmatched_expected, (
                typed_unmatched_predicted
            ) = match_actions(typed_expected, typed_predicted)

            tp = len(typed_matches)
            fp = len(typed_unmatched_predicted)
            fn = len(typed_unmatched_expected)

            if action_type == "explicit":
                explicit_tp += tp
                explicit_fp += fp
                explicit_fn += fn
            else:
                implicit_tp += tp
                implicit_fp += fp
                implicit_fn += fn

    print("\n\n========== AGGREGATE RESULTS ==========")

    print_metrics(
        "Overall semantic action extraction",
        calculate_metrics(overall_tp, overall_fp, overall_fn),
    )

    print_metrics(
        "Explicit action extraction",
        calculate_metrics(explicit_tp, explicit_fp, explicit_fn),
    )

    print_metrics(
        "Implicit action extraction",
        calculate_metrics(implicit_tp, implicit_fp, implicit_fn),
    )

    total_typed_matches = type_correct + type_incorrect

    print("\nAction type classification")
    print("--------------------------")
    print(f"Correct types:   {type_correct}")
    print(f"Incorrect types: {type_incorrect}")

    type_accuracy = (
        type_correct / total_typed_matches
        if total_typed_matches
        else 0.0
    )
    print(f"Type accuracy:   {type_accuracy:.2%}")
    print(
        "Note: Type accuracy is calculated only for semantically "
        "matched action pairs."
    )


if __name__ == "__main__":
    main()
