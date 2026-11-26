"""Score manually reviewed predictions without using an LLM as the judge.

Usage: python scripts/score_claim_quality.py path/to/predictions.json
Prediction JSON: {"claims":[{"id":"valid","extraction":true,
"verification":"accepted"}], "pairs":[{"id":"exact","relationship":"exact"}]}
"""

import argparse
import json
from pathlib import Path


def scores(expected, predicted):
    tp = sum(e and p for e, p in zip(expected, predicted))
    fp = sum(not e and p for e, p in zip(expected, predicted))
    fn = sum(e and not p for e, p in zip(expected, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_acceptance_rate": fp / sum(not e for e in expected) if any(not e for e in expected) else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    fixture = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "claim_quality_labels.json"
    labels = json.loads(fixture.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    claims = {x["id"]: x for x in predictions["claims"]}
    pairs = {x["id"]: x for x in predictions["pairs"]}
    if set(claims) != {x["id"] for x in labels["claims"]} or set(pairs) != {x["id"] for x in labels["pairs"]}:
        raise ValueError("Predictions must contain exactly one entry for every labeled ID.")
    extraction = scores(
        [x["extraction_expected"] for x in labels["claims"]],
        [claims[x["id"]]["extraction"] for x in labels["claims"]],
    )
    verification = scores(
        [x["verification_expected"] == "accepted" for x in labels["claims"]],
        [claims[x["id"]]["verification"] == "accepted" for x in labels["claims"]],
    )
    duplicate = scores(
        [x["relationship"] in {"exact", "semantic"} for x in labels["pairs"]],
        [pairs[x["id"]]["relationship"] in {"exact", "semantic"} for x in labels["pairs"]],
    )
    print(json.dumps({"extraction": extraction, "verification": verification, "duplicates": duplicate}, indent=2))


if __name__ == "__main__":
    main()
