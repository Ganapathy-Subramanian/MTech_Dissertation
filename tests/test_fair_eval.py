import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.fair_eval import create_fixed_split, evaluate_external_predictions, load_dataset


def test_split_and_metrics():
    data = load_dataset()
    assert len(data) > 100

    train_pairs, test_pairs = create_fixed_split(data, test_size=0.2, random_state=42)
    assert len(train_pairs) + len(test_pairs) == len(data)
    assert len(test_pairs) > 0

    metrics = evaluate_external_predictions(test_pairs, ["General Inquiry"] * len(test_pairs), "dummy")
    assert metrics["test_size"] == len(test_pairs)
    assert 0.0 <= metrics["accuracy"] <= 1.0
