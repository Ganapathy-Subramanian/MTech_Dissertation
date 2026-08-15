import argparse
import json
import os
from typing import List, Tuple, Dict, Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import sys

# Add parent directory to path for comprehensive_evaluation import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from comprehensive_evaluation import ComprehensiveEvaluator, force_to_10_categories
    from business_evaluation_report import BusinessEvaluationReport
    HAS_COMPREHENSIVE = True
except ImportError:
    HAS_COMPREHENSIVE = False

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DEFAULT_DATASET_PATH = os.path.join(MODELS_DIR, "bitext_insurance_mapped.json")


def load_dataset(dataset_path: str = DEFAULT_DATASET_PATH) -> List[Tuple[str, str]]:
    """Load the same training data source used by EnhancedTriageModel."""
    import sys

    sys.path.insert(0, ROOT_DIR)
    from models.enhanced_triage import EnhancedTriageModel

    triage_stub = EnhancedTriageModel.__new__(EnhancedTriageModel)
    triage_stub.base_dir = MODELS_DIR
    triage_stub.labels = getattr(EnhancedTriageModel, "LABELS", [])
    data = triage_stub._get_training_data()
    return [(text, label) for text, label in data if text and label]


def create_fixed_split(
    data: List[Tuple[str, str]],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    texts = [text for text, _ in data]
    labels = [label for _, label in data]

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    train_pairs = list(zip(X_train, y_train))
    test_pairs = list(zip(X_test, y_test))
    return train_pairs, test_pairs


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=8000,
                    ngram_range=(1, 3),
                    stop_words="english",
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    C=2.0,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train_and_evaluate(
    train_pairs: List[Tuple[str, str]],
    test_pairs: List[Tuple[str, str]],
    random_state: int = 42,
) -> Dict[str, Any]:
    train_texts = [text for text, _ in train_pairs]
    train_labels = [label for _, label in train_pairs]
    test_texts = [text for text, _ in test_pairs]
    test_labels = [label for _, label in test_pairs]

    pipeline = build_pipeline()
    pipeline.fit(train_texts, train_labels)
    predictions = pipeline.predict(test_texts)

    return {
        "train_size": len(train_pairs),
        "test_size": len(test_pairs),
        "random_state": random_state,
        "accuracy": round(float(accuracy_score(test_labels, predictions)), 4),
        "weighted_f1": round(float(f1_score(test_labels, predictions, average="weighted")), 4),
        "macro_f1": round(float(f1_score(test_labels, predictions, average="macro")), 4),
        "predictions": predictions,
        "pipeline": pipeline,  # Added for confidence score extraction
        "test_labels": test_labels,
    }


def evaluate_external_predictions(
    test_pairs: List[Tuple[str, str]],
    predictions: List[str],
    name: str = "external_model",
) -> Dict[str, Any]:
    actual_labels = [label for _, label in test_pairs]
    return {
        "name": name,
        "test_size": len(test_pairs),
        "accuracy": round(float(accuracy_score(actual_labels, predictions)), 4),
        "weighted_f1": round(float(f1_score(actual_labels, predictions, average="weighted")), 4),
        "macro_f1": round(float(f1_score(actual_labels, predictions, average="macro")), 4),
    }


def _normalize_column_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _pick_prediction_column(columns: List[str]) -> str:
    normalized_columns = {_normalize_column_name(col): col for col in columns}
    for candidate in [
        "predicted_label",
        "predicted_category",
        "prediction",
        "predicted",
        "label",
        "category",
        "actual",
        "expected",
    ]:
        if candidate in normalized_columns:
            return normalized_columns[candidate]
    raise ValueError("Could not find a prediction column. Expected one of: predicted_label, predicted_category, prediction, label, category")


def _pick_query_column(columns: List[str]) -> str:
    normalized_columns = {_normalize_column_name(col): col for col in columns}
    for candidate in ["query", "text", "input", "sentence", "prompt"]:
        if candidate in normalized_columns:
            return normalized_columns[candidate]
    raise ValueError("Could not find a query column. Expected one of: query, text, input, sentence, prompt")


def _normalize_query(value: str) -> str:
    return str(value).strip().lower()


def load_predictions_from_file(path: str, reference_queries: List[str] | None = None) -> Tuple[List[str], List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        if reference_queries is not None:
            query_col = _pick_query_column(df.columns.tolist())
            pred_col = _pick_prediction_column(df.columns.tolist())
            query_map = {
                _normalize_query(row[query_col]): str(row[pred_col])
                for _, row in df.iterrows()
                if str(row[query_col]).strip() != ""
            }

            predictions = []
            missing_queries = []
            for query in reference_queries:
                key = _normalize_query(query)
                if key in query_map:
                    predictions.append(query_map[key])
                else:
                    missing_queries.append(query)
            return predictions, missing_queries

        pred_col = _pick_prediction_column(df.columns.tolist())
        return [str(x) for x in df[pred_col].tolist()], []

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return [str(x) for x in payload], []
        if isinstance(payload, dict) and "predictions" in payload:
            return [str(x) for x in payload["predictions"]], []
        raise ValueError("JSON must be a list or contain a 'predictions' key")

    raise ValueError("Only .csv or .json files are supported")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair train/test evaluation for the insurance triage model")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Path to the mapped Hugging Face JSON file")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for the held-out test set")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for the split")
    parser.add_argument("--baseline-file", default=None, help="Optional CSV/JSON file with external predictions")
    parser.add_argument("--baseline-name", default="external_model", help="Label for the external comparison")
    parser.add_argument("--comprehensive", action="store_true", help="Generate comprehensive evaluation with business metrics")
    parser.add_argument("--output-dir", default="eval_output", help="Output directory for evaluation results")
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    train_pairs, test_pairs = create_fixed_split(data, test_size=args.test_size, random_state=args.random_state)

    result = train_and_evaluate(train_pairs, test_pairs, random_state=args.random_state)

    print("Fair evaluation setup")
    print("-" * 60)
    print(f"Dataset size        : {len(data):,}")
    print(f"Train size          : {result['train_size']:,}")
    print(f"Test size           : {result['test_size']:,}")
    print(f"Split ratio         : {args.test_size}")
    print(f"Random seed         : {args.random_state}")
    print("-" * 60)
    print("Current project model")
    print(f"Accuracy            : {result['accuracy']:.4f}")
    print(f"Weighted F1         : {result['weighted_f1']:.4f}")
    print(f"Macro F1            : {result['macro_f1']:.4f}")
    print("")

    # ── Comprehensive Evaluation (with business metrics) ──────────────────
    if args.comprehensive and HAS_COMPREHENSIVE:
        print("=" * 60)
        print("COMPREHENSIVE EVALUATION (STRICT 10-CLASS + BUSINESS METRICS)")
        print("=" * 60)
        print()
        
        # Get predictions and ground truth
        y_true = [label for _, label in test_pairs]
        y_pred_raw = result["predictions"]
        test_queries = [text for text, _ in test_pairs]
        
        # Force predictions to 10 fixed categories
        y_pred = force_to_10_categories(y_pred_raw)
        
        # Get confidence scores if available
        pipeline = result.get("pipeline")
        confidence_scores = None
        if pipeline:
            try:
                proba = pipeline.predict_proba(test_queries)
                confidence_scores = proba.max(axis=1).tolist()
            except:
                confidence_scores = None
        
        # Run comprehensive evaluation
        evaluator = ComprehensiveEvaluator(
            y_true=y_true,
            y_pred=y_pred,
            confidence_scores=confidence_scores or [0.5] * len(y_pred),
            test_queries=test_queries,
            complex_query_indices=None
        )
        
        # Generate report
        report = evaluator.generate_report()
        print(report)
        print()
        
        # Save outputs
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save JSON results
        evaluator.export_to_json(os.path.join(args.output_dir, "comprehensive_evaluation.json"))
        
        # Save text report
        with open(os.path.join(args.output_dir, "evaluation_report.txt"), "w") as f:
            f.write(report)
        
        # Generate business evaluation report
        dataset_info = {
            "name": os.path.basename(args.dataset),
            "total_size": len(data),
            "num_classes": 10,
            "train_test_split": f"{int((1-args.test_size)*100)}/{int(args.test_size*100)}",
            "train_size": result['train_size'],
            "test_size": result['test_size'],
            "random_seed": args.random_state,
        }
        
        comp_results = evaluator.evaluate()
        bus_report = BusinessEvaluationReport(comp_results, dataset_info)
        bus_report_text = bus_report.generate_full_report(
            os.path.join(args.output_dir, "Business_Evaluation_Report.md")
        )
        
        print("\n" + "=" * 60)
        print("Business evaluation report saved to:")
        print(f"  {args.output_dir}/Business_Evaluation_Report.md")
        print("=" * 60)
        print()
    
    elif args.comprehensive and not HAS_COMPREHENSIVE:
        print("⚠ Comprehensive evaluation requested but modules not available")
        print("  Ensure comprehensive_evaluation.py is in root directory")
        print()

    if args.baseline_file:
        reference_queries = [text for text, _ in test_pairs]
        baseline_predictions, missing_queries = load_predictions_from_file(args.baseline_file, reference_queries=reference_queries)
        if not baseline_predictions:
            print("Note: no baseline predictions were loaded from the supplied file; skipping baseline comparison")
            return

        matched_test_pairs = []
        matched_predictions = []
        test_label_lookup = {_normalize_query(text): label for text, label in test_pairs}
        for query, prediction in zip(reference_queries, baseline_predictions):
            label = test_label_lookup.get(_normalize_query(query))
            if label is None:
                continue
            matched_test_pairs.append((query, label))
            matched_predictions.append(prediction)

        if len(matched_test_pairs) != len(matched_predictions):
            raise ValueError("Internal mismatch while aligning baseline predictions")

        if len(missing_queries) > 0:
            print(f"Note: {len(missing_queries)} held-out queries had no baseline prediction in the supplied file")

        if not matched_test_pairs and len(baseline_predictions) == len(test_pairs):
            matched_test_pairs = list(test_pairs)
            matched_predictions = baseline_predictions
            print("Note: using row-order alignment because no query-based matches were found")

        if not matched_test_pairs:
            print("Note: no baseline predictions could be aligned to the held-out queries; skipping baseline comparison")
        else:
            baseline_result = evaluate_external_predictions(matched_test_pairs, matched_predictions, args.baseline_name)
            print("-" * 60)
            print(f"{args.baseline_name}")
            print(f"Evaluated samples   : {len(matched_test_pairs)}")
            print(f"Accuracy            : {baseline_result['accuracy']:.4f}")
            print(f"Weighted F1         : {baseline_result['weighted_f1']:.4f}")
            print(f"Macro F1            : {baseline_result['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
