#!/usr/bin/env python3
"""
run_comprehensive_evaluation.py
================================
End-to-end evaluation workflow for dissertation.

This script:
1. Loads the enhanced triage model and test data
2. Generates 10-class predictions (forced, no catch-all)
3. Runs comprehensive evaluation (math + business metrics)
4. Generates business evaluation report
5. Saves all outputs to eval_output/

Usage:
    python run_comprehensive_evaluation.py
    python run_comprehensive_evaluation.py --model bert  # Use DistilBERT if available
    python run_comprehensive_evaluation.py --output-dir my_results/
"""

import os
import sys
import argparse
import json
from typing import List, Tuple, Dict, Any
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from comprehensive_evaluation import ComprehensiveEvaluator, force_to_10_categories, FIXED_CATEGORIES
from business_evaluation_report import BusinessEvaluationReport
from scripts.fair_eval import load_dataset, create_fixed_split, build_pipeline
from models.enhanced_triage import EnhancedTriageModel


def load_and_evaluate(model_type: str = "tfidf", output_dir: str = "eval_output") -> Dict[str, Any]:
    """
    Load model and run complete evaluation.
    
    Args:
        model_type: "tfidf" (Phase-1, lightweight) or "bert" (Phase-2, if available)
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing all evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("COMPREHENSIVE INSURANCE TRIAGE EVALUATION")
    print("=" * 80)
    print()
    
    # ── STEP 1: Load Dataset ────────────────────────────────────────────────
    print("[1/5] Loading dataset...")
    data_path = os.path.join(PROJECT_ROOT, "models", "bitext_insurance_mapped.json")
    data = load_dataset(data_path)
    print(f"  ✓ Loaded {len(data):,} samples")
    print()
    
    # ── STEP 2: Create Train/Test Split ─────────────────────────────────────
    print("[2/5] Creating stratified train/test split (80/20)...")
    train_pairs, test_pairs = create_fixed_split(
        data,
        test_size=0.2,
        random_state=42
    )
    print(f"  ✓ Train: {len(train_pairs):,} samples")
    print(f"  ✓ Test:  {len(test_pairs):,} samples")
    print(f"  ✓ Random seed: 42 (reproducible)")
    print()
    
    # ── STEP 3: Train Model & Generate Predictions ──────────────────────────
    print(f"[3/5] Training {model_type.upper()} model...")
    
    if model_type == "tfidf":
        # Phase-1: TF-IDF + Logistic Regression
        pipeline = build_pipeline()
        train_texts = [text for text, _ in train_pairs]
        train_labels = [label for _, label in train_pairs]
        pipeline.fit(train_texts, train_labels)
        
        test_texts = [text for text, _ in test_pairs]
        y_pred_raw = pipeline.predict(test_texts)
        
        # Get confidence scores
        try:
            proba = pipeline.predict_proba(test_texts)
            confidence_scores = proba.max(axis=1).tolist()
        except:
            confidence_scores = [0.5] * len(y_pred_raw)
        
        print(f"  ✓ TF-IDF + LogisticRegression trained")
    
    elif model_type == "bert":
        # Phase-2: DistilBERT (if available)
        try:
            from models.bert_triage import BERTTriageModel
            model = BERTTriageModel()
            test_texts = [text for text, _ in test_pairs]
            predictions_data = [model.predict(text) for text in test_texts]
            y_pred_raw = [p["category"] for p in predictions_data]
            confidence_scores = [p["confidence"] for p in predictions_data]
            print(f"  ✓ DistilBERT model loaded")
        except ImportError:
            print("  ⚠ DistilBERT not available, falling back to TF-IDF")
            return load_and_evaluate(model_type="tfidf", output_dir=output_dir)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print()
    
    # ── STEP 4: Force to 10 Categories ──────────────────────────────────────
    print("[4/5] Forcing predictions to 10 fixed categories (no catch-all)...")
    y_true = [label for _, label in test_pairs]
    y_pred = force_to_10_categories(y_pred_raw)
    
    # Verify all categories are valid
    invalid = set(y_pred) - set(FIXED_CATEGORIES)
    if invalid:
        raise ValueError(f"Invalid categories found: {invalid}")
    print(f"  ✓ All {len(y_pred):,} predictions are in 10-class set")
    print(f"  ✓ Categories: {', '.join(FIXED_CATEGORIES)}")
    print()
    
    # ── STEP 5: Comprehensive Evaluation ────────────────────────────────────
    print("[5/5] Running comprehensive evaluation...")
    
    test_queries = [text for text, _ in test_pairs]
    evaluator = ComprehensiveEvaluator(
        y_true=y_true,
        y_pred=y_pred,
        confidence_scores=confidence_scores,
        test_queries=test_queries,
        complex_query_indices=None  # No complex queries marked in this run
    )
    
    # Generate all outputs
    eval_results = evaluator.evaluate()
    report = evaluator.generate_report(os.path.join(output_dir, "evaluation_report.txt"))
    evaluator.export_to_json(os.path.join(output_dir, "comprehensive_evaluation.json"))
    evaluator.plot_confusion_matrix(os.path.join(output_dir, "confusion_matrix.png"))
    
    print(f"  ✓ Evaluation complete")
    print()
    
    # ── Business Evaluation Report ──────────────────────────────────────────
    print("Generating business evaluation report...")
    dataset_info = {
        "name": "Bitext Insurance LLM Dataset",
        "total_size": len(data),
        "num_classes": 10,
        "train_test_split": "80/20",
        "train_size": len(train_pairs),
        "test_size": len(test_pairs),
        "random_seed": 42,
        "model": model_type.upper(),
        "source_url": "https://huggingface.co/datasets/bitext/Bitext-insurance-llm-chatbot-training-dataset"
    }
    
    claimsense_baseline = {
        "accuracy": 0.93,
        "source": "HuggingFace model card (self-reported)",
        "dataset": "Unknown (not independently verified)",
        "model": "ClaimSense-AI v1",
    }
    
    bus_report = BusinessEvaluationReport(eval_results, dataset_info, claimsense_baseline)
    bus_report_text = bus_report.generate_full_report(
        os.path.join(output_dir, "Business_Evaluation_Report.md")
    )
    
    print()
    
    # ── Summary ─────────────────────────────────────────────────────────────
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print()
    print("KEY RESULTS:")
    print(f"  Accuracy:                    {eval_results['accuracy']:.4f} ({eval_results['accuracy']*100:.2f}%)")
    print(f"  Macro F1:                    {eval_results['macro_f1']:.4f}")
    print(f"  Weighted F1:                 {eval_results['weighted_f1']:.4f}")
    print(f"  Business Score (normalized): {eval_results['normalized_business_score']:.6f}")
    print()
    print(f"AUTO-ROUTING ANALYSIS:")
    ar = eval_results['auto_routing_analysis']
    print(f"  Auto-routed:                 {ar['auto_routed_tickets']:,} / {ar['auto_routed_tickets'] + ar['human_handled_tickets']:,} ({ar['auto_routed_percentage']:.1f}%)")
    print(f"  Human-handled:               {ar['human_handled_tickets']:,} / {ar['auto_routed_tickets'] + ar['human_handled_tickets']:,} ({ar['human_handled_percentage']:.1f}%)")
    print()
    print(f"OUTPUT FILES:")
    print(f"  📊 {output_dir}/evaluation_report.txt")
    print(f"  📈 {output_dir}/Business_Evaluation_Report.md")
    print(f"  📉 {output_dir}/confusion_matrix.png")
    print(f"  📋 {output_dir}/comprehensive_evaluation.json")
    print()
    
    return eval_results


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive evaluation for dissertation"
    )
    parser.add_argument(
        "--model",
        default="tfidf",
        choices=["tfidf", "bert"],
        help="Model to evaluate (tfidf=Phase-1, bert=Phase-2)"
    )
    parser.add_argument(
        "--output-dir",
        default="eval_output",
        help="Directory to save evaluation results"
    )
    
    args = parser.parse_args()
    
    results = load_and_evaluate(model_type=args.model, output_dir=args.output_dir)
    
    acc_pct = results["accuracy"] * 100
    ar = results["auto_routing_analysis"]
    auto_pct = ar["auto_routed_percentage"]
    low_conf_pct = ar["human_handled_percentage"]



if __name__ == "__main__":
    main()
