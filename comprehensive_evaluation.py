"""
comprehensive_evaluation.py
=============================
REVISED EVALUATION METHODOLOGY FOR DISSERTATION

This module implements a strict 10-class classification evaluation with business-oriented
metrics. Key differences from previous evaluation:

1. **NO catch-all category** - every ticket is forced to one of 10 predefined departments
2. **Confusion matrix focus** - 10×10 matrix showing all classification outcomes
3. **Business reward/penalty model** - reflects operational impact, not just accuracy
4. **Complex queries tracked separately** - escalation is a separate experiment
5. **Fair comparison** - explicitly documents evaluation conditions vs ClaimSense baseline

EVALUATION FRAMEWORK
────────────────────────────────────────────────────────────────────────────────

┌─ PRIMARY EXPERIMENT: 10-CLASS CLASSIFICATION ──────────────────────────────┐
│                                                                              │
│ Every ticket → Assigned to ONE of 10 categories                             │
│ No exclusions, no catch-all, no escalation in primary eval                  │
│                                                                              │
│ Output:                                                                      │
│  • Accuracy (correct / total)                                               │
│  • Macro F1, Weighted F1                                                    │
│  • 10×10 confusion matrix                                                   │
│  • Per-category precision, recall, F1                                       │
│  • Business reward/penalty matrix                                           │
│  • Business value score (normalized)                                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ SECONDARY EXPERIMENT: COMPLEX QUERIES ───────────────────────────────────┐
│                                                                              │
│ Separate analysis of difficult/ambiguous queries                            │
│ When forced into 10 categories, how well do they perform?                   │
│                                                                              │
│ Output:                                                                      │
│  • Count and percentage of complex queries                                  │
│  • Their distribution across categories                                     │
│  • Forced 10-class performance (accuracy, F1)                               │
│  • LLM escalation rate (if applicable)                                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ OPERATIONAL METRICS ─────────────────────────────────────────────────────┐
│                                                                              │
│ • Auto-routing percentage (system confidence → auto-routed / total)         │
│ • Human-intervention percentage (system low confidence → manual)            │
│ • Breakdown by category (which ones auto-route well, which don't)           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ BUSINESS REWARD/PENALTY MODEL ───────────────────────────────────────────┐
│                                                                              │
│ Principle: Correct routing saves effort; incorrect routing adds cost        │
│                                                                              │
│ BASE PENALTY (all misclassifications):                                      │
│  • Wrong category = +1 unit of rework (manual triage + rerouting)           │
│  • Customer delay (SLA clock reset)                                         │
│  • Support agent inspection time                                            │
│                                                                              │
│ MULTIPLIER: SLA RISK FACTOR BY CATEGORY                                     │
│  • Emergency Services:        3.0× (high urgency, SLA violation risk)        │
│  • Claims:                    2.5× (financial impact, settlement delays)    │
│  • Complaints & Feedback:     2.0× (retention risk, escalation potential)   │
│  • Policy Changes:            1.8× (contract modifications, legal risk)     │
│  • Technical Support:         1.5× (service availability impact)            │
│  • Billing & Payments:        1.3× (revenue/payment flow)                   │
│  • Policy & Coverage:         1.2× (informational, moderate impact)         │
│  • Refund & Returns:          1.2× (financial impact, lower urgency)        │
│  • Account & Password:        1.1× (account access, recovery time needed)   │
│  • General Inquiry:           1.0× (baseline, informational)                │
│                                                                              │
│ CALCULATION:                                                                 │
│  Penalty[i→j] = SLA_FACTOR[j] * (1 if i≠j else 0)                           │
│  Reward[i→j] = ROUTING_EFFICIENCY * (1 if i==j else 0)                     │
│  Where ROUTING_EFFICIENCY = 0.1 (represents cost savings per correct route) │
│                                                                              │
│ Business Score = (Correct predictions × 0.1) - (Sum of penalties)           │
│ Normalized Business Score = Business Score / Total tickets                  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

COMPARISON WITH CLAIMSENSE
─────────────────────────────
The evaluation document will clearly state:

  Dataset:           Bitext Insurance (public, 7000+ samples)
  Classes:           10 fixed insurance categories
  Train/Test Split:  80/20 stratified (seed=42)
  Evaluation Metric: Accuracy on held-out test set
  Model:             Phase-1 TF-IDF (lightweight), Phase-2 DistilBERT (optional)
  ClaimSense Ref:    ~93% accuracy (self-reported from HuggingFace)
  Our Result:        [independently measured accuracy]

Claim: "Our system achieves [X]% on a strict 10-class insurance triage task,
       comparable to ClaimSense's reported [~93%] on their benchmark."

We do NOT claim higher SOTA performance unless independently verified.

"""

import os
import sys
import json
import csv
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict

# Sklearn imports for metrics
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Try to import matplotlib for visualizations
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION: 10 FIXED INSURANCE CATEGORIES
# ────────────────────────────────────────────────────────────────────────────

FIXED_CATEGORIES = [
    "Claims",
    "Policy & Coverage",
    "Billing & Payments",
    "Complaints & Feedback",
    "General Inquiry",
    "Account & Password",
    "Technical Support",
    "Policy Changes",
    "Emergency Services",
    "Refund & Returns",
]

# ────────────────────────────────────────────────────────────────────────────
# BUSINESS IMPACT MODEL: SLA RISK FACTORS
# ────────────────────────────────────────────────────────────────────────────

SLA_RISK_FACTORS = {
    "Emergency Services": 3.0,      # Highest urgency, immediate SLA impact
    "Claims": 2.5,                  # Financial & settlement delays
    "Complaints & Feedback": 2.0,   # Retention risk, escalation potential
    "Policy Changes": 1.8,          # Contract modifications, legal review
    "Technical Support": 1.5,       # Service availability & recovery
    "Billing & Payments": 1.3,      # Revenue & payment flow
    "Policy & Coverage": 1.2,       # Informational, moderate impact
    "Refund & Returns": 1.2,        # Financial impact, lower urgency
    "Account & Password": 1.1,      # Account recovery needed
    "General Inquiry": 1.0,         # Baseline, purely informational
}

ROUTING_EFFICIENCY = 0.1  # Cost savings per correct route


# ────────────────────────────────────────────────────────────────────────────
# CORE EVALUATION CLASS
# ────────────────────────────────────────────────────────────────────────────

class ComprehensiveEvaluator:
    """
    Strict 10-class evaluation with business metrics.
    
    Usage:
        evaluator = ComprehensiveEvaluator(y_true, y_pred, confidence_scores=None)
        results = evaluator.evaluate()
        report = evaluator.generate_report()
    """
    
    def __init__(
        self,
        y_true: List[str],
        y_pred: List[str],
        confidence_scores: Optional[List[float]] = None,
        test_queries: Optional[List[str]] = None,
        complex_query_indices: Optional[List[int]] = None
    ):
        """
        Initialize evaluator with predictions.
        
        Args:
            y_true: Ground truth labels (must be from FIXED_CATEGORIES)
            y_pred: Predicted labels (forced to FIXED_CATEGORIES)
            confidence_scores: Optional confidence per prediction (0-1)
            test_queries: Optional query texts for complex query analysis
            complex_query_indices: Optional indices of queries marked as complex
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.confidence_scores = confidence_scores or [0.5] * len(y_pred)
        self.test_queries = test_queries or [""] * len(y_pred)
        self.complex_query_indices = set(complex_query_indices or [])
        
        # Validate categories
        unique_true = set(y_true)
        unique_pred = set(y_pred)
        invalid_true = unique_true - set(FIXED_CATEGORIES)
        invalid_pred = unique_pred - set(FIXED_CATEGORIES)
        
        if invalid_true:
            raise ValueError(f"y_true contains invalid categories: {invalid_true}")
        if invalid_pred:
            raise ValueError(f"y_pred contains invalid categories: {invalid_pred}")
        
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: {len(y_true)} true vs {len(y_pred)} pred")
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Run complete evaluation: accuracy, F1, confusion matrix, business metrics.
        
        Returns:
            Dictionary containing all evaluation results
        """
        results = {}
        
        # ── Mathematical Metrics ────────────────────────────────────────────
        results["total_tickets"] = len(self.y_true)
        results["accuracy"] = accuracy_score(self.y_true, self.y_pred)
        results["macro_f1"] = f1_score(self.y_true, self.y_pred, average="macro", zero_division=0)
        results["weighted_f1"] = f1_score(self.y_true, self.y_pred, average="weighted", zero_division=0)
        
        # ── Confusion Matrix ────────────────────────────────────────────────
        cm = confusion_matrix(self.y_true, self.y_pred, labels=FIXED_CATEGORIES)
        results["confusion_matrix"] = cm.tolist()
        results["confusion_matrix_labels"] = FIXED_CATEGORIES
        
        # ── Per-Category Metrics ────────────────────────────────────────────
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true, self.y_pred,
            labels=FIXED_CATEGORIES,
            zero_division=0
        )
        
        per_category = {}
        for cat, p, r, f, s in zip(FIXED_CATEGORIES, precision, recall, f1, support):
            per_category[cat] = {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
                "correct": int(cm[FIXED_CATEGORIES.index(cat), FIXED_CATEGORIES.index(cat)]),
            }
        results["per_category_metrics"] = per_category
        
        # ── Business Reward/Penalty Matrix ──────────────────────────────────
        reward_penalty_matrix = self._compute_reward_penalty_matrix(cm)
        results["reward_penalty_matrix"] = reward_penalty_matrix
        
        # ── Business Score Calculation ──────────────────────────────────────
        correct_count = sum(1 for t, p in zip(self.y_true, self.y_pred) if t == p)
        total_reward = correct_count * ROUTING_EFFICIENCY
        total_penalty = self._compute_total_penalty()
        
        results["total_correct"] = correct_count
        results["total_reward"] = float(total_reward)
        results["total_penalty"] = float(total_penalty)
        results["business_score"] = float(total_reward - total_penalty)
        results["normalized_business_score"] = float(
            (total_reward - total_penalty) / len(self.y_true) if len(self.y_true) > 0 else 0
        )
        
        # ── Complex Query Analysis (separate) ───────────────────────────────
        complex_analysis = self._analyze_complex_queries()
        results["complex_query_analysis"] = complex_analysis
        
        # ── Auto-Routing Percentage ────────────────────────────────────────
        auto_routing = self._compute_auto_routing_percentage()
        results["auto_routing_analysis"] = auto_routing
        
        # ── Confidence Distribution ────────────────────────────────────────
        results["confidence_stats"] = {
            "mean": float(np.mean(self.confidence_scores)),
            "median": float(np.median(self.confidence_scores)),
            "std": float(np.std(self.confidence_scores)),
            "min": float(np.min(self.confidence_scores)),
            "max": float(np.max(self.confidence_scores)),
        }
        
        return results
    
    def _compute_reward_penalty_matrix(self, cm: np.ndarray) -> Dict[str, List[List[float]]]:
        """
        Create 10×10 business impact matrix.
        Diagonal (correct): +ROUTING_EFFICIENCY
        Off-diagonal (wrong): -SLA_FACTOR of actual category
        """
        matrix = []
        for i, actual_cat in enumerate(FIXED_CATEGORIES):
            row = []
            for j, pred_cat in enumerate(FIXED_CATEGORIES):
                if i == j:
                    # Correct routing: reward
                    value = ROUTING_EFFICIENCY
                else:
                    # Incorrect routing: penalty based on actual category SLA risk
                    sla_factor = SLA_RISK_FACTORS.get(actual_cat, 1.0)
                    value = -sla_factor
                row.append(float(value))
            matrix.append(row)
        
        return {
            "matrix": matrix,
            "labels": FIXED_CATEGORIES,
            "sla_factors": SLA_RISK_FACTORS,
            "routing_efficiency": ROUTING_EFFICIENCY,
        }
    
    def _compute_total_penalty(self) -> float:
        """Calculate total penalty across all misclassifications."""
        total_penalty = 0.0
        for actual, predicted in zip(self.y_true, self.y_pred):
            if actual != predicted:
                sla_factor = SLA_RISK_FACTORS.get(actual, 1.0)
                total_penalty += sla_factor
        return total_penalty
    
    def _analyze_complex_queries(self) -> Dict[str, Any]:
        """Analyze performance on complex/ambiguous queries (if flagged)."""
        if not self.complex_query_indices:
            return {
                "total_complex": 0,
                "percentage_of_total": 0.0,
                "analysis": "No complex queries flagged in this evaluation"
            }
        
        complex_indices = list(self.complex_query_indices)
        total_complex = len(complex_indices)
        percentage = (total_complex / len(self.y_true) * 100) if len(self.y_true) > 0 else 0
        
        # Complex subset metrics
        complex_true = [self.y_true[i] for i in complex_indices]
        complex_pred = [self.y_pred[i] for i in complex_indices]
        complex_correct = sum(1 for t, p in zip(complex_true, complex_pred) if t == p)
        complex_accuracy = complex_correct / total_complex if total_complex > 0 else 0
        
        # Distribution by category
        category_counts = defaultdict(int)
        for idx in complex_indices:
            category_counts[self.y_true[idx]] += 1
        
        return {
            "total_complex": total_complex,
            "percentage_of_total": float(percentage),
            "complex_accuracy": float(complex_accuracy),
            "complex_macro_f1": float(
                f1_score(complex_true, complex_pred, average="macro", zero_division=0)
            ),
            "complex_weighted_f1": float(
                f1_score(complex_true, complex_pred, average="weighted", zero_division=0)
            ),
            "distribution_by_category": dict(category_counts),
            "note": "These queries are evaluated in the primary 10-class benchmark, not excluded"
        }
    
    def _compute_auto_routing_percentage(self) -> Dict[str, Any]:
        """
        Compute automation percentage based on confidence threshold.
        Tickets above threshold are "auto-routed", below require human handling.
        """
        auto_threshold = 0.7  # Configurable: tickets with conf > 70% are auto-routed
        
        auto_routed = sum(1 for conf in self.confidence_scores if conf >= auto_threshold)
        human_handled = len(self.confidence_scores) - auto_routed
        
        auto_routed_correct = 0
        human_handled_correct = 0
        
        for i, (true, pred, conf) in enumerate(zip(self.y_true, self.y_pred, self.confidence_scores)):
            if true == pred:
                if conf >= auto_threshold:
                    auto_routed_correct += 1
                else:
                    human_handled_correct += 1
        
        return {
            "auto_threshold": auto_threshold,
            "auto_routed_tickets": auto_routed,
            "human_handled_tickets": human_handled,
            "auto_routed_percentage": float(auto_routed / len(self.confidence_scores) * 100),
            "human_handled_percentage": float(human_handled / len(self.confidence_scores) * 100),
            "auto_routed_correct": auto_routed_correct,
            "auto_routed_accuracy": float(
                auto_routed_correct / auto_routed if auto_routed > 0 else 0
            ),
            "human_handled_correct": human_handled_correct,
            "human_handled_accuracy": float(
                human_handled_correct / human_handled if human_handled > 0 else 0
            ),
        }
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive human-readable evaluation report.
        
        Args:
            output_file: Optional file path to save report
            
        Returns:
            Report as formatted string
        """
        results = self.evaluate()
        
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE EVALUATION REPORT: 10-CLASS INSURANCE TRIAGE")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        report.append("SUMMARY METRICS")
        report.append("─" * 80)
        report.append(f"Total Tickets Evaluated:        {results['total_tickets']:,}")
        report.append(f"Overall Accuracy:               {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        report.append(f"Macro F1 Score:                 {results['macro_f1']:.4f}")
        report.append(f"Weighted F1 Score:              {results['weighted_f1']:.4f}")
        report.append("")
        
        # Confusion matrix summary
        report.append("CONFUSION MATRIX (Predicted vs Actual)")
        report.append("─" * 80)
        report.append(self._format_confusion_matrix(results["confusion_matrix"], results["confusion_matrix_labels"]))
        report.append("")
        
        # Per-category breakdown
        report.append("PER-CATEGORY PERFORMANCE")
        report.append("─" * 80)
        report.append(f"{'Category':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        report.append("─" * 80)
        for cat, metrics in results["per_category_metrics"].items():
            report.append(
                f"{cat:<25} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                f"{metrics['f1']:>10.4f} {metrics['support']:>10}"
            )
        report.append("")
        
        # Business metrics
        report.append("BUSINESS-ORIENTED METRICS")
        report.append("─" * 80)
        report.append(f"Correct Routing (Tickets):      {results['total_correct']:,}")
        report.append(f"Total Routing Rewards:          {results['total_reward']:.4f}")
        report.append(f"Total Routing Penalties:        {results['total_penalty']:.4f}")
        report.append(f"Business Score:                 {results['business_score']:.4f}")
        report.append(f"Normalized Business Score:      {results['normalized_business_score']:.6f}")
        report.append("  (per ticket, accounting for operational cost)")
        report.append("")
        
        # Complex queries
        report.append("COMPLEX QUERY ANALYSIS")
        report.append("─" * 80)
        cx = results["complex_query_analysis"]
        if cx["total_complex"] > 0:
            report.append(f"Total Complex Queries:          {cx['total_complex']}")
            report.append(f"Percentage of Total:            {cx['percentage_of_total']:.2f}%")
            report.append(f"Complex Query Accuracy:         {cx['complex_accuracy']:.4f}")
            report.append(f"Complex Macro F1:               {cx['complex_macro_f1']:.4f}")
            report.append(f"Note:                           These are INCLUDED in primary benchmark, not excluded")
        else:
            report.append("No complex queries tracked in this evaluation")
        report.append("")
        
        # Auto-routing analysis
        report.append("AUTOMATION & OPERATIONAL EFFICIENCY")
        report.append("─" * 80)
        ar = results["auto_routing_analysis"]
        report.append(f"Auto-Routing Threshold:         {ar['auto_threshold']:.2f} confidence")
        report.append(f"Auto-Routed Tickets:            {ar['auto_routed_tickets']:,} ({ar['auto_routed_percentage']:.2f}%)")
        report.append(f"  → Correct auto-routes:        {ar['auto_routed_correct']:,} ({ar['auto_routed_accuracy']:.2f}%)")
        report.append(f"Human-Handled Tickets:          {ar['human_handled_tickets']:,} ({ar['human_handled_percentage']:.2f}%)")
        report.append(f"  → Correct after human review: {ar['human_handled_correct']:,} ({ar['human_handled_accuracy']:.2f}%)")
        report.append("")
        
        # Confidence distribution
        report.append("CONFIDENCE DISTRIBUTION")
        report.append("─" * 80)
        cs = results["confidence_stats"]
        report.append(f"Mean Confidence:                {cs['mean']:.4f}")
        report.append(f"Median Confidence:              {cs['median']:.4f}")
        report.append(f"Std Dev:                        {cs['std']:.4f}")
        report.append(f"Range:                          {cs['min']:.4f} - {cs['max']:.4f}")
        report.append("")
        
        # SLA Risk Model Documentation
        report.append("BUSINESS IMPACT MODEL: SLA RISK FACTORS")
        report.append("─" * 80)
        report.append("Category                        Risk Factor")
        report.append("─" * 80)
        for cat in FIXED_CATEGORIES:
            factor = SLA_RISK_FACTORS.get(cat, 1.0)
            report.append(f"{cat:<30} {factor:.1f}×")
        report.append("")
        
        report.append("METHODOLOGY NOTES")
        report.append("─" * 80)
        report.append("• This is a STRICT 10-CLASS CLASSIFICATION evaluation")
        report.append("• NO catch-all or contextual category used")
        report.append("• Every ticket is assigned to one of 10 departments")
        report.append("• Complex/ambiguous tickets are INCLUDED in the primary benchmark")
        report.append("• LLM escalation (if applicable) is tracked separately")
        report.append("• Business metrics reflect operational impact of misrouting")
        report.append("• Accuracy formula: Correct predictions / Total tickets")
        report.append("")
        
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"✓ Report saved to {output_file}")
        
        return report_text
    
    def _format_confusion_matrix(self, cm: List[List[int]], labels: List[str]) -> str:
        """Format confusion matrix for display."""
        lines = []
        
        # Determine column widths
        max_label_width = max(len(label) for label in labels)
        col_width = max(4, max_label_width)
        
        # Header row
        header = " " * (max_label_width + 2)
        for label in labels:
            short = label[:col_width]
            header += f"{short:>{col_width}} "
        lines.append(header)
        lines.append("─" * len(header))
        
        # Data rows
        for i, actual_label in enumerate(labels):
            row = f"{actual_label[:max_label_width]:<{max_label_width}} "
            for j in range(len(labels)):
                value = cm[i][j]
                row += f"{value:>{col_width}} "
            lines.append(row)
        
        return "\n".join(lines)
    
    def export_to_json(self, output_file: str) -> None:
        """Export evaluation results to JSON."""
        results = self.evaluate()
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"✓ JSON results saved to {output_file}")
    
    def plot_confusion_matrix(self, output_file: Optional[str] = None) -> None:
        """Plot confusion matrix heatmap (requires matplotlib)."""
        if not HAS_MATPLOTLIB:
            print("⚠ matplotlib not installed, skipping plot")
            return
        
        results = self.evaluate()
        cm = np.array(results["confusion_matrix"])
        labels = results["confusion_matrix_labels"]
        
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        
        # Set ticks and labels
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        short_labels = [label[:14] for label in labels]
        ax.set_xticklabels(short_labels, rotation=45, ha="right")
        ax.set_yticklabels(short_labels)
        
        ax.set_xlabel("Predicted Category", fontsize=12, fontweight="bold")
        ax.set_ylabel("Actual Category", fontsize=12, fontweight="bold")
        ax.set_title("10×10 Confusion Matrix: Insurance Triage Classification", fontsize=14, fontweight="bold")
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Number of Tickets", fontsize=11)
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = cm[i, j]
                if value > 0:
                    color = "white" if value > thresh else "black"
                    ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=9)
        
        fig.tight_layout()
        
        if output_file:
            fig.savefig(output_file, dpi=150, bbox_inches="tight")
            print(f"✓ Confusion matrix plot saved to {output_file}")
        else:
            plt.show()
        
        plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# HELPER: FORCE PREDICTIONS TO 10 CATEGORIES
# ────────────────────────────────────────────────────────────────────────────

def force_to_10_categories(
    predictions: List[str],
    fallback_category: str = "General Inquiry"
) -> List[str]:
    """
    Ensure all predictions are from FIXED_CATEGORIES.
    Any prediction not in the list is mapped to fallback_category.
    
    This is crucial for the primary benchmark: no catches-all or unknowns.
    
    Args:
        predictions: Raw predictions from the model
        fallback_category: Default category for invalid predictions
        
    Returns:
        List of valid predictions
    """
    if fallback_category not in FIXED_CATEGORIES:
        raise ValueError(f"Fallback category '{fallback_category}' not in FIXED_CATEGORIES")
    
    fixed_predictions = []
    for pred in predictions:
        if str(pred).strip() in FIXED_CATEGORIES:
            fixed_predictions.append(str(pred).strip())
        else:
            fixed_predictions.append(fallback_category)
    
    return fixed_predictions


# ────────────────────────────────────────────────────────────────────────────
# DEMO / USAGE
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simple demo
    print("Comprehensive Insurance Triage Evaluator")
    print("=" * 60)
    print()
    
    # Generate synthetic test data
    np.random.seed(42)
    n_samples = 100
    
    # Synthetic ground truth
    y_true = np.random.choice(FIXED_CATEGORIES, size=n_samples).tolist()
    
    # Synthetic predictions (80% match + 20% random errors)
    y_pred = []
    confidence = []
    for true_label in y_true:
        if np.random.random() < 0.8:
            # 80% match
            y_pred.append(true_label)
            confidence.append(np.random.uniform(0.7, 0.99))
        else:
            # 20% mismatch
            wrong_label = np.random.choice([c for c in FIXED_CATEGORIES if c != true_label])
            y_pred.append(wrong_label)
            confidence.append(np.random.uniform(0.4, 0.7))
    
    # Run evaluation
    evaluator = ComprehensiveEvaluator(y_true, y_pred, confidence_scores=confidence)
    
    # Print report
    report = evaluator.generate_report()
    print(report)
    
    # Save outputs
    output_dir = "eval_output"
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator.export_to_json(os.path.join(output_dir, "comprehensive_evaluation.json"))
    evaluator.plot_confusion_matrix(os.path.join(output_dir, "confusion_matrix.png"))
    
    with open(os.path.join(output_dir, "evaluation_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n✓ All outputs saved to {output_dir}/")
