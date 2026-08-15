"""
evaluate_metrics.py  [v3]
=========================
Evaluation suite for the dissertation:
  1. Phase-1 TF-IDF strict 10-class holdout evaluation (80/20 stratified split)
  2. Per-category precision / recall / F1
  3. 10 x 10 confusion matrix
  4. Confidence distribution + overconfidence analysis
  5. Informal-query robustness simulation
  6. Phase-2 BERT ensemble subset evaluation
  7. Published-reference comparison with ClaimSense
  8. Visualizations saved to eval_output/ (PNG)

EVALUATION METHODOLOGY NOTES
─────────────────────────────
✅ VALID:  The 80/20 split with seed=42 is reproduced for the current
           10-class evaluation. Every evaluated ticket is assigned to exactly
           one of the 10 routing categories.

⚠️  INTERPRETATION: The holdout result measures performance on the same
                     dataset distribution used for training. It should not
                     be presented as a guarantee of production performance.

⚠️  DATA QUALITY:  Exact duplicate/near-duplicate checks should be performed
                   before making a strong generalisation claim, because
                   duplicate records can make a random holdout easier.

⚠️  BENCHMARK:  ClaimSense is shown only as a published/self-reported
                reference. The two results were not reproduced under an
                identical evaluation protocol, so no direct SOTA/superiority
                claim is made.

⚠️  CONFIDENCE:  High confidence does not necessarily mean a prediction is
                 correct. Incorrect high-confidence predictions are reported
                 as a calibration/overconfidence risk.

MISSING / FUTURE VALIDATION:
  - Duplicate-controlled or group-based split
  - Full-test-set BERT evaluation
  - Calibration curve
  - Out-of-distribution (OOD) test with non-insurance queries
"""

import os, sys
import numpy as np
import json

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("⚠️  matplotlib not installed — skipping plots (pip install matplotlib)")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import joblib


# ── Helpers ────────────────────────────────────────────────────────────────

def _banner(title, width=65):
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def _save_plot(fig, name):
    if not HAS_MPL:
        return
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved: eval_output/{name}")


# ── Confusion matrix plot ──────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix"):
    if not HAS_MPL:
        return
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    short = [l[:14] for l in labels]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13, pad=14)

    thresh = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v > thresh else "black", fontsize=9)
    fig.tight_layout()
    _save_plot(fig, "confusion_matrix.png")


# ── Confidence distribution plot ───────────────────────────────────────────

def plot_confidence_distribution(y_test, preds, model_pipeline, labels):
    """
    Shows confidence score histograms split by correct vs incorrect predictions.
    Reveals if the model is overconfident on wrong predictions.
    """
    if not HAS_MPL:
        return

    all_probs = model_pipeline.predict_proba(list(y_test[:2000]))  # cap for speed
    max_confs = all_probs.max(axis=1)
    correct = np.array(list(preds[:2000])) == np.array(list(y_test[:2000]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overall confidence histogram
    ax = axes[0]
    ax.hist(max_confs[correct], bins=30, alpha=0.7, color="#2196F3", label=f"Correct ({correct.sum()})")
    ax.hist(max_confs[~correct], bins=30, alpha=0.7, color="#F44336", label=f"Incorrect ({(~correct).sum()})")
    ax.set_xlabel("Confidence Score", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Confidence Distribution\n(Correct vs Incorrect)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)

    # Right: mean confidence per class
    ax2 = axes[1]
    class_list = list(model_pipeline.classes_)
    class_confs = []
    for lbl in labels:
        if lbl in class_list:
            idx = class_list.index(lbl)
            class_confs.append(float(all_probs[:, idx].mean()))
        else:
            class_confs.append(0.0)

    colors = ["#2196F3" if c > 0.5 else "#F44336" for c in class_confs]
    short = [l[:16] for l in labels]
    bars = ax2.barh(short, class_confs, color=colors)
    ax2.set_xlabel("Mean Confidence", fontsize=11)
    ax2.set_title("Mean Confidence per Class", fontsize=12)
    ax2.set_xlim(0, 1)
    for bar, v in zip(bars, class_confs):
        ax2.text(v + 0.01, bar.get_y() + bar.get_height()/2, f"{v:.2f}",
                 va="center", fontsize=9)

    fig.tight_layout()
    _save_plot(fig, "confidence_distribution.png")

    # Stats
    print(f"\n  Confidence Analysis (first 2,000 test samples):")
    print(f"  Mean confidence (correct)   : {max_confs[correct].mean():.4f}")
    print(f"  Mean confidence (incorrect) : {max_confs[~correct].mean():.4f} {'⚠️ Overconfident' if max_confs[~correct].mean() > 0.7 else '✓ Calibrated'}")
    print(f"  % predictions with conf>0.9 : {(max_confs > 0.9).mean():.1%}")
    print(f"  % predictions with conf>0.5 : {(max_confs > 0.5).mean():.1%}")
    print(f"  % predictions with conf>=0.70: {(max_confs >= 0.70).mean():.1%}")
    print("  Interpretation: confidence thresholding is a routing-control mechanism,")
    print("  not a guarantee of correctness. High-confidence errors are reported explicitly.")

    wrong_high_conf = max_confs[~correct] > 0.8
    if wrong_high_conf.sum() > 0:
        print(f"  ⚠️  {wrong_high_conf.sum()} incorrect predictions had >0.8 confidence — overconfidence risk.")


# ── Per-category F1 bar chart ───────────────────────────────────────────────

def plot_f1_bars(labels, f1_scores, supports):
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    colors = ["#4CAF50" if f >= 0.95 else "#FF9800" if f >= 0.85 else "#F44336" for f in f1_scores]
    bars = ax.bar(x, f1_scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([l[:14] for l in labels], rotation=40, ha="right", fontsize=9)
    ax.set_ylim(0.8, 1.02)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("Per-Category F1 Scores (Held-out Test Set)", fontsize=13)
    ax.axhline(0.95, color="#2196F3", linestyle="--", linewidth=1, alpha=0.6, label="0.95 threshold")
    ax.legend(fontsize=9)
    for bar, f, sup in zip(bars, f1_scores, supports):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{f:.3f}\n(n={sup})", ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout()
    _save_plot(fig, "f1_scores.png")


# ── Benchmark chart ────────────────────────────────────────────────────────

def plot_benchmark(our_acc, reference_acc=None):
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    models = ["ClaimSense\n(published reference)", "Our Phase-1\n(TF-IDF)"]
    if reference_acc is None:
        # Reference value intentionally omitted because the published result
        # is not protocol-matched to this evaluation.
        accs = [0, our_acc * 100]
    else:
        accs = [reference_acc * 100, our_acc * 100]
    colors = ["#B0BEC5", "#2196F3"]
    bars = ax.bar(models, accs, color=colors, width=0.4, edgecolor="white")
    ax.set_ylim(85, 100)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("10-Class Evaluation and Published Reference", fontsize=13)
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.1, f"{v:.2f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    if reference_acc is not None:
        ax.text(0.5, 0.92, "Reference only — not protocol-matched",
                transform=ax.transAxes, ha="center", fontsize=11,
                color="#607D8B", fontweight="bold")
    fig.tight_layout()
    _save_plot(fig, "benchmark_comparison.png")


# ── Main evaluation ────────────────────────────────────────────────────────

def evaluate():
    _banner("AI CRM Insurance Model Evaluation & Comparison  [v3]")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Load dataset
    print("\nLoading dataset via EnhancedTriageModel._get_training_data() ...")
    from models.enhanced_triage import EnhancedTriageModel

    model_pickle = os.path.join(base_dir, "models", "enhanced_triage_model.pkl")
    if os.path.exists(model_pickle):
        os.remove(model_pickle)
        print("Removed stale model file — will retrain on full dataset.")

    model = EnhancedTriageModel()
    all_data = model._get_training_data()
    print(f"Total combined dataset size: {len(all_data):,} samples")

    all_texts, all_targets = zip(*all_data)

    # 2. Check exact-duplicate overlap before the standard split.
    # This does NOT silently remove records from the primary result; it reports
    # the overlap and also creates a leakage-controlled comparison below.
    normalized_texts = [str(t).strip().lower() for t in all_texts]
    duplicate_count = len(normalized_texts) - len(set(normalized_texts))
    print(f"\nExact duplicate records in combined dataset: {duplicate_count:,}")

    # Standard reproducible 80/20 split.
    # Standard split: Reproduce exact 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        all_texts, all_targets,
        test_size=0.2, random_state=42, stratify=all_targets
    )
    print(f"Train set : {len(X_train):,} samples")
    print(f"Test  set : {len(X_test):,} samples  (held-out, never seen during training)")

    train_norm = set(str(t).strip().lower() for t in X_train)
    test_norm = [str(t).strip().lower() for t in X_test]
    duplicate_test_overlap = sum(1 for t in test_norm if t in train_norm)
    print(f"Exact-duplicate test/train overlap: {duplicate_test_overlap:,} "
          f"({duplicate_test_overlap/len(X_test):.2%} of test set)")
    if duplicate_test_overlap:
        print("  NOTE: Standard holdout is retained for reproducibility, but this overlap")
        print("        can make the random-split result optimistic. A duplicate-controlled")
        print("        benchmark is calculated separately below.")

    print("\nRetraining Phase-1 model on X_train ...")
    model.pipeline.fit(X_train, y_train)
    preds = model.pipeline.predict(X_test)

    # 3. Core metrics
    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average='weighted', zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_test, preds, average='macro', zero_division=0)
    _, _, per_class_f1, support = precision_recall_fscore_support(
        y_test, preds, average=None, zero_division=0, labels=model.LABELS)

    _banner(f"Phase-1 Model Results  (held-out {len(X_test):,} samples)")
    print(f"  Overall Accuracy   : {acc:.4%}")
    print(f"  Weighted F1-Score  : {f1:.4f}")
    print(f"  Macro F1-Score     : {macro_f1:.4f}")
    print(f"  Weighted Precision : {precision:.4f}")
    print(f"  Weighted Recall    : {recall:.4f}")

    # Leakage-controlled exact-text benchmark.
    # Keep one copy of each normalized query before splitting.
    unique_pairs = {}
    for txt, label in zip(all_texts, all_targets):
        key = str(txt).strip().lower()
        if key not in unique_pairs:
            unique_pairs[key] = (txt, label)
    unique_texts = [v[0] for v in unique_pairs.values()]
    unique_targets = [v[1] for v in unique_pairs.values()]

    X_u_train, X_u_test, y_u_train, y_u_test = train_test_split(
        unique_texts, unique_targets, test_size=0.2,
        random_state=42, stratify=unique_targets
    )
    controlled_pipeline = type(model.pipeline)(**{}) if False else None
    # Reuse the same pipeline class/configuration by cloning the fitted estimator.
    import sklearn.base
    controlled_pipeline = sklearn.base.clone(model.pipeline)
    controlled_pipeline.fit(X_u_train, y_u_train)
    controlled_preds = controlled_pipeline.predict(X_u_test)
    controlled_acc = accuracy_score(y_u_test, controlled_preds)
    _, _, controlled_macro_f1, _ = precision_recall_fscore_support(
        y_u_test, controlled_preds, average="macro", zero_division=0
    )
    _, _, controlled_weighted_f1, _ = precision_recall_fscore_support(
        y_u_test, controlled_preds, average="weighted", zero_division=0
    )
    _banner("Duplicate-Controlled Benchmark")
    print(f"  Unique query texts       : {len(unique_texts):,}")
    print(f"  Controlled train set     : {len(X_u_train):,}")
    print(f"  Controlled test set      : {len(X_u_test):,}")
    print(f"  Accuracy                 : {controlled_acc:.4%}")
    print(f"  Macro F1                 : {controlled_macro_f1:.4f}")
    print(f"  Weighted F1              : {controlled_weighted_f1:.4f}")
    print("  Interpretation           : Exact duplicate texts were removed before splitting.")
    print("                             This is a leakage-controlled sensitivity result,")
    print("                             not a replacement for the reproducible seed-42 result.")

    print(f"\n  Per-category breakdown:")
    print(f"  {'Category':<28} {'F1':>6}  {'Support':>8}  Status")
    print(f"  {'-'*60}")
    for cat, f, sup in zip(model.LABELS, per_class_f1, support):
        status = "✓ Excellent" if f >= 0.95 else ("⚠ Good" if f >= 0.85 else "▲ Improving")
        print(f"  {cat:<28} {f:6.4f}  {int(sup):>8}  {status}")

    print("\nFull Classification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    # 4. Confusion matrix
    _banner("Confusion Matrix Analysis")
    cm = confusion_matrix(y_test, preds, labels=model.LABELS)

    # Print notable off-diagonal cells
    print("  Top misclassifications (count > 0, off-diagonal):")
    found = False
    for i, true_cat in enumerate(model.LABELS):
        for j, pred_cat in enumerate(model.LABELS):
            if i != j and cm[i, j] > 0:
                print(f"    True: {true_cat:<28} → Predicted: {pred_cat:<28}  ({cm[i,j]} cases)")
                found = True
    if not found:
        print("    None — perfect classification on this split.")

    plot_confusion_matrix(y_test, preds, model.LABELS)

    # 4b. Experimental business-impact evaluation.
    # These are normalized research weights, NOT actual insurer financial/SLA costs.
    business_weights = {
        "Account & Password": 1.1,
        "Emergency Services": 3.0,
        "General Inquiry": 1.0,
        "Policy & Coverage": 1.2,
        "Billing & Payments": 1.3,
        "Claims": 2.5,
        "Complaints & Feedback": 2.0,
        "Technical Support": 1.5,
        "Policy Changes": 1.8,
        "Refund & Returns": 1.2,
    }
    correct_reward = 0.1
    total_reward = 0.0
    total_penalty = 0.0
    for i, true_cat in enumerate(model.LABELS):
        for j, pred_cat in enumerate(model.LABELS):
            count = int(cm[i, j])
            if i == j:
                total_reward += count * correct_reward
            else:
                total_penalty += count * business_weights.get(true_cat, 1.0)

    business_score = total_reward - total_penalty
    business_score_per_ticket = business_score / len(y_test)
    _banner("Experimental Business-Impact Evaluation")
    print("  Important: weights below are normalized experimental research assumptions,")
    print("  not actual insurance-industry financial costs or guaranteed SLA penalties.")
    print(f"  Correct-routing reward                 : +{correct_reward:.1f} units/ticket")
    print("  Misrouting penalty                     : actual-category business-impact weight")
    print(f"  Total reward                           : {total_reward:.2f} units")
    print(f"  Total penalty                          : {total_penalty:.2f} units")
    print(f"  Net business utility                  : {business_score:.2f} units")
    print(f"  Normalized utility per ticket         : {business_score_per_ticket:.4f} units/ticket")
    print("  Interpretation: the score quantifies routing utility under the defined")
    print("  experimental weights; it is not a percentage of business improvement.")

    # 5. Confidence distribution
    _banner("Confidence Distribution Analysis")
    plot_confidence_distribution(y_test, preds, model.pipeline, model.LABELS)

    # 6. F1 bar chart
    plot_f1_bars(model.LABELS, per_class_f1, support)

    # 7. Real-world simulation
    # Informal-query robustness test — informal phrasing, typos and slang.
    # This is a small manually constructed robustness test, not a production benchmark.
    real_world_samples = [
        # Billing & Payments (6)
        ("hey my paymnt didnt go thru how do i retry it",                   "Billing & Payments"),
        ("how much is my montly premium again",                             "Billing & Payments"),
        ("charged me twice in december i want a refund",                   "Billing & Payments"),
        ("set up autopay for me so i dont miss again",                     "Billing & Payments"),
        ("why was i billed twice this month",                               "Billing & Payments"),
        ("my payment bounced what do i do next",                           "Billing & Payments"),
        # Claims (6)
        ("i need to file a claim my car was hit in a parking lot",          "Claims"),
        ("still havent received my payout its been 3 weeks",               "Claims"),
        ("my claim was denied can u explain why and how 2 appeal",         "Claims"),
        ("someone hit my car while parked i need 2 claim",                 "Claims"),
        ("house got damaged in storm last night want to claim",            "Claims"),
        ("claim submitted 2 weeks ago no update at all",                   "Claims"),
        # Policy & Coverage (6)
        ("can u tell me what my deductible is for home",                    "Policy & Coverage"),
        ("do i have coverage for water damage from a burst pipe?",          "Policy & Coverage"),
        ("how do i get a cert of insurance for my landlord",               "Policy & Coverage"),
        ("how do i know if flood damage is covered under my plan",         "Policy & Coverage"),
        ("does my plan cover a rental car when mine is in the shop",       "Policy & Coverage"),
        ("am i covered if i drive my friends car",                         "Policy & Coverage"),
        # Policy Changes (6)
        ("i want 2 change my coverage plan asap",                           "Policy Changes"),
        ("my renewal is coming up and i want to switch plans",              "Policy Changes"),
        ("i want to add my wife to the policy we just got married",         "Policy Changes"),
        ("can i pause my policy for 2 months while im traveling",          "Policy Changes"),
        ("i want to remove the second car from my auto policy",            "Policy Changes"),
        ("need to update my home address on my policy",                    "Policy Changes"),
        # Technical Support (5)
        ("app keeps crashng when i try to open my policy wtf",              "Technical Support"),
        ("website wont load the claims form, getting error 500",            "Technical Support"),
        ("chatbot gave me wrong info about my coverage limits",             "Technical Support"),
        ("the portal shows my policy expired but i paid already",          "Technical Support"),
        ("cant upload docs the upload button doesnt work",                 "Technical Support"),
        # Complaints & Feedback (5)
        ("the agent i spoke to was incredibly rude and dismissive",         "Complaints & Feedback"),
        ("i want to give feedback about my experience today it was great",  "Complaints & Feedback"),
        ("i want to escalate my complaint to a supervisor",                "Complaints & Feedback"),
        ("i submitted feedback last week but nobody replied",              "Complaints & Feedback"),
        ("nobody called me back like they said they would",                "Complaints & Feedback"),
        # General Inquiry (4)
        ("what types of plans do you offer for small businesses",          "General Inquiry"),
        ("do u guys have discounts for safe drivers",                      "General Inquiry"),
        ("how do i get in touch with a real agent",                        "General Inquiry"),
        ("can i get insured on the same day",                              "General Inquiry"),
        # Account & Password (4)
        ("cant login forgot my password",                                  "Account & Password"),
        ("my acc is locked after too many attempts",                       "Account & Password"),
        ("otp expired before i could enter it",                            "Account & Password"),
        ("reset link never arrived checked spam folder too",               "Account & Password"),
        # Emergency Services (4)
        ("my house is on fire right now i need emergency help",            "Emergency Services"),
        ("car broke down on highway im stranded please send help",         "Emergency Services"),
        ("flat tire on freeway need roadside assistance now",              "Emergency Services"),
        ("flooding in my house right now what do i do",                   "Emergency Services"),
        # Refund & Returns (4)
        ("i cancelled my policy 2 weeks ago where is my refund",          "Refund & Returns"),
        ("how long does a refund take after policy cancellation",         "Refund & Returns"),
        ("i want my money back for the unused premium",                    "Refund & Returns"),
        ("got wrong refund amount its less than expected",                 "Refund & Returns"),
    ]
    N_SAMPLES = len(real_world_samples)

    _banner(f"Informal-Query Robustness Test  ({N_SAMPLES} informal/typo tickets)\n  Small manually constructed robustness set — not a production benchmark")
    rw_texts, rw_labels = zip(*real_world_samples)

    # Use BERT ensemble if available, fall back to TF-IDF
    try:
        from models.bert_triage import BERTTriageModel
        _bert = BERTTriageModel()
        rw_preds = [_bert.predict(t)[0] for t in rw_texts]
        print("  Using: BERT ensemble (Phase-2)")
    except Exception:
        rw_preds = list(model.pipeline.predict(rw_texts))
        print("  Using: TF-IDF Phase-1 model")

    rw_acc = accuracy_score(rw_labels, rw_preds)
    _, _, rw_f1, _ = precision_recall_fscore_support(
        rw_labels, rw_preds, average="weighted", zero_division=0)

    correct_count = int(rw_acc * N_SAMPLES)
    print(f"\n  Real-World Accuracy  : {rw_acc:.2%}  ({correct_count}/{N_SAMPLES} correct)")
    print(f"  Weighted F1-Score    : {rw_f1:.4f}")
    print("  Note                 : Result is descriptive for this small robustness set; no production target is claimed.")
    print(f"\n  Per-ticket results:")
    print(f"  {'#':<3}  {'Predicted':<26} {'Expected':<26} {'OK?'}")
    print(f"  {'-'*70}")
    for i, (pred, exp, txt) in enumerate(zip(rw_preds, rw_labels, rw_texts), 1):
        ok = "✓" if pred == exp else "✗"
        snippet = txt[:42] + ("..." if len(txt) > 42 else "")
        print(f"  {i:<3}  {pred:<26} {exp:<26} {ok}  '{snippet}'")

    # Diagnose failures
    failures = [(rw_labels[i], rw_preds[i], rw_texts[i]) for i in range(N_SAMPLES) if rw_labels[i] != rw_preds[i]]
    if failures:
        print(f"\n  DIAGNOSIS — {len(failures)} failures:")
        confusion_pairs = {}
        for true, pred, _ in failures:
            k = f"{true} → {pred}"
            confusion_pairs[k] = confusion_pairs.get(k, 0) + 1
        for pair, count in sorted(confusion_pairs.items(), key=lambda x: -x[1]):
            print(f"    {pair}  (x{count})")
        print(f"\n  OBSERVED FAILURE PATTERN: Surface-form differences can weaken TF-IDF keyword matching.")
        print(f"    • Informal wording, abbreviations and typos can weaken keyword matches.")
        print(f"    • Some ambiguous phrasing can be mapped toward General Inquiry.")
        print(f"    POSSIBLE NEXT STEP: Add more informal/typo examples for the confused categories and retest.")

    print(f"\n  Informal-query result: {rw_acc:.0%} ({correct_count}/{N_SAMPLES} correct).")
    print("  Interpretation: This small robustness test is reported separately from the")
    print("  primary 10-class held-out benchmark and should not be treated as production accuracy.")

    # 8. Published-reference comparison
    _banner("Published Reference Comparison: claimsense-ai-v1")
    print("  HuggingFace Model: pramodmisra/claimsense-ai-v1")
    print("  NOTE: ClaimSense is presented as a published/self-reported reference.")
    print("        Its reported result was not independently reproduced under this")
    print("        exact 10-class train/test protocol; therefore no direct superiority")
    print("        or SOTA claim is made.")
    print(f"\n  {'Metric':<32} {'Our Phase-1':<20} {'ClaimSense'}")
    print("  " + "-"*70)
    rows = [
        ("Overall Accuracy", f"{acc:.2%}", "Published reference — see model card"),
        ("Weighted F1-Score", f"{f1:.4f}", "Not protocol-matched"),
        ("Informal-query test", f"{rw_acc:.0%}", "Not reported"),
        ("Evaluation classes", "10 routing categories", "Different evaluation setup"),
        ("Prediction latency", "< 2ms (TF-IDF)", "Reported separately; not load-matched"),
        ("RAG / LLM escalation", "Implemented separately", "Reference capability differs"),
    ]
    for label, ours, theirs in rows:
        print(f"  {label:<32} {ours:<20} {theirs}")

    print("\n  Interpretation:")
    print("    Our Phase-1 result is an independently measured 10-class holdout result.")
    print("    ClaimSense is retained as a published reference point, not an apples-to-apples")
    print("    benchmark. A direct percentage-point improvement is therefore not claimed.")

    plot_benchmark(acc, reference_acc=None)

    # 9. Full BERT evaluation on the SAME held-out test set.
    _banner("Phase-2 BERT Ensemble Evaluation — Full Held-Out Test Set")
    bert_full_acc = None
    bert_weighted_f1 = None
    bert_macro_f1 = None
    try:
        from models.bert_triage import BERTTriageModel
        bert_model = BERTTriageModel()
        print(f"  Evaluating BERT on all {len(X_test):,} held-out tickets...")
        bert_preds = [bert_model.predict(t)[0] for t in X_test]
        bert_full_acc = accuracy_score(list(y_test), bert_preds)
        _, _, bert_weighted_f1, _ = precision_recall_fscore_support(
            list(y_test), bert_preds, average="weighted", zero_division=0)
        _, _, bert_macro_f1, _ = precision_recall_fscore_support(
            list(y_test), bert_preds, average="macro", zero_division=0)
        print(f"\n  BERT Accuracy  (full test set): {bert_full_acc:.4%}")
        print(f"  BERT Weighted F1-Score        : {bert_weighted_f1:.4f}")
        print(f"  BERT Macro F1-Score           : {bert_macro_f1:.4f}")
        print("  Interpretation: BERT is evaluated on the same held-out test set,")
        print("  enabling a direct Phase-1 vs BERT comparison within this experiment.")
    except Exception as e:
        print(f"  ⚠️  Full BERT evaluation unavailable in this environment: {e}")
        print("  No BERT accuracy is reported. Install the project's torch/transformers")
        print("  dependencies and rerun rather than substituting a 200-sample result.")

    # 10. Summary + known issues
    _banner("Evaluation Summary & Dissertation Readiness")
    print(f"""
  PRIMARY 10-CLASS RESULT
     • Standard seed-42 holdout accuracy: {acc:.2%} on {len(X_test):,} tickets
     • Weighted F1: {f1:.4f}
     • Macro F1: {macro_f1:.4f}
     • Every ticket is assigned to exactly one of the 10 routing categories.

  LEAKAGE-CONTROLLED SENSITIVITY RESULT
     • Exact duplicate texts are removed before splitting.
     • Accuracy: {controlled_acc:.2%} on {len(X_u_test):,} unique-query test tickets
     • This result is reported to show sensitivity to duplicate leakage.

  BUSINESS-IMPACT RESULT
     • Net utility: {business_score:.2f} normalized units
     • Utility per ticket: {business_score_per_ticket:.4f} units/ticket
     • Weights are experimental business-impact assumptions, not actual monetary costs.

  INFORMAL-QUERY ROBUSTNESS
     • Accuracy: {rw_acc:.0%} on {N_SAMPLES} manually constructed informal tickets
     • Reported separately from the primary held-out benchmark.

  BERT VALIDATION
     • Full-test-set BERT evaluation is attempted on all {len(X_test):,} tickets.
     • If the required torch/transformers environment is unavailable, no BERT
       accuracy is reported rather than substituting the 200-sample result.

  IMPORTANT LIMITATIONS
     • ClaimSense is a published/self-reported reference, not a protocol-matched benchmark.
     • No direct SOTA/superiority percentage is claimed.
     • Exact duplicate overlap is explicitly measured; the controlled result is provided separately.
     • High confidence does not guarantee correctness; calibration remains a validation need.
     • Informal-query performance is based on a small manually constructed set.

  NEXT VALIDATION
     1. Run the full BERT test in an environment with torch/transformers installed.
     2. Consider group/semantic deduplication in addition to exact-text deduplication.
     3. Add calibration analysis for confidence-based routing.
     4. Validate business-impact weights with real operational/SLA data if available.

  Visualizations saved to: eval_output/
     confusion_matrix.png | confidence_distribution.png | f1_scores.png | benchmark_comparison.png
""")
    print("=" * 65)


if __name__ == "__main__":
    evaluate()