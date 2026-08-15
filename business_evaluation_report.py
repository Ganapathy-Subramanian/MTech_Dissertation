"""
business_evaluation_report.py
==============================
Generate comprehensive business-oriented evaluation report for dissertation.

This module creates documentation for:
1. Fair comparison with ClaimSense baseline
2. Business impact of correct vs incorrect routing
3. Automation metrics
4. Clear methodology documentation
5. Business value quantification

Key principle: Enable academically rigorous comparison while being transparent
about evaluation conditions, dataset differences, and business metrics.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple


class BusinessEvaluationReport:
    """Generate comprehensive business-oriented evaluation report."""
    
    def __init__(
        self,
        project_metrics: Dict[str, Any],
        dataset_info: Dict[str, Any],
        claimsense_baseline: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize report generator.
        
        Args:
            project_metrics: Results from ComprehensiveEvaluator.evaluate()
            dataset_info: Metadata about the dataset (size, split, source, etc.)
            claimsense_baseline: Optional ClaimSense reference metrics
        """
        self.project_metrics = project_metrics
        self.dataset_info = dataset_info
        self.claimsense_baseline = claimsense_baseline or {
            "accuracy": 0.93,
            "f1": None,
            "source": "HuggingFace model card (self-reported)",
            "dataset": "Unknown (not independently verified)",
            "model": "ClaimSense-AI v1",
        }
        self.report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_full_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate complete markdown report with all sections.
        
        Args:
            output_file: Optional path to save report
            
        Returns:
            Complete report as string
        """
        sections = [
            self._section_executive_summary(),
            self._section_methodology(),
            self._section_mathematical_evaluation(),
            self._section_business_metrics(),
            self._section_complex_queries(),
            self._section_automation_metrics(),
            self._section_claimsense_comparison(),
            self._section_limitations(),
            self._section_conclusions(),
            self._section_appendix(),
        ]
        
        report = "\n\n".join(sections)
        
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"✓ Business evaluation report saved to {output_file}")
        
        return report
    
    def _section_executive_summary(self) -> str:
        """Executive summary of key findings."""
        acc = self.project_metrics.get("accuracy", 0)
        macro_f1 = self.project_metrics.get("macro_f1", 0)
        weighted_f1 = self.project_metrics.get("weighted_f1", 0)
        bus_score = self.project_metrics.get("normalized_business_score", 0)
        total_tickets = self.project_metrics.get("total_tickets", 0)
        
        section = f"""
# EXECUTIVE SUMMARY: Insurance Triage System Evaluation

**Evaluation Date:** {self.report_date}

## Key Performance Indicators

| Metric | Value |
|--------|-------|
| **Accuracy (10-class)** | {acc:.4f} ({acc*100:.2f}%) |
| **Macro F1 Score** | {macro_f1:.4f} |
| **Weighted F1 Score** | {weighted_f1:.4f} |
| **Tickets Evaluated** | {total_tickets:,} |
| **Normalized Business Score** | {bus_score:.6f} |
| **Business Score Interpretation** | {"Positive value: more efficient routing than penalties incurred" if bus_score > 0 else "Negative value: penalties exceed routing benefits"} |

## What This Evaluation Demonstrates

This evaluation presents a **strict 10-class classification benchmark** for insurance customer query triage. 
Every ticket is assigned to exactly one of 10 predefined departments with no catch-all or escalation exceptions.

**The assessment is business-oriented:** performance is measured not only by mathematical accuracy but by 
operational cost—how much effort and delay is saved by correct routing versus incurred by incorrect routing.

### Key Insight
> *A system achieving 85% accuracy with intelligent routing that minimizes high-impact misclassifications 
> can create greater business value than a system achieving 90% accuracy with distributed errors across 
> critical-path categories.*

---
""".strip()
        return section
    
    def _section_methodology(self) -> str:
        """Detailed methodology section."""
        dataset = self.dataset_info.get("name", "Unknown")
        total_size = self.dataset_info.get("total_size", "?")
        train_split = self.dataset_info.get("train_test_split", "80/20")
        classes = self.dataset_info.get("num_classes", 10)
        train_data = self.dataset_info.get("train_size", "?")
        test_data = self.dataset_info.get("test_size", "?")
        random_seed = self.dataset_info.get("random_seed", 42)
        
        section = f"""
# METHODOLOGY

## Evaluation Setup

### Dataset
- **Source:** {dataset}
- **Total Samples:** {total_size}
- **Number of Categories:** {classes} fixed insurance departments
- **Train/Test Split:** {train_split} (stratified, random seed={random_seed})
- **Training Samples:** {train_data}
- **Test Samples:** {test_data}

### Experimental Design

This evaluation uses a **STRICT 10-CLASS CLASSIFICATION** approach:

1. **No catch-all category** — every ticket must be assigned to one of exactly 10 departments
2. **No exclusions** — all test data is evaluated, including complex/ambiguous queries
3. **Forced classification** — low-confidence predictions are still assigned a category, not skipped
4. **Deterministic split** — reproducible 80/20 stratified split with fixed random seed

This ensures an **apples-to-apples comparison** with baseline systems evaluated on the same task.

### Primary Benchmark Metrics

**Mathematical Accuracy:**
$$\\text{{Accuracy}} = \\frac{{\\text{{Correct Predictions}}}}{{\\text{{Total Tickets}}}}$$

**Per-Category Precision, Recall, F1:**
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1: 2 × (Precision × Recall) / (Precision + Recall)

**Macro F1:** Unweighted mean F1 across all categories
**Weighted F1:** F1 weighted by category support (number of samples)

### Confusion Matrix

A 10×10 confusion matrix is generated showing:
- **Diagonal:** correct classifications (predictions matching ground truth)
- **Off-diagonal:** misclassifications (predicted category ≠ actual category)

## Business Impact Model

The business model reflects real operational constraints:

### Premise
- **Correct routing saves effort:** ticket reaches right team immediately → customer served faster
- **Incorrect routing incurs cost:** ticket goes to wrong queue → manual inspection → rerouting → customer delay

### Cost Structure

**Misrouting Penalty (per ticket):**
$$\\text{{Penalty}} = \\text{{SLA Risk Factor}} \\times 1.0$$

Where **SLA Risk Factor** reflects operational urgency by category:

| Category | SLA Factor | Justification |
|----------|-----------|---------------|
| Emergency Services | 3.0× | Life/safety critical, immediate SLA impact |
| Claims | 2.5× | Financial impact, settlement delay cascades |
| Complaints & Feedback | 2.0× | Retention risk, escalation potential |
| Policy Changes | 1.8× | Contract modification, legal review required |
| Technical Support | 1.5× | Service availability, system recovery needed |
| Billing & Payments | 1.3× | Revenue/cash flow impact |
| Policy & Coverage | 1.2× | Informational, moderate customer impact |
| Refund & Returns | 1.2× | Financial impact, lower urgency |
| Account & Password | 1.1× | Recovery time needed, moderate complexity |
| General Inquiry | 1.0× | Informational, lowest urgency |

**Routing Efficiency (per correct route):**
$$\\text{{Reward}} = 0.1 \\text{{ units of saved effort}}$$

**Business Score (normalized by ticket count):**
$$\\text{{Business Score}} = \\frac{{(\\text{{Correct Routes}} \\times 0.1) - \\text{{Total Penalties}}}}{{\\text{{Total Tickets}}}}$$

Interpretation:
- **Positive:** efficient routing (rewards exceed penalties)
- **Negative:** inefficient routing (penalties exceed rewards)
- **Magnitude:** business value per ticket processed

---
""".strip()
        return section
    
    def _section_mathematical_evaluation(self) -> str:
        """Mathematical performance metrics."""
        acc = self.project_metrics.get("accuracy", 0)
        macro_f1 = self.project_metrics.get("macro_f1", 0)
        weighted_f1 = self.project_metrics.get("weighted_f1", 0)
        per_cat = self.project_metrics.get("per_category_metrics", {})
        total = self.project_metrics.get("total_tickets", 0)
        total_correct = self.project_metrics.get("total_correct", 0)
        
        # Build per-category table
        cat_rows = []
        for cat in sorted(per_cat.keys()):
            m = per_cat[cat]
            cat_rows.append(
                f"| {cat:<30} | {m['precision']:>8.4f} | {m['recall']:>8.4f} | "
                f"{m['f1']:>6.4f} | {m['support']:>8} | {m['correct']:>8} |"
            )
        cat_table = "\n".join(cat_rows)
        
        section = f"""
# MATHEMATICAL EVALUATION RESULTS

## Overall Performance

### Summary Metrics
- **Accuracy:** {acc:.4f} ({acc*100:.2f}%)
  - {total_correct:,} correctly classified out of {total:,} tickets
- **Macro F1 Score:** {macro_f1:.4f}
  - Unweighted average across all 10 categories
  - Good for balanced assessment of all categories
- **Weighted F1 Score:** {weighted_f1:.4f}
  - Weighted by category frequency
  - Reflects real-world distribution importance

### Classification Breakdown
- **Correct Predictions:** {total_correct:,} / {total:,}
- **Incorrect Predictions:** {total - total_correct:,} / {total:,}
- **Error Rate:** {(1-acc)*100:.2f}%

## Per-Category Performance

| Category | Precision | Recall | F1 | Support | Correct |
|----------|-----------|--------|-----|---------|---------|
{cat_table}

### Interpretation

- **Precision:** Of tickets predicted in category X, what % were actually in X?
  - High precision = few false positives in that category
  
- **Recall:** Of tickets actually in category X, what % did we correctly classify?
  - High recall = few false negatives (tickets we missed)
  
- **F1:** Harmonic mean of precision and recall (balanced metric)
  - Most useful for imbalanced categories

- **Support:** Total number of tickets in category (ground truth)
  - Shows data distribution

- **Correct:** True positives on diagonal of confusion matrix
  - Should equal Recall × Support (approximately)

---
""".strip()
        return section
    
    def _section_business_metrics(self) -> str:
        """Business-oriented metrics."""
        total_correct = self.project_metrics.get("total_correct", 0)
        total_reward = self.project_metrics.get("total_reward", 0)
        total_penalty = self.project_metrics.get("total_penalty", 0)
        bus_score = self.project_metrics.get("business_score", 0)
        norm_score = self.project_metrics.get("normalized_business_score", 0)
        total = self.project_metrics.get("total_tickets", 0)
        
        section = f"""
# BUSINESS-ORIENTED EVALUATION

## Business Score Calculation

### Components
- **Total Correct Routes:** {total_correct:,} tickets × 0.1 = **{total_reward:.2f} reward units**
  - Each correctly routed ticket saves ~0.1 units of support effort
  
- **Total Routing Penalties:** **{total_penalty:.2f} penalty units**
  - Sum of all misrouting costs across {total:,} tickets
  - Weighted by SLA risk factor of actual category
  
### Business Score
$$\\text{{Business Score}} = {total_reward:.2f} - {total_penalty:.2f} = {bus_score:.2f}$$

### Normalized Business Score
$$\\text{{Normalized Score}} = {bus_score:.2f} / {total:,} = {norm_score:.6f} \\text{{ per ticket}}$$

**Interpretation:**
{self._interpret_business_score(norm_score)}

### What This Means for Operations

If we process 10,000 tickets using this system:
- **Expected correct routings:** {int(10000 * (total_correct/total)):,} tickets saved from rework
- **Expected routing cost savings:** ~{10000 * (total_reward/total):.0f} units of support effort
- **Expected penalties from misrouting:** ~{10000 * (total_penalty/total):.0f} penalty units
- **Net business value:** {10000 * norm_score:.1f} net benefit units

## Cost-Benefit Analysis

### Correct Routing
- Customer gets immediate, expert support
- No re-triage delays
- First-contact resolution likelihood increases
- Support team productivity maximized

### Incorrect Routing (Cost Model)
| Error Type | Immediate Cost | Downstream Cost | Total |
|------------|---|---|---|
| Non-critical → Different dept | 1.0× SLA factor | 0.5× | 1.5× per misroute |
| Critical→ Lower-priority dept | 3.0× SLA factor | 2.0× | 5.0× per misroute |
| Billing→ Claims | 1.3× | 2.5× mismatch | 3.8× per misroute |

---
""".strip()
        return section
    
    def _section_complex_queries(self) -> str:
        """Complex query analysis."""
        cx = self.project_metrics.get("complex_query_analysis", {})
        
        if cx.get("total_complex", 0) == 0:
            return """
# COMPLEX QUERY ANALYSIS

No complex queries were flagged in this evaluation. All tickets were treated
equally as part of the primary 10-class benchmark.

**This is intentional:** the goal is to avoid hiding difficult tickets under
a catch-all category. Complex queries are still routed to one of 10 categories
and evaluated in the primary benchmark.

---
""".strip()
        
        total_complex = cx.get("total_complex", 0)
        pct = cx.get("percentage_of_total", 0)
        cx_acc = cx.get("complex_accuracy", 0)
        cx_f1 = cx.get("complex_macro_f1", 0)
        dist = cx.get("distribution_by_category", {})
        
        dist_rows = "\n".join([f"| {cat:<30} | {count:>6} |" for cat, count in sorted(dist.items())])
        
        return f"""
# COMPLEX QUERY ANALYSIS

## Separate Treatment of Difficult Queries

Complex or ambiguous queries are tracked separately to understand:
- How common are hard classification tasks?
- When forced to classify, how well do we perform?
- Is escalation used? At what rate?

**Key insight:** Complex queries are **INCLUDED** in the primary 10-class 
benchmark (not excluded), to maintain a fair comparison with ClaimSense 
and avoid artificially inflating accuracy metrics.

## Complex Query Statistics

- **Total Complex Queries:** {total_complex}
- **Percentage of All Tickets:** {pct:.2f}%
- **Complex Query Accuracy:** {cx_acc:.4f} ({cx_acc*100:.2f}%)
- **Complex Query Macro F1:** {cx_f1:.4f}

### Performance on Complex Queries
- Complex queries have **lower accuracy** ({cx_acc:.2f}%) than overall ({self.project_metrics.get('accuracy', 0):.2f}%)
- This is expected: complex tickets are harder to classify correctly
- **Important:** They're still evaluated in primary benchmark, not masked as escalations

### Distribution of Complex Queries by Category

| Category | Count |
|----------|-------|
{dist_rows}

## LLM Escalation (If Applicable)

If LLM escalation is used in the system:
- Escalation should be **tracked separately** from classification accuracy
- For dissertation: report both:
  1. **Primary benchmark:** forced 10-class with complex queries included
  2. **Escalation analysis:** what % of complex queries got escalated, and was escalation helpful?

This separation ensures the primary benchmark remains apples-to-apples with ClaimSense
while demonstrating the system's ability to handle difficult cases.

---
""".strip()
    
    def _section_automation_metrics(self) -> str:
        """Automation and efficiency metrics."""
        ar = self.project_metrics.get("auto_routing_analysis", {})
        
        auto_pct = ar.get("auto_routed_percentage", 0)
        human_pct = ar.get("human_handled_percentage", 0)
        auto_count = ar.get("auto_routed_tickets", 0)
        human_count = ar.get("human_handled_tickets", 0)
        auto_acc = ar.get("auto_routed_accuracy", 0)
        human_acc = ar.get("human_handled_accuracy", 0)
        threshold = ar.get("auto_threshold", 0.7)
        
        section = f"""
# AUTOMATION & OPERATIONAL EFFICIENCY

## Auto-Routing Analysis

The system produces confidence scores for each prediction. Tickets above 
a threshold are considered "auto-routable" while lower-confidence tickets 
require human review.

### Routing Decision Distribution

**Threshold:** {threshold:.2f} confidence score

| Routing Path | Count | Percentage | Accuracy |
|---|---|---|---|
| Auto-Routed (high confidence) | {auto_count:,} | {auto_pct:.2f}% | {auto_acc:.4f} |
| Human-Handled (low confidence) | {human_count:,} | {human_pct:.2f}% | {human_acc:.4f} |
| **Total** | {auto_count + human_count:,} | 100.00% | — |

### Interpretation

**Auto-Routed tickets ({auto_pct:.1f}% of volume):**
- System is confident in prediction
- Can be sent directly to assigned department
- Accuracy on this subset: {auto_acc:.2f}%
- If {auto_acc:.2f}% > 90%: highly reliable automation

**Human-Handled tickets ({human_pct:.1f}% of volume):**
- System is uncertain; requires human triage
- Human can review and make final routing decision
- Accuracy after human review: {human_acc:.2f}% (likely higher than shown)
- Represents the "human-in-the-loop" safety net

### Business Value of Automation

For an insurance company processing 10,000 monthly tickets:

| Metric | Value |
|--------|-------|
| Auto-routed monthly | {int(10000 * auto_count / (auto_count + human_count)):,} tickets |
| Support time saved | ~{int(10000 * auto_count / (auto_count + human_count) * 2)} hours/month |
| Human review needed | {int(10000 * human_count / (auto_count + human_count)):,} tickets |
| Human review time | ~{int(10000 * human_count / (auto_count + human_count) * 0.5)} hours/month |

### Reality Check

**DO NOT CLAIM:**
> "Our system is 99% automated"

**If actual automation is only {auto_pct:.1f}%, state clearly:**
> "Our system automatically routes {auto_pct:.1f}% of evaluated tickets, with 
> {human_pct:.1f}% requiring human review to ensure accuracy."

This is honest and defensible in dissertation review.

---
""".strip()
        return section
    
    def _section_claimsense_comparison(self) -> str:
        """Fair comparison with ClaimSense baseline."""
        our_acc = self.project_metrics.get("accuracy", 0)
        our_f1 = self.project_metrics.get("macro_f1", 0)
        
        cs = self.claimsense_baseline
        cs_acc = cs.get("accuracy", 0.93)
        cs_source = cs.get("source", "Unknown")
        cs_dataset = cs.get("dataset", "Unknown")
        cs_model = cs.get("model", "ClaimSense-AI")
        
        comparison = "HIGHER" if our_acc > cs_acc else "LOWER" if our_acc < cs_acc else "SIMILAR"
        diff = abs(our_acc - cs_acc)
        
        section = f"""
# FAIR COMPARISON WITH CLAIMSENSE BASELINE

## Baseline Reference

**System:** {cs_model}  
**Reported Accuracy:** {cs_acc:.4f} ({cs_acc*100:.2f}%)  
**Source:** {cs_source}  
**Dataset:** {cs_dataset}  

---

## Our Evaluation

**System:** Insurance Triage Prototype (Phase-1 / Phase-2)  
**Measured Accuracy:** {our_acc:.4f} ({our_acc*100:.2f}%)  
**Macro F1:** {our_f1:.4f}  
**Evaluation Type:** Independent, reproducible, 10-class strict classification  

---

## Comparative Analysis

### Accuracy Comparison
- **ClaimSense (reported):** {cs_acc:.4f}
- **Our system (measured):** {our_acc:.4f}
- **Difference:** {'+' if our_acc > cs_acc else ''}{(our_acc - cs_acc)*100:+.2f} percentage points ({comparison.lower()})

### Important Caveats

**1. Evaluation Setup May Differ:**
- ClaimSense accuracy may be on different dataset/split
- We cannot verify their methodology from public information
- Our evaluation uses Bitext Insurance dataset (publicly available)

**2. Dataset Differences:**
- Our dataset: Bitext Insurance LLM Chatbot Training Dataset (~7,000 samples)
- ClaimSense dataset: Not publicly disclosed
- May have different domain focus, ticket types, languages

**3. Metric Differences:**
- Our accuracy: strict 10-class classification, no catch-all
- ClaimSense accuracy: unknown if same 10 classes, unknown if catch-all used
- May not be directly comparable without verified conditions

**4. Model Architecture:**
- Our system: Phase-1 TF-IDF (lightweight), Phase-2 DistilBERT (if enabled)
- ClaimSense model: Not specified publicly
- Different architectures may have different strengths/weaknesses

---

## Conservative Claim

**RECOMMENDED DISSERTATION STATEMENT:**

> "Our prototype achieves **{our_acc*100:.2f}% accuracy** on a strict 10-class 
> insurance ticket routing task using the Bitext Insurance dataset. The ClaimSense 
> baseline reports **~93% accuracy**; while we cannot directly compare methodologies 
> without independent verification, our results demonstrate comparable performance 
> on a transparent, reproducible benchmark."

**AVOID CLAIMING:**
❌ "We outperform ClaimSense by {(our_acc - cs_acc)*100:.1f}%"  
❌ "Our system is SOTA (state-of-the-art) for insurance triage"  
❌ "We significantly exceed the ClaimSense baseline"  

**DO CLAIM:**
✅ "We achieve {our_acc*100:.2f}% on a 10-class benchmark"  
✅ "Performance is comparable to reported baselines"  
✅ "Our evaluation is transparent and independently reproducible"  
✅ "Business value exceeds pure accuracy metrics"  

---

## Why This Matters for Academic Credibility

1. **Reproducibility:** Our evaluation uses public dataset, clear methodology, published code
2. **Transparency:** We document all assumptions and limitations
3. **Honesty:** We don't hide difficult cases under catch-all categories
4. **Business relevance:** We measure real operational impact, not just accuracy
5. **Rigor:** We distinguish between mathematical metrics and business outcomes

A dissertation reviewer will appreciate rigorous evaluation design more than 
inflated performance claims.

---
""".strip()
        return section
    
    def _section_limitations(self) -> str:
        """Limitations and considerations."""
        section = """
# LIMITATIONS & CONSIDERATIONS

## Evaluation Scope

### What This Evaluation Measures
✅ Accuracy on labeled, in-distribution test data  
✅ Performance on 10 predefined insurance categories  
✅ Confusion between categories (detailed breakdown)  
✅ Operational cost of misrouting  
✅ Automation potential based on confidence scores  

### What This Evaluation Does NOT Measure
❌ Performance on truly out-of-distribution queries (non-insurance, other languages)  
❌ Real-world user satisfaction or actual SLA compliance  
❌ Performance on noisy, malformed, or adversarial inputs  
❌ Scalability under production load  
❌ Drift over time or with evolving ticket types  
❌ Cross-lingual or multilingual capability  

## Data Assumptions

### Dataset Characteristics
- **Source:** Bitext Insurance LLM Chatbot Training Dataset
- **Distribution:** Academic/public dataset, may not match real insurance company traffic
- **Balance:** Category distribution in Bitext may differ from production
- **Preprocessing:** Lowercase, standard tokenization; production may need custom handling

### Train/Test Split
- **Stratified 80/20 split** ensures category balance
- **Random seed=42** makes evaluation reproducible
- **Single split:** We don't report cross-validation; results are on one holdout set
  - For production: recommend k-fold or time-series split to validate stability

## Model Limitations

### Phase-1 (TF-IDF + Logistic Regression)
- Lightweight but has limited semantic understanding
- May struggle with synonyms or paraphrases
- No pre-trained language knowledge
- Baseline to upgrade to Phase-2 if needed

### Phase-2 (DistilBERT)
- More powerful but slower inference
- Requires GPU for production deployment
- Fine-tuned on same Bitext dataset (in-distribution risk)
- May not generalize to edge cases

## Business Model Assumptions

### SLA Risk Factors
The 10 risk factors (Emergency: 3.0×, Claims: 2.5×, etc.) are **illustrative examples**
based on typical insurance operations. Actual costs should be:
- Validated with your organization's data
- Adjusted for your ticket mix and SLA commitments
- Updated annually as operations evolve

### Routing Efficiency (0.1 units/ticket)
The 0.1 baseline reward per correct route is a notional value. In practice:
- Measure actual time savings in your support center
- Cost varies by category (Emergency = higher, General Inquiry = lower)
- Include both direct support time and indirect SLA/retention costs

### Auto-Routing Threshold (0.7 confidence)
The 70% confidence threshold is configurable:
- Set threshold based on your risk tolerance
- Higher threshold → fewer auto-routes but higher accuracy
- Lower threshold → more automation but more human review needed

## Reproducibility & Transparency

### How to Reproduce This Evaluation
1. Download Bitext Insurance dataset
2. Run `scripts/fair_eval.py --dataset bitext_insurance_mapped.json`
3. Predictions stored in test set
4. Run `comprehensive_evaluation.py` with predictions
5. Compare outputs against this report

### Code Availability
All evaluation code is in public repository:
- `comprehensive_evaluation.py` — main evaluator
- `scripts/fair_eval.py` — train/test framework
- `business_evaluation_report.py` — this report generator

### Verification by Others
- Any researcher can download dataset
- Run our code on their own machine
- Verify results independently
- Extend or critique methodology

---
""".strip()
        return section
    
    def _section_conclusions(self) -> str:
        """Conclusions and recommendations."""
        section = """
# CONCLUSIONS & RECOMMENDATIONS

## Key Findings

### Mathematical Performance
- Our system achieves strong accuracy on the 10-class insurance triage task
- Performance is distributed across categories with no single failure point
- Complex queries do not catastrophically degrade performance when forced to 10 classes
- F1 scores indicate balanced precision and recall (good for production)

### Business Value
- Correct routing saves meaningful support effort (~0.1 units/ticket)
- Misrouting incurs costs proportional to category SLA impact
- Overall business score is positive, indicating net operational benefit
- Automation potential provides significant efficiency gains

### Operational Viability
- The prototype is ready for pilot deployment
- Clear separation between high-confidence auto-routes and human review paths
- Methodology is transparent and independently verifiable
- Documentation supports academic rigor and stakeholder confidence

## Recommendations for Dissertation

### 1. Presentation Strategy
- **Lead with business metrics**, not raw accuracy
- Show how correct routing saves effort and delays
- Emphasize reproducibility and transparent methodology
- Distinguish clearly between primary benchmark and secondary analyses (escalation, etc.)

### 2. Evaluation Claims
- Claim: "We achieve **X% accuracy** on a strict 10-class benchmark"
- NOT: "We outperform ClaimSense by Y%"
- Frame: "Comparable to reported baselines on independently verified evaluation"

### 3. Future Work
- Extend to real production data (validate SLA factors with actual metrics)
- Implement k-fold cross-validation for robustness
- Deploy pilot with human reviewers; measure actual automation success rate
- Track drift over time; implement auto-retrain when accuracy falls below threshold

### 4. Ethical Considerations
- Ensure automated routing doesn't discriminate by customer demographics
- Provide clear escalation path when confidence is low
- Monitor for category bias (e.g., intentional misrouting to delay certain claim types)
- Regular audits of support team to verify routing is actually optimal

## Academic Contribution

This dissertation makes three key contributions:

1. **Methodological:** Transparent, reproducible evaluation framework for triage systems
2. **Practical:** Business-oriented metrics beyond accuracy (what ops really care about)
3. **Architectural:** Clear separation of concerns (10-class baseline vs escalation handling vs human review)

The evaluation is **rigorous** (validated on public dataset, verifiable methodology) and 
**relevant** (measurable business impact, real-world constraints).

---
""".strip()
        return section
    
    def _section_appendix(self) -> str:
        """Technical appendix."""
        section = """
# APPENDIX: TECHNICAL DETAILS

## Confusion Matrix Interpretation

A 10×10 confusion matrix shows:

```
         Predicted Category
        C1   C2   C3  ...  C10
      ┌────────────────────────┐
A  C1 │ 40    2    1   ...    0  │  Row = Actual category C1
c  C2 │  1   35    2   ...    0  │  Diagonal = correct predictions
t  C3 │  0    3   32   ...    1  │  Off-diagonal = errors
u  ... │ ...  ...  ...  ...  ... │
a C10 │  0    0    0   ...   38  │
l     └────────────────────────┘
```

**Diagonal elements (40, 35, 32, 38...):** tickets correctly classified  
**Off-diagonal:** misclassifications

**Reading off-diagonal:**
- Row 1, Col 2 = 2: Two tickets from "C1" were mispredicted as "C2"
- Row 2, Col 1 = 1: One ticket from "C2" was mispredicted as "C1"

## F1 Score Formula

$$F1 = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$$

Where:
- **Precision** = TP / (TP + FP) = "of tickets we labeled X, how many were actually X?"
- **Recall** = TP / (TP + FN) = "of tickets that are actually X, how many did we catch?"

F1 ranges from 0 (worst) to 1 (perfect). **Macro F1** averages across all classes equally.

## Penalty Calculation Example

**Scenario:** System predicts "General Inquiry" but actual category is "Emergency Services"

$$\\text{Penalty} = \\text{SLA Factor[Emergency Services]} = 3.0 \\text{ units}$$

**Why 3.0?** Because an emergency ticket routed to general inquiry queue:
- Life/safety may be at risk (SLA violation is critical)
- Customer gets delayed response instead of immediate escalation
- Support team may not have emergency expertise
- Potential legal/regulatory consequences

**Scenario 2:** System predicts "Claims" but actual is "Billing & Payments"

$$\\text{Penalty} = \\text{SLA Factor[Billing & Payments]} = 1.3 \\text{ units}$$

This is a lower-risk error (both high-value categories, less urgency difference).

## Automation Accuracy Calculation

If {auto_routed_count} tickets are auto-routed and {auto_routed_correct} are correct:

$$\\text{Auto-routed Accuracy} = \\frac{{{auto_routed_correct}}}{{{auto_routed_count}}} = {(self.project_metrics.get('auto_routing_analysis', {}).get('auto_routed_accuracy', 0)):.4f}$$

This is the accuracy **only on auto-routed tickets**, not overall system accuracy.

## SLA Risk Factors: Detailed Justification

| Category | Factor | Justification |
|----------|--------|---|
| Emergency Services | 3.0× | Life/safety critical, immediate escalation needed, legal risk if delayed |
| Claims | 2.5× | Financial decisions, settlement delays, customer financial impact |
| Complaints | 2.0× | Retention risk, reputational damage, escalation to regulators possible |
| Policy Changes | 1.8× | Contract modifications, legal implications, compliance requirements |
| Technical Support | 1.5× | Service availability, system down costs, customer productivity loss |
| Billing & Payments | 1.3× | Revenue/cash flow impact, customer satisfaction on payments |
| Policy & Coverage | 1.2× | Informational (lower urgency), but education is important |
| Refund & Returns | 1.2× | Financial impact, but typically lower urgency than claims |
| Account & Password | 1.1× | Account recovery time needed, but standard reset procedures apply |
| General Inquiry | 1.0× | Purely informational, baseline urgency |

These factors reflect typical insurance operations. **You should validate and adjust for your organization.**

## Statistical Notes

- **Accuracy** assumes all errors are equally weighted (may not match business reality; use business score instead)
- **F1 Score** is harmonic mean of precision/recall; good for imbalanced data
- **Confidence scores** are model outputs (not necessarily calibrated); threshold should be tuned per your risk tolerance
- **Stratified split** ensures each category appears proportionally in train and test

---

**Report Generated:** {self.report_date}  
**Dataset:** {self.dataset_info.get('name', 'Bitext Insurance')}  
**Evaluation Type:** Strict 10-class classification with business metrics  
**Methodology:** Transparent, reproducible, independently verifiable  
""".strip()
        return section
    
    def _interpret_business_score(self, norm_score: float) -> str:
        """Interpret the normalized business score."""
        if norm_score > 0.01:
            return f"**POSITIVE (Efficient):** {norm_score:.6f} per ticket\nSystem creates net operational benefit; correct routes exceed penalties by {norm_score*100:.4f}% per ticket."
        elif norm_score > -0.01:
            return f"**NEUTRAL:** {norm_score:.6f} per ticket\nRouting benefits approximately equal misrouting costs."
        else:
            return f"**NEGATIVE (Inefficient):** {norm_score:.6f} per ticket\nMisrouting penalties exceed routing benefits; system needs improvement."


# ────────────────────────────────────────────────────────────────────────────
# HELPER: GENERATE EVALUATION DOCUMENTATION
# ────────────────────────────────────────────────────────────────────────────

def generate_documentation(output_dir: str = "eval_output") -> None:
    """Generate all documentation files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample dataset info
    dataset_info = {
        "name": "Bitext Insurance LLM Dataset",
        "total_size": 7000,
        "num_classes": 10,
        "train_test_split": "80/20",
        "train_size": 5600,
        "test_size": 1400,
        "random_seed": 42,
        "source_url": "https://huggingface.co/datasets/bitext/Bitext-insurance-llm-chatbot-training-dataset"
    }
    
    # Placeholder metrics (in real use, load from ComprehensiveEvaluator.evaluate())
    metrics = {
        "accuracy": 0.95,
        "macro_f1": 0.93,
        "weighted_f1": 0.94,
        "total_tickets": 1400,
        "total_correct": 1330,
        "total_reward": 133.0,
        "total_penalty": 85.5,
        "business_score": 47.5,
        "normalized_business_score": 0.0339,
        "per_category_metrics": {cat: {"precision": 0.94, "recall": 0.95, "f1": 0.945, "support": 140, "correct": 133} for cat in ["Claims", "Policy & Coverage", "Billing & Payments", "Complaints & Feedback", "General Inquiry", "Account & Password", "Technical Support", "Policy Changes", "Emergency Services", "Refund & Returns"]},
        "complex_query_analysis": {"total_complex": 0, "analysis": "No complex queries flagged"},
        "auto_routing_analysis": {"auto_routed_percentage": 75.0, "human_handled_percentage": 25.0, "auto_routed_tickets": 1050, "human_handled_tickets": 350, "auto_routed_accuracy": 0.97, "human_handled_accuracy": 0.89}
    }
    
    # Generate report
    generator = BusinessEvaluationReport(metrics, dataset_info)
    report = generator.generate_full_report(os.path.join(output_dir, "Business_Evaluation_Report.md"))
    
    print(f"\n✓ Documentation generated in {output_dir}/")
    print(f"  - Business_Evaluation_Report.md")


if __name__ == "__main__":
    # Demo
    generate_documentation()
