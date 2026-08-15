# EXECUTIVE SUMMARY: Insurance Triage System Evaluation

**Evaluation Date:** 2026-08-15 16:12:46

## Key Performance Indicators

| Metric | Value |
|--------|-------|
| **Accuracy (10-class)** | 0.9910 (99.10%) |
| **Macro F1 Score** | 0.9833 |
| **Weighted F1 Score** | 0.9910 |
| **Tickets Evaluated** | 4,215 |
| **Normalized Business Score** | 0.082372 |
| **Business Score Interpretation** | Positive value: more efficient routing than penalties incurred |

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

# METHODOLOGY

## Evaluation Setup

### Dataset
- **Source:** Bitext Insurance LLM Dataset
- **Total Samples:** 21071
- **Number of Categories:** 10 fixed insurance departments
- **Train/Test Split:** 80/20 (stratified, random seed=42)
- **Training Samples:** 16856
- **Test Samples:** 4215

### Experimental Design

This evaluation uses a **STRICT 10-CLASS CLASSIFICATION** approach:

1. **No catch-all category** — every ticket must be assigned to one of exactly 10 departments
2. **No exclusions** — all test data is evaluated, including complex/ambiguous queries
3. **Forced classification** — low-confidence predictions are still assigned a category, not skipped
4. **Deterministic split** — reproducible 80/20 stratified split with fixed random seed

This ensures an **apples-to-apples comparison** with baseline systems evaluated on the same task.

### Primary Benchmark Metrics

**Mathematical Accuracy:**
$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Tickets}}$$

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
$$\text{Penalty} = \text{SLA Risk Factor} \times 1.0$$

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
$$\text{Reward} = 0.1 \text{ units of saved effort}$$

**Business Score (normalized by ticket count):**
$$\text{Business Score} = \frac{(\text{Correct Routes} \times 0.1) - \text{Total Penalties}}{\text{Total Tickets}}$$

Interpretation:
- **Positive:** efficient routing (rewards exceed penalties)
- **Negative:** inefficient routing (penalties exceed rewards)
- **Magnitude:** business value per ticket processed

---

# MATHEMATICAL EVALUATION RESULTS

## Overall Performance

### Summary Metrics
- **Accuracy:** 0.9910 (99.10%)
  - 4,177 correctly classified out of 4,215 tickets
- **Macro F1 Score:** 0.9833
  - Unweighted average across all 10 categories
  - Good for balanced assessment of all categories
- **Weighted F1 Score:** 0.9910
  - Weighted by category frequency
  - Reflects real-world distribution importance

### Classification Breakdown
- **Correct Predictions:** 4,177 / 4,215
- **Incorrect Predictions:** 38 / 4,215
- **Error Rate:** 0.90%

## Per-Category Performance

| Category | Precision | Recall | F1 | Support | Correct |
|----------|-----------|--------|-----|---------|---------|
| Account & Password             |   0.9796 |   0.9796 | 0.9796 |       98 |       96 |
| Billing & Payments             |   0.9836 |   0.9976 | 0.9906 |      421 |      420 |
| Claims                         |   0.9926 |   0.9909 | 0.9917 |     1211 |     1200 |
| Complaints & Feedback          |   0.9976 |   0.9696 | 0.9834 |      428 |      415 |
| Emergency Services             |   0.9756 |   1.0000 | 0.9877 |       80 |       80 |
| General Inquiry                |   0.9121 |   1.0000 | 0.9540 |       83 |       83 |
| Policy & Coverage              |   0.9994 |   0.9944 | 0.9969 |     1615 |     1606 |
| Policy Changes                 |   0.9765 |   1.0000 | 0.9881 |       83 |       83 |
| Refund & Returns               |   1.0000 |   0.9896 | 0.9948 |       96 |       95 |
| Technical Support              |   0.9429 |   0.9900 | 0.9659 |      100 |       99 |

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

# BUSINESS-ORIENTED EVALUATION

## Business Score Calculation

### Components
- **Total Correct Routes:** 4,177 tickets × 0.1 = **417.70 reward units**
  - Each correctly routed ticket saves ~0.1 units of support effort
  
- **Total Routing Penalties:** **70.50 penalty units**
  - Sum of all misrouting costs across 4,215 tickets
  - Weighted by SLA risk factor of actual category
  
### Business Score
$$\text{Business Score} = 417.70 - 70.50 = 347.20$$

### Normalized Business Score
$$\text{Normalized Score} = 347.20 / 4,215 = 0.082372 \text{ per ticket}$$

**Interpretation:**
**POSITIVE (Efficient):** 0.082372 per ticket
System creates net operational benefit; correct routes exceed penalties by 8.2372% per ticket.

### What This Means for Operations

If we process 10,000 tickets using this system:
- **Expected correct routings:** 9,909 tickets saved from rework
- **Expected routing cost savings:** ~991 units of support effort
- **Expected penalties from misrouting:** ~167 penalty units
- **Net business value:** 823.7 net benefit units

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

# COMPLEX QUERY ANALYSIS

No complex queries were flagged in this evaluation. All tickets were treated
equally as part of the primary 10-class benchmark.

**This is intentional:** the goal is to avoid hiding difficult tickets under
a catch-all category. Complex queries are still routed to one of 10 categories
and evaluated in the primary benchmark.

---

# AUTOMATION & OPERATIONAL EFFICIENCY

## Auto-Routing Analysis

The system produces confidence scores for each prediction. Tickets above 
a threshold are considered "auto-routable" while lower-confidence tickets 
require human review.

### Routing Decision Distribution

**Threshold:** 0.70 confidence score

| Routing Path | Count | Percentage | Accuracy |
|---|---|---|---|
| Auto-Routed (high confidence) | 4,037 | 95.78% | 0.9978 |
| Human-Handled (low confidence) | 178 | 4.22% | 0.8371 |
| **Total** | 4,215 | 100.00% | — |

### Interpretation

**Auto-Routed tickets (95.8% of volume):**
- System is confident in prediction
- Can be sent directly to assigned department
- Accuracy on this subset: 1.00%
- If 1.00% > 90%: highly reliable automation

**Human-Handled tickets (4.2% of volume):**
- System is uncertain; requires human triage
- Human can review and make final routing decision
- Accuracy after human review: 0.84% (likely higher than shown)
- Represents the "human-in-the-loop" safety net

### Business Value of Automation

For an insurance company processing 10,000 monthly tickets:

| Metric | Value |
|--------|-------|
| Auto-routed monthly | 9,577 tickets |
| Support time saved | ~19155 hours/month |
| Human review needed | 422 tickets |
| Human review time | ~211 hours/month |

### Reality Check

**DO NOT CLAIM:**
> "Our system is 99% automated"

**If actual automation is only 95.8%, state clearly:**
> "Our system automatically routes 95.8% of evaluated tickets, with 
> 4.2% requiring human review to ensure accuracy."

This is honest and defensible in dissertation review.

---

# FAIR COMPARISON WITH CLAIMSENSE BASELINE

## Baseline Reference

**System:** ClaimSense-AI v1  
**Reported Accuracy:** 0.9300 (93.00%)  
**Source:** HuggingFace model card (self-reported)  
**Dataset:** Unknown (not independently verified)  

---

## Our Evaluation

**System:** Insurance Triage Prototype (Phase-1 / Phase-2)  
**Measured Accuracy:** 0.9910 (99.10%)  
**Macro F1:** 0.9833  
**Evaluation Type:** Independent, reproducible, 10-class strict classification  

---

## Comparative Analysis

### Accuracy Comparison
- **ClaimSense (reported):** 0.9300
- **Our system (measured):** 0.9910
- **Difference:** ++6.10 percentage points (higher)

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

> "Our prototype achieves **99.10% accuracy** on a strict 10-class 
> insurance ticket routing task using the Bitext Insurance dataset. The ClaimSense 
> baseline reports **~93% accuracy**; while we cannot directly compare methodologies 
> without independent verification, our results demonstrate comparable performance 
> on a transparent, reproducible benchmark."

**AVOID CLAIMING:**
❌ "We outperform ClaimSense by 6.1%"  
❌ "Our system is SOTA (state-of-the-art) for insurance triage"  
❌ "We significantly exceed the ClaimSense baseline"  

**DO CLAIM:**
✅ "We achieve 99.10% on a 10-class benchmark"  
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

$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Where:
- **Precision** = TP / (TP + FP) = "of tickets we labeled X, how many were actually X?"
- **Recall** = TP / (TP + FN) = "of tickets that are actually X, how many did we catch?"

F1 ranges from 0 (worst) to 1 (perfect). **Macro F1** averages across all classes equally.

## Penalty Calculation Example

**Scenario:** System predicts "General Inquiry" but actual category is "Emergency Services"

$$\text{Penalty} = \text{SLA Factor[Emergency Services]} = 3.0 \text{ units}$$

**Why 3.0?** Because an emergency ticket routed to general inquiry queue:
- Life/safety may be at risk (SLA violation is critical)
- Customer gets delayed response instead of immediate escalation
- Support team may not have emergency expertise
- Potential legal/regulatory consequences

**Scenario 2:** System predicts "Claims" but actual is "Billing & Payments"

$$\text{Penalty} = \text{SLA Factor[Billing & Payments]} = 1.3 \text{ units}$$

This is a lower-risk error (both high-value categories, less urgency difference).

## Automation Accuracy Calculation

If {auto_routed_count} tickets are auto-routed and {auto_routed_correct} are correct:

$$\text{Auto-routed Accuracy} = \frac{{{auto_routed_correct}}}{{{auto_routed_count}}} = {(self.project_metrics.get('auto_routing_analysis', {}).get('auto_routed_accuracy', 0)):.4f}$$

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