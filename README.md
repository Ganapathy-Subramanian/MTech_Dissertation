# M.Tech Dissertation — AI-Based Insurance Ticket Routing System

## 1. Project Overview

This project implements an AI-based insurance ticket routing system that automatically assigns every customer ticket to one of **10 predefined routing categories**.

The project evaluates an enhanced machine-learning routing model and a BERT-based model on insurance-domain ticket data.

### 10 Routing Categories

1. Account & Password
2. Billing & Payments
3. Claims
4. Complaints & Feedback
5. Emergency Services
6. General Inquiry
7. Policy & Coverage
8. Policy Changes
9. Refund & Returns
10. Technical Support

---

## 2. Repository Structure

```text
MTech_Dissertation/
├── main.py
├── main_demo.py
├── main_enhanced.py
├── evaluate_metrics.py
├── run_comprehensive_evaluation.py
├── comprehensive_evaluation.py
├── business_evaluation_report.py
├── requirements.txt
├── README.md
├── models/
│   ├── triage.py
│   ├── enhanced_triage.py
│   ├── bert_triage.py
│   ├── enhanced_triage_model.pkl
│   └── bert_insurance_triage/
├── rag/
├── analytics/
├── workflow/
├── security/
├── agents/
├── integration/
└── tests/
```

---

# 3. Google Colab — Complete Setup

The repository is designed to be executed from a fresh Google Colab environment.

### Step 1 — Open Google Colab

Create a new notebook in Google Colab.

No files need to be uploaded manually from the local Windows computer.

### Step 2 — Clone the GitHub Repository

Run:

```python
!git clone https://github.com/Ganapathy-Subramanian/MTech_Dissertation.git
%cd /content/MTech_Dissertation
```

Verify the project files:

```python
!ls
```

### Step 3 — Set Up Git LFS

The trained BERT model is stored using Git Large File Storage because of its large size.

Run:

```python
!git lfs install
!git lfs pull
```

Verify the BERT model:

```python
!ls -lh models/bert_insurance_triage/model.safetensors
```

### Step 4 — Install Dependencies

Run:

```python
!pip install -r requirements.txt
```

Verify the main ML packages:

```python
import torch
import transformers
import sklearn
import pandas
import numpy

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Pandas:", pandas.__version__)
print("NumPy:", numpy.__version__)
```

---

# 4. Verify the Project Files

Run:

```python
import os

print("TF-IDF model:",
      os.path.exists("models/triage_model.pkl"))

print("Enhanced model:",
      os.path.exists("models/enhanced_triage_model.pkl"))

print("BERT model:",
      os.path.exists("models/bert_insurance_triage/model.safetensors"))
```

Expected:

```text
TF-IDF model: True
Enhanced model: True
BERT model: True
```

---

# 5. Verify the Enhanced 10-Class Model

```python
from models.enhanced_triage import EnhancedTriageModel

model = EnhancedTriageModel()

query = "I want to know the status of my insurance claim."

result = model.predict_enhanced(query)

print(result)
```

The enhanced model returns the predicted category, confidence score, extracted query features and probability distribution across the 10 categories.

---

# 6. Verify the BERT Model

```python
from models.bert_triage import BERTTriageModel

bert_model = BERTTriageModel()

query = "My insurance claim was rejected and I want to know why."

result = bert_model.predict_enhanced(query)

print(result)
```

---

# 7. Run the Primary Dissertation Evaluation

Run:

```python
!python evaluate_metrics.py
```

The evaluation includes:

- Standard seed-42 train/test evaluation
- Accuracy
- Weighted F1-score
- Macro F1-score
- Precision and Recall
- Confusion matrix
- Duplicate-data analysis
- Business-impact evaluation
- Confidence analysis
- Informal-query robustness testing
- ClaimSense reference comparison
- Full BERT evaluation on the held-out test set

---

# 8. Run the Comprehensive Evaluation

Run:

```python
!python run_comprehensive_evaluation.py
```

---

# 9. Run the Application / Demos

Standard demo:

```python
!python main_demo.py
```

Enhanced demo:

```python
!python main_enhanced.py
```

Main application:

```python
!python main.py
```

Some application or external-integration components may require additional environment-specific configuration or credentials.

---

# 10. Current Evaluation Results

## Standard Seed-42 Holdout

| Metric | Result |
|---|---:|
| Test samples | **4,215** |
| Accuracy | **99.0985%** |
| Weighted F1 | **0.9910** |
| Macro F1 | **0.9833** |

Every ticket is assigned to exactly one of the 10 routing categories.

## Duplicate-Controlled Evaluation

Exact duplicate overlap was identified between the training and test sets. A separate duplicate-controlled sensitivity evaluation was therefore performed.

| Metric | Result |
|---|---:|
| Controlled test samples | **3,782** |
| Accuracy | **98.2020%** |
| Weighted F1 | **0.9830** |
| Macro F1 | **0.8850** |

The standard result is retained for reproducibility, while the duplicate-controlled result provides a more conservative assessment.

## BERT Evaluation

BERT was evaluated on the complete **4,215-ticket held-out test set**.

| Metric | Result |
|---|---:|
| Accuracy | **99.0985%** |
| Weighted F1 | **0.9909** |
| Macro F1 | **0.9709** |

## Informal Query Robustness

A separate set of 50 manually constructed informal/typo queries was evaluated.

**Accuracy: 94% (47/50 correct)**

This is reported separately and is not treated as production accuracy.

---

# 11. Business-Impact Evaluation

The project includes an experimental business-impact evaluation for routing decisions.

| Measure | Result |
|---|---:|
| Correct-routing reward | +0.1 units/ticket |
| Total reward | 417.70 units |
| Total penalty | 70.50 units |
| Net utility | 347.20 normalized units |
| Utility per ticket | 0.0824 units/ticket |

These weights are **experimental research assumptions**. They do not represent actual insurance-industry financial costs or guaranteed SLA penalties.

---

# 12. Duplicate Data and Evaluation Integrity

The combined dataset contains exact duplicate records.

The evaluation therefore reports:

1. Standard seed-42 holdout performance
2. Duplicate-controlled sensitivity performance

The standard result is retained for reproducibility. The duplicate-controlled result shows how performance changes when exact duplicate texts are removed before splitting.

---

# 13. ClaimSense Reference

The project uses `pramodmisra/claimsense-ai-v1` as a published reference.

The ClaimSense result is treated as a **published/self-reported reference**, not as an apples-to-apples benchmark, because its published evaluation was not independently reproduced using the exact 10-class train/test protocol used in this dissertation.

Therefore, this project does **not** claim a direct percentage-point improvement or SOTA superiority over ClaimSense.

---

# 14. Evaluation Outputs

Evaluation visualizations are generated under:

```text
eval_output/
```

Expected outputs include:

```text
confusion_matrix.png
confidence_distribution.png
f1_scores.png
benchmark_comparison.png
```

---

# 15. Reproducibility

The primary evaluation uses a **seed-42 holdout split**.

The repository contains the source code, trained model artifacts, evaluation scripts, BERT model files and requirements required to reproduce the reported experiments.

The BERT model is stored using **Git LFS** because of its large file size.

The project can be cloned into a fresh Colab environment without depending on the original Windows computer.

---

# 16. Security

Do not commit API keys, passwords, Salesforce credentials or other secrets to GitHub.

External integrations requiring credentials should be configured separately through environment variables.

A `.env` file containing real credentials must never be uploaded to the repository.

---

# 17. Quick Start

For the primary dissertation evaluation, run the following commands in order:

```python
!git clone https://github.com/YOUR_USERNAME/MTech_Dissertation.git
%cd MTech_Dissertation
!git lfs install
!git lfs pull
!pip install -r requirements.txt
!python evaluate_metrics.py
```

For the comprehensive evaluation:

```python
!python run_comprehensive_evaluation.py
```

---

# 18. Main Commands

Primary evaluation:

```bash
python evaluate_metrics.py
```

Comprehensive evaluation:

```bash
python run_comprehensive_evaluation.py
```

Standard demo:

```bash
python main_demo.py
```

Enhanced demo:

```bash
python main_enhanced.py
```

Main application:

```bash
python main.py
```

---

# 19. Research Scope

The primary research evaluation focuses on **10-class insurance ticket routing**.

The project separately reports:

- Standard holdout performance
- Duplicate-controlled performance
- Full-test-set BERT performance
- Informal-query robustness
- Experimental business utility
- ClaimSense published-reference comparison

These results are kept separate to avoid combining different evaluation settings into a single accuracy claim.
