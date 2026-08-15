import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SEP = "=" * 60

# ── STEP 1: Inspect bert_insurance_triage folder ──────────────────
print(f"\n{SEP}")
print("STEP 1: bert_insurance_triage folder contents")
print(SEP)

BERT_DIR = os.path.join("models", "bert_insurance_triage")

if not os.path.exists(BERT_DIR):
    print(f"  ❌ Folder NOT FOUND: {BERT_DIR}")
    print("     Run: python -m models.bert_triage --train")
    sys.exit(1)

files = []
for root, dirs, fnames in os.walk(BERT_DIR):
    # skip checkpoint sub-dirs from main listing
    rel = os.path.relpath(root, BERT_DIR)
    for f in fnames:
        full = os.path.join(root, f)
        size = os.path.getsize(full)
        files.append((os.path.join(rel, f) if rel != "." else f, size))

if not files:
    print(f"  ❌ Folder EXISTS but is EMPTY → model was never saved")
    sys.exit(1)

for fname, size in sorted(files):
    tag = "✅" if size > 0 else "⚠️ (empty)"
    print(f"  {tag}  {fname}  ({size/1024:.1f} KB)")

# Key files check
has_config   = any("config.json" in f for f, _ in files)
has_model    = any("model.safetensors" in f or "pytorch_model.bin" in f for f, _ in files)
has_tokenizer = any("tokenizer" in f for f, _ in files) or any("vocab.txt" in f for f, _ in files)
has_ckpt     = any("checkpoint-" in f for f, _ in files)

print()
print(f"  config.json    : {'✅ found' if has_config else '❌ missing'}")
print(f"  model weights  : {'✅ found' if has_model else '❌ missing (no safetensors/pytorch_model.bin)'}")
print(f"  tokenizer files: {'✅ found' if has_tokenizer else '❌ missing'}")
print(f"  checkpoints    : {'✅ found' if has_ckpt else 'ℹ️  none (ok if config.json exists)'}")

if not has_config and not has_ckpt:
    print("\n  ❌ CRITICAL: No config.json and no checkpoints found.")
    print("     Training may have not completed. Re-run:")
    print("     python -m models.bert_triage --train")
    sys.exit(1)

# ── STEP 2: Try loading BERTTriageModel ───────────────────────────
print(f"\n{SEP}")
print("STEP 2: Load BERTTriageModel")
print(SEP)

try:
    from models.bert_triage import BERTTriageModel
    bert = BERTTriageModel()

    if hasattr(bert, '_fallback'):
        print("  ⚠️  BERT load FAILED — running on Phase-1 TF-IDF fallback")
        print("     Reason: transformers/torch not installed OR model files corrupted")
        BERT_LOADED = False
    elif bert.model is None:
        print("  ❌ model attribute is None — load failed silently")
        BERT_LOADED = False
    else:
        print(f"  ✅ BERTTriageModel loaded successfully")
        print(f"     Device : {getattr(bert, '_device', 'unknown')}")
        print(f"     Labels : {bert.labels}")
        BERT_LOADED = True
except Exception as e:
    print(f"  ❌ Exception during BERTTriageModel init: {e}")
    BERT_LOADED = False

# ── STEP 3: Load EnsembleTriageModel and check mode ───────────────
print(f"\n{SEP}")
print("STEP 3: Load EnsembleTriageModel (what main_enhanced.py uses)")
print(SEP)

try:
    from models.enhanced_triage import EnsembleTriageModel
    ens = EnsembleTriageModel()
    mode = ens._mode
    print(f"  Mode : {mode}")
    if mode == "ensemble":
        print("  ✅ Running ENSEMBLE (BERT 70% + TF-IDF 30%) — best accuracy")
    elif mode == "bert":
        print("  ✅ Running BERT-only")
    elif mode == "tfidf":
        print("  ⚠️  Running TF-IDF only — BERT not loaded")
    else:
        print("  ❌ No models loaded")
except Exception as e:
    print(f"  ❌ EnsembleTriageModel load error: {e}")
    sys.exit(1)

# ── STEP 4: Routing test ──────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 4: Routing Test (simulates /process-ticket threshold logic)")
print(SEP)

import os
CONFIDENCE_THRESHOLD = float(os.getenv('TRIAGE_CONFIDENCE_THRESHOLD', 0.70))
SENTIMENT_THRESHOLD  = -0.3

# Build ~50 diverse queries by templating common phrasings for each label
templates = {
    "Policy & Coverage": [
        "What does my comprehensive coverage include?",
        "Does my policy cover rental car?",
        "How do I check my current coverage?",
        "Is flood damage covered under my policy?",
        "What is included in my comprehensive coverage?"
    ],
    "Claims": [
        "I need to file a claim for my car accident",
        "How do I submit an insurance claim?",
        "My claim has been pending for weeks",
        "I want to track the status of my claim",
        "My property was damaged in a storm, need to claim"
    ],
    "Billing & Payments": [
        "My payment failed how do I retry?",
        "I was charged twice this month",
        "Where is my invoice or billing statement?",
        "How do I update my payment method?",
        "When is my next payment due?"
    ],
    "Account & Password": [
        "I forgot my password",
        "I cannot log in to my account",
        "How do I reset my password?",
        "My account is locked, please help",
        "Two-factor authentication not working"
    ],
    "Emergency Services": [
        "My car broke down on the highway need help",
        "There is an emergency flood at my property",
        "I need immediate roadside assistance",
        "My house is on fire please advise",
        "I need urgent medical claim assistance"
    ],
    "Refund & Returns": [
        "I want a refund for my cancelled policy",
        "How do I cancel my policy and get refund?",
        "I overpaid my premium I want a refund",
        "Policy cancelled — request refund",
        "Refund status for my overpayment?"
    ],
    "Technical Support": [
        "Your website keeps crashing",
        "The claims portal crashed when uploading files",
        "App not loading on my phone",
        "Error 500 when I try to submit form",
        "The chatbot keeps giving me wrong answers"
    ],
    "Policy Changes": [
        "I want to cancel my policy",
        "How do I renew my policy?",
        "I want to change my coverage",
        "Can I update my policy details?",
        "How to modify my policy?"
    ],
    "Complaints & Feedback": [
        "I want to file a formal complaint",
        "Your agent was extremely rude to me",
        "I want to escalate this issue",
        "Very poor communication from your team",
        "I want to submit feedback about my experience"
    ],
    "General Inquiry": [
        "Tell me more about your insurance products",
        "What are your business hours?",
        "Can I get an insurance quote?",
        "How do I contact customer service?",
        "Do you offer umbrella insurance?"
    ]
}

# Flatten templates to ~50 queries (10 categories × 5 each)
test_cases = []
for cat, qs in templates.items():
    for q in qs:
        test_cases.append((q, cat))

from models.auto_retrain import SelfLearningWrapper
wrapper = SelfLearningWrapper(model=ens)

all_pass = True
for query, expected in test_cases:
    label, conf, feats = wrapper.predict_and_log(query)
    sentiment = ens.analyze_sentiment(query)
    sent_score = sentiment.get("compound", 0)

    # Simulate main_enhanced.py routing decision
    if conf > CONFIDENCE_THRESHOLD and sent_score > SENTIMENT_THRESHOLD:
        route = "✅ Local Triage"
    else:
        route = "❌ ESCALATED (Complex/Contextual)"

    category_ok = "✅" if label == expected else f"❌ (expected {expected})"
    conf_ok     = "✅" if conf >= CONFIDENCE_THRESHOLD else f"⚠️  conf={conf:.2f} < {CONFIDENCE_THRESHOLD}"

    print(f"\n  Query    : \"{query}\"")
    print(f"  Category : {label} {category_ok}")
    print(f"  Confidence: {conf:.4f}  {conf_ok}")
    print(f"  Routing  : {route}")

    if label != expected or conf <= CONFIDENCE_THRESHOLD:
        all_pass = False

# ── STEP 5: BERT-specific inference test (if loaded) ─────────────
if BERT_LOADED:
    print(f"\n{SEP}")
    print("STEP 5: Direct BERT inference check")
    print(SEP)
    q = "What does my comprehensive coverage include?"
    b_label, b_conf, b_feats = bert.predict_enhanced(q)
    all_probs = b_feats.get("all_probabilities", {})
    print(f"  Query     : \"{q}\"")
    print(f"  BERT label: {b_label}  (conf={b_conf:.4f})")
    print("  Top-5 probs:")
    for k, v in sorted(all_probs.items(), key=lambda x: -x[1])[:5]:
        print(f"    {k:<30} {v:.4f}")

# ── Summary ───────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SUMMARY")
print(SEP)
print(f"  BERT loaded       : {'✅ Yes' if BERT_LOADED else '❌ No (TF-IDF fallback)'}")
print(f"  Ensemble mode     : {mode}")
print(f"  All routing tests : {'✅ PASS' if all_pass else '❌ SOME FAILED — check above'}")
print()