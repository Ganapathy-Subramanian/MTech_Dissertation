"""Quick test of EnsembleTriageModel + SelfLearningWrapper"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== Loading EnsembleTriageModel ===")
from models.enhanced_triage import EnsembleTriageModel
model = EnsembleTriageModel()
print(f"Mode: {model._mode}")
print()

print("=== Test Predictions ===")
queries = [
    "I need to file a car accident claim",
    "My insurance payment failed this month",
    "I forgot my password and cannot login",
    "I want to cancel my policy",
    "There is an emergency flood at my property",
    "What does my comprehensive coverage include?",
]

for q in queries:
    label, conf, feats = model.predict_enhanced(q)
    bert_top   = feats.get("bert_top", "N/A")
    tfidf_top  = feats.get("tfidf_top", "N/A")
    mode       = feats.get("model_mode", model._mode)
    print(f"Query : {q[:55]}")
    print(f"Result: [{label}] conf={conf:.3f}  mode={mode}")
    if mode == "ensemble":
        print(f"  BERT={bert_top}  TF-IDF={tfidf_top}")
    print()

print("=== SelfLearningWrapper ===")
from models.auto_retrain import SelfLearningWrapper
wrapper = SelfLearningWrapper(model=model)
cat, conf2, feats2 = wrapper.predict_and_log("I need to file a claim")
print(f"SelfLearning predict: [{cat}] conf={conf2:.3f}")
print(f"Pending corrections: {wrapper._corrections_since_last_retrain}")
metrics = wrapper.get_metrics_report()
print(f"Model type: {metrics['model_type']}")
print(f"Total corrections: {metrics['total_corrections']}")
print(f"Total retrains: {metrics['total_retrains']}")
print()
print("=== ALL TESTS PASSED ===")
