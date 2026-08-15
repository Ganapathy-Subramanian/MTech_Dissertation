from models.auto_retrain import SelfLearningWrapper
import json
w=SelfLearningWrapper()
# ⚠️ FIXED: Simulate correction with valid 10-class category instead of "Complex/Contextual"
w.add_correction('What does my comprehensive coverage include?','Policy & Coverage','Policy & Coverage')
print(json.dumps(w._read_json(w.correction_log_path,[])[-1],indent=2))
