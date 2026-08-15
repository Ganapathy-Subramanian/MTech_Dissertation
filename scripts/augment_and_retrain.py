import json, os
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BITEXT = os.path.join(BASE, 'models', 'bitext_insurance_mapped.json')

extras = [
    # Claims vs Emergency paraphrases
    {"text":"My property was damaged in a storm, how do I claim?","category":"Claims","intent":"file_claim","split":"train"},
    {"text":"I need to report storm damage and file a claim","category":"Claims","intent":"file_claim","split":"train"},
    {"text":"My car was damaged in the flood, how do I file a claim?","category":"Claims","intent":"file_claim","split":"train"},
    # Technical vs Account
    {"text":"App not loading on my phone — app crashes on startup","category":"Technical Support","intent":"technical_error","split":"train"},
    {"text":"App crashes when I login, not a password issue","category":"Technical Support","intent":"technical_error","split":"train"},
    {"text":"The mobile app won't open, getting an error","category":"Technical Support","intent":"technical_error","split":"train"},
    # Complaints vs Technical
    {"text":"The chatbot keeps giving me wrong answers, it's unusable","category":"Complaints & Feedback","intent":"file_complaint","split":"train"},
    {"text":"I want to complain about the chatbot's responses","category":"Complaints & Feedback","intent":"file_complaint","split":"train"},
]

if not os.path.exists(BITEXT):
    print('Bitext file not found:', BITEXT)
    raise SystemExit(1)

with open(BITEXT, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Append and avoid duplicates by text
existing_texts = {entry['text'] for entry in data}
added = 0
for e in extras:
    if e['text'] not in existing_texts:
        data.append(e)
        added += 1

with open(BITEXT, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print(f"Appended {added} examples to {BITEXT}")

# Trigger TF-IDF retrain via auto_retrain wrapper
print('Starting TF-IDF retrain (this may take a few seconds)...')
os.system('python -m models.auto_retrain --retrain')
print('Retrain command finished at', datetime.utcnow().isoformat())
