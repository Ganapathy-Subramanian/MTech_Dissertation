import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os

# Base directory for models
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "triage_model.pkl")

class TriageModel:
    def __init__(self):
        self.pipeline = None
        self.labels = [
            "Billing",
            "Technical Support",
            "Password Reset",
            "General Inquiry",
            "Refund Request"
        ]
        
        # Load model if exists, else train a default one
        if os.path.exists(MODEL_PATH):
            self.pipeline = joblib.load(MODEL_PATH)
        else:
            self.train_default_model()

    def train_default_model(self):
        # Dummy data for initial training
        data = [
            ("I need to pay my bill", "Billing"),
            ("Where is my invoice?", "Billing"),
            ("Subscription payment failed", "Billing"),
            ("How do I update my credit card?", "Billing"),
            
            ("My account is locked", "Technical Support"),
            ("The website is not loading", "Technical Support"),
            ("I found a bug in the app", "Technical Support"),
            ("The software keeps crashing", "Technical Support"),
            
            ("Forgot my password", "Password Reset"),
            ("I can't log in to my account", "Password Reset"),
            ("How to reset password?", "Password Reset"),
            ("Change my login credentials", "Password Reset"),
            
            ("Tell me more about your services", "General Inquiry"),
            ("Who are you?", "General Inquiry"),
            ("I have a question about the product", "General Inquiry"),
            ("Contact sales team", "General Inquiry"),

            ("I want my money back", "Refund Request"),
            ("Cancel my order and refund", "Refund Request"),
            ("Wrong item sent, need refund", "Refund Request"),
            ("Requesting a credit", "Refund Request")
        ]
        
        texts, targets = zip(*data)
        
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression())
        ])
        
        self.pipeline.fit(texts, targets)
        joblib.dump(self.pipeline, MODEL_PATH)
        print(f"Model trained and saved to {MODEL_PATH}")

    def predict(self, text: str):
        if not self.pipeline:
            return "General Inquiry", 0.0
            
        probs = self.pipeline.predict_proba([text])[0]
        max_idx = probs.argmax()
        label = self.pipeline.classes_[max_idx]
        confidence = probs[max_idx]
        
        return label, confidence

if __name__ == "__main__":
    triage = TriageModel()
    test_text = "I can't access my account, password not working"
    label, conf = triage.predict(test_text)
    print(f"Prediction: {label} (Confidence: {conf:.2f})")
