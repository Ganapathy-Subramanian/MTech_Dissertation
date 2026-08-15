import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    example_path = os.path.join(ROOT, "claimsense_predictions_example.csv")
    if os.path.exists(example_path):
        print(f"Example baseline file already exists: {example_path}")
        return

    df = pd.DataFrame(
        [
            {"query": "I want to accept a settlement offer", "prediction": "Claims"},
            {"query": "My password reset is not working", "prediction": "Account & Password"},
            {"query": "How do I check my policy coverage?", "prediction": "Policy & Coverage"},
            {"query": "I have a billing issue and was overcharged", "prediction": "Billing & Payments"},
            {"query": "I need help with a refund request", "prediction": "Refund & Returns"},
            {"query": "My house is on fire right now", "prediction": "Emergency Services"},
            {"query": "I need to contact a support agent", "prediction": "General Inquiry"},
        ]
    )
    df.to_csv(example_path, index=False)
    print(f"Created example baseline file: {example_path}")


if __name__ == "__main__":
    main()
