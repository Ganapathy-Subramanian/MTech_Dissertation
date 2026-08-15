import pytest
from models.enhanced_triage import EnhancedTriageModel
from rag.vector_db import AdaptiveMemory
from analytics.dashboard import AnalyticsDashboard
from workflow.automation import WorkflowEngine

class TestEnhancedTriageModel:
    def setup_method(self):
        self.model = EnhancedTriageModel()

    def test_predict_enhanced(self):
        text = "I can't access my account"
        label, confidence, entities = self.model.predict_enhanced(text)

        assert isinstance(label, str)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
        assert isinstance(entities, dict)

    def test_analyze_sentiment(self):
        text = "This is amazing!"
        sentiment = self.model.analyze_sentiment(text)

        assert 'polarity' in sentiment
        assert 'subjectivity' in sentiment
        assert 'compound' in sentiment
        assert sentiment['polarity'] > 0  # Should be positive

    def test_determine_priority(self):
        text = "URGENT: System is completely broken!"
        sentiment = {'compound': -0.8}
        entities = {'has_account_issues': True}

        priority = self.model.determine_priority(text, sentiment, entities)
        assert priority in ['Low', 'Medium', 'High', 'Critical']

class TestAdaptiveMemory:
    def setup_method(self):
        self.memory = AdaptiveMemory()

    def test_add_and_query_correction(self):
        text = "Test correction query"
        label = "Technical Support"

        self.memory.add_correction(text, label)
        result_label, distance = self.memory.query_memory(text)

        assert result_label == label
        assert distance is not None
        assert distance < 1.0  # Should be very similar

class TestAnalyticsDashboard:
    def setup_method(self):
        self.analytics = AnalyticsDashboard()

    def test_log_and_get_dashboard(self):
        # Test logging
        class MockTicket:
            def __init__(self):
                self.text = "Test ticket"
                self.customer_id = "test_customer"
                self.channel = "web"
                self.priority = "Medium"

        ticket = MockTicket()
        self.analytics.log_ticket(ticket)

        # Test dashboard data retrieval
        dashboard_data = self.analytics.get_dashboard_data()
        assert isinstance(dashboard_data, dict)
        assert 'summary' in dashboard_data

class TestWorkflowEngine:
    def setup_method(self):
        self.workflow = WorkflowEngine()

    def test_get_actions_for_category(self):
        actions = self.workflow.get_actions_for_category("Technical Support")
        assert isinstance(actions, list)

        if actions:
            assert 'action' in actions[0]
            assert 'status' in actions[0]
            assert 'description' in actions[0]

    def test_escalate_to_human(self):
        actions = self.workflow.escalate_to_human("Billing & Payments", "High")
        assert isinstance(actions, list)
        assert len(actions) > 0

if __name__ == "__main__":
    # Run basic functionality tests
    print("Running basic functionality tests...")

    # Test triage model
    model = EnhancedTriageModel()
    test_text = "I forgot my password"
    label, confidence, entities = model.predict_enhanced(test_text)
    print(f"Triage Test: {test_text} -> {label} ({confidence:.2f})")

    # Test sentiment
    sentiment = model.analyze_sentiment("This service is terrible!")
    print(f"Sentiment Test: polarity={sentiment['polarity']:.2f}")

    # Test memory
    memory = AdaptiveMemory()
    memory.add_correction("Test issue", "Technical Support")
    result, dist = memory.query_memory("Test issue")
    print(f"Memory Test: Found {result} with distance {dist}")

    # Test analytics
    analytics = AnalyticsDashboard()
    dashboard = analytics.get_dashboard_data()
    print(f"Analytics Test: Dashboard has {len(dashboard)} sections")

    # Test workflow
    workflow = WorkflowEngine()
    actions = workflow.get_actions_for_category("Technical Support")
    print(f"Workflow Test: {len(actions)} actions for Technical Support")

    print("All basic tests completed successfully!")