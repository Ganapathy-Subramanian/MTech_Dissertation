"""
Real tests for models/triage.py (legacy simple TriageModel) and the
EnsembleTriageModel / EnhancedTriageModel classes used in production.
"""
import pytest
from models.triage import TriageModel
from models.enhanced_triage import EnsembleTriageModel, EnhancedTriageModel


class TestLegacyTriageModel:
    def setup_method(self):
        self.model = TriageModel()

    def test_predict_returns_known_label(self):
        label, confidence = self.model.predict("I forgot my password")
        assert label in self.model.labels
        assert 0 <= confidence <= 1

    def test_predict_billing_text(self):
        label, confidence = self.model.predict("I need to pay my bill")
        assert isinstance(label, str)
        assert isinstance(confidence, float)


class TestEnsembleTriageModel:
    def setup_method(self):
        self.model = EnsembleTriageModel()

    def test_mode_is_set(self):
        assert self.model._mode in ("ensemble", "bert", "tfidf", "none")

    def test_labels_property_returns_list(self):
        assert isinstance(self.model.labels, list)
        assert len(self.model.labels) > 0

    def test_predict_enhanced_returns_valid_shape(self):
        label, confidence, features = self.model.predict_enhanced(
            "My payment failed twice this month"
        )
        assert isinstance(label, str)
        assert 0 <= confidence <= 1
        assert isinstance(features, dict)

    def test_analyze_sentiment(self):
        sentiment = self.model.analyze_sentiment("I am extremely happy with the service")
        assert "polarity" in sentiment
        assert sentiment["polarity"] > 0

    def test_determine_priority_critical_for_urgent(self):
        sentiment = {"compound": -0.9}
        entities = {}
        priority = self.model.determine_priority(
            "URGENT emergency, need help immediately", sentiment, entities
        )
        assert priority in ["Low", "Medium", "High", "Critical"]

    def test_predict_alias_matches_predict_enhanced_label(self):
        label, confidence = self.model.predict("I can't log in")
        assert isinstance(label, str)
        assert 0 <= confidence <= 1


class TestEnhancedTriageModelDirect:
    def setup_method(self):
        self.model = EnhancedTriageModel()

    def test_check_urgent_words_true(self):
        assert self.model._check_urgent_words("This is an EMERGENCY, urgent help needed") is True

    def test_check_urgent_words_false(self):
        assert self.model._check_urgent_words("Just a general question about pricing") is False

    def test_extract_entities_returns_dict(self):
        entities = self.model._extract_entities("My account number is 12345 and email is a@b.com")
        assert isinstance(entities, dict)
