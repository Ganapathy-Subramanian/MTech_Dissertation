"""
Real tests for models/auto_retrain.py's SelfLearningWrapper.

All file paths (correction_log, training_history, bitext, pending-count)
are pointed at tmp_path, and a lightweight fake model is injected, so
these tests never touch the real (16 MB) bitext_insurance_mapped.json,
never trigger a real sklearn retrain, and run fast.
"""
import os
import threading
import pytest
from models.auto_retrain import SelfLearningWrapper


class FakeModel:
    """Minimal stand-in for EnsembleTriageModel/EnhancedTriageModel."""
    labels = ["Billing & Payments", "Technical Support", "General Inquiry"]
    _mode = "tfidf"
    pipeline = None

    def predict_enhanced(self, text):
        if "bill" in text.lower():
            return "Billing & Payments", 0.9, {}
        return "General Inquiry", 0.3, {}

    def predict(self, text):
        label, conf, _ = self.predict_enhanced(text)
        return label, conf


@pytest.fixture
def wrapper(tmp_path):
    w = SelfLearningWrapper.__new__(SelfLearningWrapper)
    w.base_dir = str(tmp_path)
    w.correction_log_path = os.path.join(str(tmp_path), "correction_log.json")
    w.training_history_path = os.path.join(str(tmp_path), "training_history.json")
    w.bitext_path = os.path.join(str(tmp_path), "bitext.json")
    w._counter_path = os.path.join(str(tmp_path), "pending_count.json")
    w._model = FakeModel()
    w._model_type = "tfidf"
    w._lock = threading.Lock()
    w._scheduler = None
    w._corrections_since_last_retrain = 0
    w._write_json(w.correction_log_path, [])
    return w


class TestPredictAndLog:
    def test_high_confidence_not_logged(self, wrapper):
        category, confidence, _ = wrapper.predict_and_log("I need to pay my bill")
        assert category == "Billing & Payments"
        log = wrapper._read_json(wrapper.correction_log_path, [])
        assert log == []

    def test_low_confidence_gets_logged(self, wrapper):
        category, confidence, _ = wrapper.predict_and_log("something vague")
        log = wrapper._read_json(wrapper.correction_log_path, [])
        assert len(log) == 1
        assert log[0]["predicted_category"] == category


class TestAddCorrection:
    def test_add_correction_appends_to_log(self, wrapper):
        wrapper.add_correction("I was double charged", "Billing & Payments",
                                original_category="General Inquiry", confidence=0.4)
        log = wrapper._read_json(wrapper.correction_log_path, [])
        assert len(log) == 1
        assert log[0]["correct_category"] == "Billing & Payments"

    def test_add_correction_unknown_label_maps_to_general_inquiry(self, wrapper):
        wrapper.add_correction("some ticket text", "Totally Unknown Label")
        log = wrapper._read_json(wrapper.correction_log_path, [])
        assert log[0]["correct_category"] == "General Inquiry"

    def test_add_correction_merges_into_bitext(self, wrapper):
        wrapper.add_correction("ticket about billing", "Billing & Payments")
        bitext = wrapper._read_json(wrapper.bitext_path, [])
        assert len(bitext) == 1
        assert bitext[0]["intent"] == "human_correction"

    def test_add_correction_increments_counter(self, wrapper):
        wrapper.add_correction("first", "Billing & Payments")
        assert wrapper._corrections_since_last_retrain == 1
        wrapper.add_correction("second", "Billing & Payments")
        assert wrapper._corrections_since_last_retrain == 2


class TestCheckDrift:
    def test_check_drift_with_insufficient_data_does_nothing(self, wrapper):
        # Fewer than 5 corrections logged -> should return early without error.
        wrapper.add_correction("only one correction", "Billing & Payments")
        wrapper.check_drift()  # should not raise


class TestMetricsReport:
    def test_get_metrics_report_shape(self, wrapper):
        report = wrapper.get_metrics_report()
        assert report["model_type"] == "tfidf"
        assert report["total_retrains"] == 0
        assert "targets" in report
        assert "retrain_trigger_at" in report

    def test_metrics_report_reflects_pending_corrections(self, wrapper):
        wrapper.add_correction("a ticket", "Billing & Payments")
        report = wrapper.get_metrics_report()
        assert report["pending_corrections"] == 1
        assert report["total_corrections"] == 1


class TestJsonHelpers:
    def test_read_json_missing_file_returns_default(self, tmp_path):
        missing = os.path.join(str(tmp_path), "does_not_exist.json")
        assert SelfLearningWrapper._read_json(missing, {"default": True}) == {"default": True}

    def test_write_then_read_json_roundtrip(self, tmp_path):
        path = os.path.join(str(tmp_path), "roundtrip.json")
        SelfLearningWrapper._write_json(path, {"a": 1})
        assert SelfLearningWrapper._read_json(path, None) == {"a": 1}

    def test_read_json_corrupted_file_returns_default(self, tmp_path):
        path = os.path.join(str(tmp_path), "corrupt.json")
        with open(path, "w") as f:
            f.write("{not valid json")
        assert SelfLearningWrapper._read_json(path, []) == []


class TestScheduler:
    def test_stop_scheduler_without_start_is_safe(self, wrapper):
        wrapper.stop_scheduler()  # scheduler is None -> should not raise
