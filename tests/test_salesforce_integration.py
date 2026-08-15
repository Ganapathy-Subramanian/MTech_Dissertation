"""
Real tests for integration/salesforce.py.

SALESFORCE_ENABLED is never set to true here, so SalesforceIntegration
stays in mock mode: no real network/API calls, only local JSON files.
Each instance is pointed at a tmp_path json pair so the real
integration/mock_tickets.json and mock_contacts.json are never touched.
"""
import os
import pytest
from integration.salesforce import SalesforceIntegration


@pytest.fixture
def sf(tmp_path, monkeypatch):
    monkeypatch.delenv("SALESFORCE_ENABLED", raising=False)
    instance = SalesforceIntegration.__new__(SalesforceIntegration)
    instance.enabled = False
    instance.instance_url = "https://example.my.salesforce.com"
    instance.client_id = None
    instance.client_secret = None
    instance.username = None
    instance.password = None
    instance.api_version = "v57.0"
    instance.login_url = "https://login.salesforce.com"
    instance.security_token = ""
    instance.access_token = None
    instance.token_expiry = None
    instance.last_error = None
    instance.base_dir = str(tmp_path)
    instance.mock_tickets_file = os.path.join(str(tmp_path), "mock_tickets.json")
    instance.mock_contacts_file = os.path.join(str(tmp_path), "mock_contacts.json")
    instance.mock_tickets = {}
    instance.mock_contacts = {}
    return instance


class TestSalesforceMockMode:
    def test_authenticate_disabled_returns_false_without_error(self, sf):
        # In mock mode authenticate() should not attempt a real network call.
        assert sf.enabled is False

    def test_format_status_normalizes_text(self, sf):
        assert sf._format_status("in_progress") == "In Progress"
        assert sf._format_status(None) == "Unknown"
        assert sf._format_status("open") == "Open"

    def test_normalize_status_tag(self, sf):
        assert sf._normalize_status_tag("In Progress") == "in-progress"
        assert sf._normalize_status_tag(None) == ""

    def test_create_ticket_persists_locally(self, sf):
        ticket_id = sf.create_ticket({
            "subject": "Cannot login",
            "text": "I cannot access my account",
            "customer_id": "CUST_001",
            "owner_id": "AGENT_001",
            "category": "Technical Support",
        })
        assert ticket_id is not None
        assert ticket_id in sf.mock_tickets
        assert os.path.exists(sf.mock_tickets_file)

    def test_create_ticket_strips_customer_question_prefix(self, sf):
        ticket_id = sf.create_ticket({
            "subject": "Billing issue",
            "text": "CUSTOMER QUESTION: Why was I charged twice?",
            "customer_id": "CUST_002",
        })
        ticket = sf.mock_tickets[ticket_id]
        assert "CUSTOMER QUESTION: Why was I charged twice?" in ticket["messages"][0]["text"]

    def test_get_ticket_details_returns_created_ticket(self, sf):
        ticket_id = sf.create_ticket({"subject": "Test", "text": "hello", "customer_id": "CUST_001"})
        details = sf.get_ticket_details(ticket_id)
        assert details is not None
        assert details["id"] == ticket_id

    def test_get_ticket_details_missing_returns_none(self, sf):
        assert sf.get_ticket_details("TICKET-DOES-NOT-EXIST") is None

    def test_update_ticket_status(self, sf):
        ticket_id = sf.create_ticket({"subject": "Test", "text": "hello", "customer_id": "CUST_001"})
        ok = sf.update_ticket_status(ticket_id, "Resolved")
        assert ok is True
        assert sf.mock_tickets[ticket_id]["status"] == "Resolved"

    def test_add_ticket_message(self, sf):
        ticket_id = sf.create_ticket({"subject": "Test", "text": "hello", "customer_id": "CUST_001"})
        ok = sf.add_ticket_message(ticket_id, "We are looking into this", author_role="agent")
        assert ok is True
        assert any(m["text"] == "We are looking into this" for m in sf.mock_tickets[ticket_id]["messages"])

    def test_delete_ticket_removes_locally(self, sf):
        ticket_id = sf.create_ticket({"subject": "Test", "text": "hello", "customer_id": "CUST_001"})
        ok, owner_id = sf.delete_ticket(ticket_id)
        assert ok is True
        assert ticket_id not in sf.mock_tickets

    def test_delete_nonexistent_ticket_still_returns_true(self, sf):
        ok, owner_id = sf.delete_ticket("TICKET-NEVER-EXISTED")
        assert ok is True

    def test_search_tickets_finds_by_subject(self, sf):
        sf.create_ticket({"subject": "Password reset help", "text": "I forgot my password", "customer_id": "CUST_001"})
        sf.create_ticket({"subject": "Billing question", "text": "Why the extra charge", "customer_id": "CUST_002"})
        results = sf.search_tickets("password")
        assert len(results) == 1
        assert "Password" in results[0]["subject"]

    def test_search_tickets_no_match_returns_empty(self, sf):
        sf.create_ticket({"subject": "Test", "text": "hello", "customer_id": "CUST_001"})
        assert sf.search_tickets("nonexistent-query-xyz") == []

    def test_assign_ticket_sets_owner(self, sf):
        ticket_id = sf.create_ticket({"subject": "Test", "text": "hello", "customer_id": "CUST_001"})
        ok = sf.assign_ticket(ticket_id, "AGENT_007")
        assert ok is True
        assert sf.mock_tickets[ticket_id]["owner_id"] == "AGENT_007"

    def test_create_contact_mock_mode(self, sf):
        contact_id = sf.create_contact({
            "customer_id": "CUST_001",
            "first_name": "Rajesh",
            "last_name": "Kumar",
            "email": "rajesh.customer@email.com",
        })
        assert contact_id == "CUST_001"
        assert contact_id in sf.mock_contacts

    def test_get_contact_history_with_tickets(self, sf):
        sf.create_contact({"customer_id": "CUST_001", "email": "rajesh@example.com"})
        sf.create_ticket({"subject": "T1", "text": "hi", "customer_id": "CUST_001"})
        sf.create_ticket({"subject": "T2", "text": "hi again", "customer_id": "CUST_001"})
        history = sf.get_contact_history("CUST_001")
        assert history["total_tickets"] == 2
        assert history["contact"]["id"] == "CUST_001"

    def test_get_contact_history_unknown_contact(self, sf):
        history = sf.get_contact_history("CUST_UNKNOWN")
        assert history["total_tickets"] == 0
        assert history["contact"] == {}

    def test_get_dashboard_metrics_empty(self, sf):
        metrics = sf.get_dashboard_metrics()
        assert metrics["total_tickets"] == 0
        assert metrics["resolution_rate"] == 0

    def test_get_dashboard_metrics_with_tickets(self, sf):
        t1 = sf.create_ticket({"subject": "A", "text": "hi", "customer_id": "CUST_001", "priority": "High"})
        sf.update_ticket_status(t1, "Resolved")
        sf.create_ticket({"subject": "B", "text": "hi2", "customer_id": "CUST_002", "priority": "Low"})

        metrics = sf.get_dashboard_metrics()
        assert metrics["total_tickets"] == 2
        assert metrics["resolved_tickets"] == 1
        assert metrics["resolution_rate"] == 0.5
        assert metrics["source"] == "local"
