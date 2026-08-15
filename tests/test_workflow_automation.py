"""
Real tests for workflow/automation.py.

WorkflowEngine reads/writes workflow/workflows.json and
workflow/workflow_logs.json, and some actions call the shared
`salesforce` singleton (which itself writes integration/mock_*.json).
All of these are backed up and restored so real project data is
never permanently changed, and salesforce.enabled is forced False so
no live network call is attempted.
"""
import os
import json
import pytest
from workflow.automation import WorkflowEngine
from integration.salesforce import salesforce


@pytest.fixture(autouse=True)
def mock_salesforce(monkeypatch):
    monkeypatch.setattr(salesforce, "enabled", False)


@pytest.fixture
def preserve_workflow_files():
    engine_probe = WorkflowEngine()
    workflows_file = engine_probe.workflows_file
    logs_file = os.path.join(engine_probe.base_dir, "workflow_logs.json")

    with open(workflows_file, "r", encoding="utf-8") as f:
        orig_workflows = f.read()
    logs_existed = os.path.exists(logs_file)
    orig_logs = None
    if logs_existed:
        with open(logs_file, "r", encoding="utf-8") as f:
            orig_logs = f.read()

    try:
        yield
    finally:
        with open(workflows_file, "w", encoding="utf-8") as f:
            f.write(orig_workflows)
        if logs_existed:
            with open(logs_file, "w", encoding="utf-8") as f:
                f.write(orig_logs)
        elif os.path.exists(logs_file):
            os.remove(logs_file)


@pytest.fixture
def preserve_salesforce_mock_files():
    tickets_file = salesforce.mock_tickets_file
    contacts_file = salesforce.mock_contacts_file
    with open(tickets_file, "r", encoding="utf-8") as f:
        orig_tickets = f.read()
    with open(contacts_file, "r", encoding="utf-8") as f:
        orig_contacts = f.read()
    try:
        yield
    finally:
        with open(tickets_file, "w", encoding="utf-8") as f:
            f.write(orig_tickets)
        with open(contacts_file, "w", encoding="utf-8") as f:
            f.write(orig_contacts)


@pytest.fixture
def engine():
    return WorkflowEngine()


class TestWorkflowActions:
    def test_get_actions_for_known_category(self, engine):
        actions = engine.get_actions_for_category("Billing & Payments")
        assert len(actions) >= 1
        action_names = [a["action"] for a in actions]
        assert "send_payment_link" in action_names
        assert actions[0]["status"] == "pending"

    def test_get_actions_for_unknown_category_returns_empty(self, engine):
        assert engine.get_actions_for_category("Nonexistent Category") == []

    def test_get_action_description_known(self, engine):
        desc = engine._get_action_description("send_payment_link")
        assert desc == "Send secure payment link to customer"

    def test_get_action_description_unknown_returns_action_id(self, engine):
        assert engine._get_action_description("some_unknown_action") == "some_unknown_action"

    def test_escalate_to_human_appends_escalation_actions(self, engine):
        actions = engine.escalate_to_human("Technical Support", "High")
        action_names = [a["action"] for a in actions]
        assert "assign_to_agent" in action_names
        assert "create_crm_ticket" in action_names


class TestExecuteWorkflowAction:
    def test_execute_send_payment_link(self, engine, preserve_workflow_files):
        result = engine.execute_workflow_action("send_payment_link", "TICKET-1", {})
        assert result["status"] == "completed"
        assert "payment_link" in result

    def test_execute_create_support_ticket(self, engine, preserve_workflow_files):
        result = engine.execute_workflow_action("create_support_ticket", "TICKET-1", {})
        assert result["status"] == "completed"
        assert result["support_ticket_id"].startswith("ENG_")

    def test_execute_send_password_reset(self, engine, preserve_workflow_files):
        result = engine.execute_workflow_action("send_password_reset", "TICKET-1", {})
        assert result["status"] == "completed"
        assert "reset_link" in result

    def test_execute_create_crm_ticket(self, engine, preserve_workflow_files, preserve_salesforce_mock_files):
        result = engine.execute_workflow_action(
            "create_crm_ticket", "TICKET-1",
            {"text": "Customer needs help", "customer_id": "CUST_001"},
        )
        assert result["status"] == "completed"
        assert "salesforce_ticket_id" in result

    def test_execute_assign_to_agent(self, engine, preserve_workflow_files, preserve_salesforce_mock_files):
        # Create a ticket first so assign_ticket has something to update.
        ticket_id = salesforce.create_ticket({"subject": "Test", "text": "hi", "customer_id": "CUST_001"})
        result = engine.execute_workflow_action(
            "assign_to_agent", ticket_id, {"priority": "Critical"}
        )
        assert result["status"] == "completed"
        assert result["assigned_agent"] == "agent_supervisor_1"

    def test_execute_unknown_action_generic_success(self, engine, preserve_workflow_files, preserve_salesforce_mock_files):
        result = engine.execute_workflow_action("totally_unknown_action", "TICKET-1", {})
        assert result["status"] == "completed"

    def test_get_available_agent_by_priority(self, engine):
        assert engine._get_available_agent("Critical") == "agent_supervisor_1"
        assert engine._get_available_agent("Low") == "agent_junior_1"
        assert engine._get_available_agent("Unknown Priority") == "agent_general_1"


class TestWorkflowStatus:
    def test_get_status_returns_metrics(self, engine, preserve_workflow_files):
        engine.execute_workflow_action("send_payment_link", "TICKET-1", {})
        status = engine.get_status()
        assert "total_actions_executed" in status
        assert status["total_actions_executed"] >= 1
        assert 0 <= status["success_rate"] <= 1

    def test_get_active_workflows_returns_int(self, engine):
        assert isinstance(engine._get_active_workflows(), int)


class TestCustomWorkflow:
    def test_create_custom_workflow_persists(self, engine, preserve_workflow_files):
        result = engine.create_custom_workflow("Custom Category", {
            "auto_actions": ["custom_action"],
            "escalation_time": 100,
            "priority": "Low",
            "department": "Custom",
        })
        assert "updated successfully" in result["message"]

        actions = engine.get_actions_for_category("Custom Category")
        assert actions[0]["action"] == "custom_action"
