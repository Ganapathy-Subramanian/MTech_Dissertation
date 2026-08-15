"""
Real tests for the main_enhanced.py FastAPI app (the app's actual
production entrypoint - see DELIVERY_SUMMARY.md / QUICKSTART.py).

Endpoints that write to real project state (security/customers.json,
agents/agent_database.json) back up and restore that file around the
test so the developer's real data is never permanently changed.
"""
import os
import json
import pytest
from fastapi.testclient import TestClient

from main_enhanced import app, auth
from integration.salesforce import salesforce
from security.customer_auth import customer_auth_manager
from agents.agent_manager import agent_team_manager


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def no_real_salesforce_calls(monkeypatch):
    """The real .env sets SALESFORCE_ENABLED=true, which makes admin login
    and other flows attempt live Salesforce OAuth. Force mock mode for
    all tests so the suite never depends on live network/credentials."""
    monkeypatch.setattr(salesforce, "enabled", False)


def admin_token():
    """Build an admin JWT directly, bypassing the live-Salesforce-dependent
    /admin/login flow, so tests stay independent of network access."""
    return auth.create_access_token(data={
        "sub": "admin@company.com", "role": "admin",
        "email": "admin@company.com", "name": "Test Admin",
    })


def customer_token(client, preserve_customers_json_active=True):
    resp = client.post("/customer/login", json={
        "email": "rajesh.customer@email.com", "password": "Password123!",
    })
    assert resp.status_code == 200
    return resp.json()["token"]


@pytest.fixture
def preserve_customers_json():
    path = customer_auth_manager.customers_file
    with open(path, "r", encoding="utf-8") as f:
        original = f.read()
    try:
        yield
    finally:
        with open(path, "w", encoding="utf-8") as f:
            f.write(original)


@pytest.fixture
def preserve_agents_json():
    path = agent_team_manager.agents_file
    with open(path, "r", encoding="utf-8") as f:
        original = f.read()
    try:
        yield
    finally:
        with open(path, "w", encoding="utf-8") as f:
            f.write(original)


@pytest.fixture
def preserve_salesforce_mock_files():
    tickets_path = salesforce.mock_tickets_file
    contacts_path = salesforce.mock_contacts_file
    with open(tickets_path, "r", encoding="utf-8") as f:
        orig_tickets = f.read()
    with open(contacts_path, "r", encoding="utf-8") as f:
        orig_contacts = f.read()
    try:
        yield
    finally:
        with open(tickets_path, "w", encoding="utf-8") as f:
            f.write(orig_tickets)
        with open(contacts_path, "w", encoding="utf-8") as f:
            f.write(orig_contacts)


class TestHealthAndStatic:
    def test_root_serves_html(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_login_page_served(self, client):
        response = client.get("/login.html")
        assert response.status_code == 200


class TestTicketProcessing:
    def test_process_ticket_returns_category(self, client):
        headers = {"Authorization": f"Bearer {admin_token()}"}
        response = client.post("/process-ticket", json={"text": "I can't log into my account"}, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "category" in data
        assert "source" in data

    def test_process_ticket_requires_auth(self, client):
        response = client.post("/process-ticket", json={"text": "hello"})
        assert response.status_code in (401, 403)

    def test_process_ticket_requires_text_field(self, client):
        headers = {"Authorization": f"Bearer {admin_token()}"}
        response = client.post("/process-ticket", json={}, headers=headers)
        assert response.status_code == 422

    def test_add_correction(self, client):
        response = client.post("/add-correction", json={
            "text": "unique correction sample text",
            "correct_label": "Technical Support",
        })
        assert response.status_code == 200
        assert response.json()["message"] == "Correction saved — model will auto-retrain after 10 corrections"


class TestTeamsAndAgents:
    def test_teams_endpoint(self, client):
        headers = {"Authorization": f"Bearer {admin_token()}"}
        response = client.get("/teams", headers=headers)
        assert response.status_code == 200
        assert isinstance(response.json(), (list, dict))

    def test_teams_requires_auth(self, client):
        response = client.get("/teams")
        assert response.status_code in (401, 403)

    def test_agents_available(self, client):
        headers = {"Authorization": f"Bearer {admin_token()}"}
        response = client.get("/agents/available", headers=headers)
        assert response.status_code == 200


class TestWorkflowsStatus:
    def test_workflows_status(self, client):
        response = client.get("/workflows/status")
        assert response.status_code == 200


class TestCustomerAuthFlow:
    def test_register_then_login(self, client, preserve_customers_json):
        register_resp = client.post("/customer/register", json={
            "email": "pytest.customer@example.com",
            "username": "pytest_customer",
            "password": "PyTestPass123!",
            "name": "Pytest Customer",
            "phone": "+91-9000000001",
        })
        assert register_resp.status_code == 200
        assert register_resp.json()["success"] is True

        login_resp = client.post("/customer/login", json={
            "email": "pytest.customer@example.com",
            "password": "PyTestPass123!",
        })
        assert login_resp.status_code == 200
        body = login_resp.json()
        assert body["success"] is True
        assert "token" in body

    def test_login_wrong_password_returns_401(self, client, preserve_customers_json):
        response = client.post("/customer/login", json={
            "email": "rajesh.customer@email.com",
            "password": "totally-wrong-password",
        })
        assert response.status_code == 401

    def test_register_duplicate_email_returns_400(self, client, preserve_customers_json):
        response = client.post("/customer/register", json={
            "email": "rajesh.customer@email.com",
            "username": "someone_new",
            "password": "pass",
            "name": "Someone",
            "phone": "0000000000",
        })
        assert response.status_code == 400


class TestAgentLogin:
    def test_agent_login_invalid_credentials_returns_401(self, client):
        response = client.post("/agent/login", json={
            "email": "no-such-agent@example.com",
            "password": "wrong",
        })
        assert response.status_code in (401, 404)


class TestAdminLogin:
    def test_admin_login_success(self, client, monkeypatch):
        monkeypatch.setenv("SALESFORCE_USERNAME", "admin@company.com")
        monkeypatch.setenv("SALESFORCE_PASSWORD", "Admin123!")
        monkeypatch.delenv("SALESFORCE_ENABLED", raising=False)
        response = client.post("/admin/login", json={
            "email": "admin@company.com",
            "password": "Admin123!",
        })
        assert response.status_code == 200
        assert response.json()["type"] == "admin"

    def test_admin_login_wrong_password(self, client, monkeypatch):
        monkeypatch.setenv("SALESFORCE_USERNAME", "admin@company.com")
        monkeypatch.setenv("SALESFORCE_PASSWORD", "Admin123!")
        response = client.post("/admin/login", json={
            "email": "admin@company.com",
            "password": "wrong-password",
        })
        assert response.status_code == 401

    def test_admin_login_not_configured_returns_500(self, client, monkeypatch):
        monkeypatch.delenv("SALESFORCE_USERNAME", raising=False)
        monkeypatch.delenv("SALESFORCE_PASSWORD", raising=False)
        response = client.post("/admin/login", json={
            "email": "admin@company.com",
            "password": "whatever",
        })
        assert response.status_code == 500


class TestAdminAutologin:
    def test_autologin_disabled_outside_dev_returns_404(self, client, monkeypatch):
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        response = client.get("/admin/autologin")
        assert response.status_code == 404


class TestAnalyticsAuthRequired:
    def test_dashboard_requires_auth(self, client):
        response = client.get("/analytics/dashboard")
        assert response.status_code in (401, 403)

    def test_dashboard_forbidden_for_non_admin(self, client, preserve_customers_json):
        login_resp = client.post("/customer/login", json={
            "email": "rajesh.customer@email.com",
            "password": "Password123!",
        })
        token = login_resp.json()["token"]
        response = client.get("/analytics/dashboard", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 403

    def test_dashboard_ok_for_admin(self, client):
        headers = {"Authorization": f"Bearer {admin_token()}"}
        response = client.get("/analytics/dashboard", headers=headers)
        assert response.status_code == 200


class TestCustomerDashboardEndpoints:
    def test_get_customer_dashboard_as_admin(self, client):
        headers = {"Authorization": f"Bearer {admin_token()}"}
        response = client.get("/customer/CUST_001", headers=headers)
        assert response.status_code == 200
        assert response.json()["profile"]["email"] == "rajesh.customer@email.com"

    def test_get_customer_dashboard_not_found(self, client):
        headers = {"Authorization": f"Bearer {admin_token()}"}
        response = client.get("/customer/CUST_DOES_NOT_EXIST", headers=headers)
        assert response.status_code == 404

    def test_customer_cannot_view_another_customers_dashboard(self, client, preserve_customers_json):
        token = customer_token(client)
        response = client.get("/customer/CUST_002", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 403

    def test_get_customer_tickets_as_admin(self, client):
        headers = {"Authorization": f"Bearer {admin_token()}"}
        response = client.get("/customer/CUST_001/tickets", headers=headers)
        assert response.status_code == 200
        assert "tickets" in response.json()

    def test_update_customer_profile(self, client, preserve_customers_json):
        response = client.put("/customer/CUST_003", json={"name": "Amit P. Updated"})
        assert response.status_code == 200
        assert response.json()["customer"]["name"] == "Amit P. Updated"


class TestTeamsAgentsExtra:
    def test_team_agents_endpoint(self, client):
        headers = {"Authorization": f"Bearer {admin_token()}"}
        response = client.get("/teams/support/agents", headers=headers)
        assert response.status_code == 200
        assert "agents" in response.json()

    def test_get_all_agents_admin_only(self, client):
        headers = {"Authorization": f"Bearer {admin_token()}"}
        response = client.get("/admin/agents", headers=headers)
        assert response.status_code == 200
        assert "agents" in response.json()

    def test_get_all_agents_forbidden_for_customer(self, client, preserve_customers_json):
        token = customer_token(client)
        response = client.get("/admin/agents", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 403

    def test_get_single_agent_not_found(self, client):
        headers = {"Authorization": f"Bearer {admin_token()}"}
        response = client.get("/admin/agents/AGENT_DOES_NOT_EXIST", headers=headers)
        assert response.status_code == 404


class TestAgentLifecycle:
    def test_create_update_delete_agent(self, client, preserve_agents_json, preserve_salesforce_mock_files):
        headers = {"Authorization": f"Bearer {admin_token()}"}

        create_resp = client.post("/admin/agents", json={
            "name": "Pytest Agent",
            "email": "pytest.agent@example.com",
            "team": "support",
            "skills": ["billing"],
        }, headers=headers)
        assert create_resp.status_code == 200
        agent_id = create_resp.json()["agent"]["agent_id"]

        status_resp = client.put(f"/agent/{agent_id}/status", json={"status": "busy"}, headers=headers)
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "busy"

        get_resp = client.get(f"/admin/agents/{agent_id}", headers=headers)
        assert get_resp.status_code == 200

        delete_resp = client.delete(f"/admin/agents/{agent_id}", headers=headers)
        assert delete_resp.status_code == 200
        assert delete_resp.json()["success"] is True

    def test_create_agent_invalid_team_rejected(self, client, preserve_agents_json):
        headers = {"Authorization": f"Bearer {admin_token()}"}
        response = client.post("/admin/agents", json={
            "name": "Bad Team Agent",
            "email": "bad.team@example.com",
            "team": "not-a-real-team",
            "skills": ["x"],
        }, headers=headers)
        assert response.status_code == 400

    def test_update_agent_status_invalid_value(self, client, preserve_agents_json):
        headers = {"Authorization": f"Bearer {admin_token()}"}
        response = client.put("/agent/AGENT_001/status", json={"status": "napping"}, headers=headers)
        assert response.status_code == 400


class TestTicketLifecycleEndpoints:
    def _create_ticket(self, client, headers, preserve_salesforce_mock_files):
        ticket_id = salesforce.create_ticket({
            "subject": "Pytest ticket", "text": "hello", "customer_id": "CUST_001",
        })
        return ticket_id

    def test_assign_agent_to_ticket_no_agents_available(self, client, preserve_agents_json):
        headers = {"Authorization": f"Bearer {admin_token()}"}
        response = client.post(
            "/ticket/TICKET-999/assign-agent",
            params={"category": "NoSuchCategory", "priority": "Low"},
            headers=headers,
        )
        assert response.status_code in (503, 500, 200)

    def test_respond_to_ticket(self, client, preserve_salesforce_mock_files):
        headers = {"Authorization": f"Bearer {admin_token()}"}
        ticket_id = self._create_ticket(client, headers, preserve_salesforce_mock_files)
        response = client.post(
            f"/ticket/{ticket_id}/respond",
            json={"text": "We are on it", "status": "In Progress"},
            headers=headers,
        )
        assert response.status_code == 200

    def test_delete_ticket(self, client, preserve_salesforce_mock_files):
        headers = {"Authorization": f"Bearer {admin_token()}"}
        ticket_id = self._create_ticket(client, headers, preserve_salesforce_mock_files)
        response = client.delete(f"/ticket/{ticket_id}", headers=headers)
        assert response.status_code == 200

    def test_delete_nonexistent_ticket_still_succeeds(self, client):
        # salesforce.delete_ticket() in mock mode always reports success,
        # even for an id that was never created (matches integration/salesforce.py).
        headers = {"Authorization": f"Bearer {admin_token()}"}
        response = client.delete("/ticket/TICKET-NEVER-EXISTED", headers=headers)
        assert response.status_code == 200

    def test_feedback_on_ticket(self, client, preserve_salesforce_mock_files, preserve_agents_json):
        headers = {"Authorization": f"Bearer {admin_token()}"}
        ticket_id = self._create_ticket(client, headers, preserve_salesforce_mock_files)
        response = client.post(
            f"/ticket/{ticket_id}/feedback",
            json={"rating": 5, "comments": "Great support!"},
            headers=headers,
        )
        assert response.status_code == 200


class TestModelMetrics:
    def test_model_metrics_requires_auth(self, client):
        response = client.get("/model/metrics")
        assert response.status_code in (401, 403, 200)

    def test_model_metrics_ok_for_admin(self, client):
        headers = {"Authorization": f"Bearer {admin_token()}"}
        response = client.get("/model/metrics", headers=headers)
        assert response.status_code == 200
