"""
Real tests for security/auth.py (JWT/bcrypt AuthManager) and
security/customer_auth.py (CustomerAuthManager).

Uses tmp_path / monkeypatch so nothing here ever touches the real
security/customers.json used by the running app.
"""
import os
import json
import pytest
from datetime import timedelta

from security.auth import AuthManager
from security.customer_auth import CustomerAuthManager


# ── AuthManager (JWT + bcrypt) ────────────────────────────────────────────

class TestAuthManagerPasswords:
    def setup_method(self):
        self.auth = AuthManager()

    def test_hash_password_produces_different_string(self):
        hashed = self.auth.hash_password("MySecret123!")
        assert hashed != "MySecret123!"
        assert isinstance(hashed, str)

    def test_verify_password_correct(self):
        hashed = self.auth.hash_password("MySecret123!")
        assert self.auth.verify_password("MySecret123!", hashed) is True

    def test_verify_password_incorrect(self):
        hashed = self.auth.hash_password("MySecret123!")
        assert self.auth.verify_password("WrongPassword", hashed) is False


class TestAuthManagerTokens:
    def setup_method(self):
        self.auth = AuthManager()

    def test_create_and_verify_token_roundtrip(self):
        token = self.auth.create_access_token({"sub": "agent1", "role": "agent"})
        payload = self.auth.verify_token(token)
        assert payload is not None
        assert payload["sub"] == "agent1"
        assert payload["role"] == "agent"

    def test_verify_expired_token_returns_none(self):
        token = self.auth.create_access_token(
            {"sub": "agent1"}, expires_delta=timedelta(seconds=-10)
        )
        assert self.auth.verify_token(token) is None

    def test_verify_garbage_token_returns_none(self):
        assert self.auth.verify_token("not-a-real-jwt") is None


class TestAuthManagerAuthenticate:
    def setup_method(self):
        self.auth = AuthManager()

    def test_authenticate_unknown_user_returns_none(self, monkeypatch):
        monkeypatch.delenv("LOCAL_ADMIN_PASSWORD", raising=False)
        result = self.auth.authenticate_user("no_such_user@example.com", "whatever")
        assert result is None

    def test_authenticate_local_admin_success(self, monkeypatch):
        monkeypatch.setenv("LOCAL_ADMIN_USERNAME", "admin")
        monkeypatch.setenv("LOCAL_ADMIN_EMAIL", "admin@company.com")
        monkeypatch.setenv("LOCAL_ADMIN_PASSWORD", "Admin123!")
        token = self.auth.authenticate_user("admin@company.com", "Admin123!")
        assert token is not None
        payload = self.auth.verify_token(token)
        assert payload["role"] == "admin"

    def test_authenticate_local_admin_wrong_password(self, monkeypatch):
        monkeypatch.setenv("LOCAL_ADMIN_PASSWORD", "Admin123!")
        result = self.auth.authenticate_user("admin@company.com", "WrongPass")
        assert result is None


class TestAuthManagerAuthorization:
    def setup_method(self):
        self.auth = AuthManager()

    def test_authorize_role_admin_bypasses_check(self):
        user = self.auth.authorize_role("agent", user={"role": "admin"})
        assert user["role"] == "admin"

    def test_authorize_role_matching_role_passes(self):
        user = self.auth.authorize_role("agent", user={"role": "agent"})
        assert user["role"] == "agent"

    def test_authorize_role_mismatch_raises_403(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            self.auth.authorize_role("admin", user={"role": "customer"})
        assert exc_info.value.status_code == 403

    def test_authorize_permission_present_passes(self):
        user = self.auth.authorize_permission(
            "delete", user={"role": "admin", "permissions": ["read", "delete"]}
        )
        assert "delete" in user["permissions"]

    def test_authorize_permission_missing_raises_403(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            self.auth.authorize_permission(
                "delete", user={"role": "agent", "permissions": ["read"]}
            )
        assert exc_info.value.status_code == 403


class TestAuthManagerLocalAgent:
    def setup_method(self):
        self.auth = AuthManager()

    def test_authenticate_local_agent_unknown_email(self):
        token = self.auth._authenticate_local_agent("no_such_agent@example.com", "whatever")
        assert token is None

    def test_authenticate_salesforce_user_handles_exception_gracefully(self, monkeypatch):
        # No SALESFORCE_ENABLED / broken lookup -> should return None, not raise.
        result = self.auth.authenticate_salesforce_user("nobody", "nopass")
        assert result is None


# ── CustomerAuthManager (registration / login) ────────────────────────────

@pytest.fixture
def temp_customer_auth(tmp_path):
    """A CustomerAuthManager pointed at a throwaway customers.json."""
    mgr = CustomerAuthManager.__new__(CustomerAuthManager)
    mgr.base_dir = str(tmp_path)
    mgr.customers_file = os.path.join(str(tmp_path), "customers.json")
    mgr._init_database()
    return mgr


class TestCustomerAuthManager:
    def test_init_database_creates_seed_customers(self, temp_customer_auth):
        with open(temp_customer_auth.customers_file) as f:
            data = json.load(f)
        assert "CUST_001" in data
        assert data["CUST_001"]["email"] == "rajesh.customer@email.com"

    def test_register_customer_success(self, temp_customer_auth):
        ok, cust_id = temp_customer_auth.register_customer(
            email="new.customer@example.com",
            username="new_customer",
            password="NewPass123!",
            name="New Customer",
            phone="+91-9000000000",
        )
        assert ok is True
        assert cust_id.startswith("CUST_")

    def test_register_customer_duplicate_email_fails(self, temp_customer_auth):
        ok, msg = temp_customer_auth.register_customer(
            email="rajesh.customer@email.com",
            username="someone_else",
            password="pass",
            name="Someone",
            phone="0000000000",
        )
        assert ok is False
        assert "already registered" in msg

    def test_login_customer_success(self, temp_customer_auth):
        ok, data = temp_customer_auth.login_customer(
            "rajesh.customer@email.com", "Password123!"
        )
        assert ok is True
        assert data["customer_id"] == "CUST_001"
        assert data["name"] == "Rajesh Kumar"

    def test_login_customer_wrong_password(self, temp_customer_auth):
        ok, msg = temp_customer_auth.login_customer(
            "rajesh.customer@email.com", "WrongPassword"
        )
        assert ok is False

    def test_login_customer_unknown_email(self, temp_customer_auth):
        ok, msg = temp_customer_auth.login_customer("nobody@example.com", "x")
        assert ok is False

    def test_get_customer_hides_password_hash(self, temp_customer_auth):
        data = temp_customer_auth.get_customer("CUST_001")
        assert data is not None
        assert "password_hash" not in data

    def test_get_customer_missing_returns_none(self, temp_customer_auth):
        assert temp_customer_auth.get_customer("CUST_999") is None

    def test_update_customer_allowed_fields(self, temp_customer_auth):
        ok = temp_customer_auth.update_customer("CUST_001", {"name": "Rajesh K."})
        assert ok is True
        assert temp_customer_auth.get_customer("CUST_001")["name"] == "Rajesh K."

    def test_update_customer_unknown_id(self, temp_customer_auth):
        assert temp_customer_auth.update_customer("CUST_999", {"name": "X"}) is False

    def test_get_all_customers_hides_password_hashes(self, temp_customer_auth):
        all_customers = temp_customer_auth.get_all_customers()
        assert len(all_customers) == 3
        for record in all_customers.values():
            assert "password_hash" not in record
