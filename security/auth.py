import os
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

load_dotenv()

class AuthManager:
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
        self.algorithm = "HS256"
        self.security = HTTPBearer()

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> Dict[str, Any]:
        """Get current authenticated user from JWT token"""
        token = credentials.credentials
        payload = self.verify_token(token)

        if payload is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        return payload

    def authenticate_user(self, username_or_email: str, password: str) -> Optional[str]:
        """Authenticate user and return access token"""
        users_db = self._get_users_db()

        user = None
        for record in users_db.values():
            if record["username"] == username_or_email or record["email"] == username_or_email:
                user = record
                break

        if user:
            stored_password = user.get("password")
            stored_hash = user.get("hashed_password")
            is_valid = (stored_password is not None and password == stored_password)
            if stored_hash:
                is_valid = self.verify_password(password, stored_hash)

            if not is_valid:
                return None

            # Create access token
            access_token_expires = timedelta(minutes=30)
            access_token = self.create_access_token(
                data={
                    "sub": user["username"],
                    "role": user["role"],
                    "email": user["email"],
                    "name": user["name"]
                },
                expires_delta=access_token_expires
            )
            return access_token

        # Fallback: authenticate locally created agents
        return self._authenticate_local_agent(username_or_email, password)

    def _get_users_db(self) -> Dict[str, Dict[str, Any]]:
        """Load local fallback users from environment variables only."""
        admin_username = os.getenv("LOCAL_ADMIN_USERNAME", "admin")
        admin_email = os.getenv("LOCAL_ADMIN_EMAIL", "admin@company.com")
        admin_password = os.getenv("LOCAL_ADMIN_PASSWORD")

        if not admin_password:
            return {}

        return {
            "admin": {
                "username": admin_username,
                "email": admin_email,
                "name": "Administrator",
                "password": admin_password,
                "role": "admin",
                "permissions": ["read", "write", "delete", "manage_users"]
            }
        }

    def authenticate_salesforce_user(self, username: str, password: str) -> Optional[str]:
        """
        Authenticate user against Salesforce and return access token
        """
        try:
            from integration.salesforce import salesforce

            # Get Salesforce users
            sf_users = salesforce.get_salesforce_users()

            # Find user by username
            sf_user = None
            for user in sf_users:
                if user.get('Username') == username:
                    sf_user = user
                    break

            if not sf_user or not sf_user.get('IsActive', False):
                return None

            if not password:
                return None

            # Get or create local agent
            from agents.agent_manager import AgentTeamManager
            agent_manager = AgentTeamManager()
            agent = agent_manager.get_salesforce_agent(sf_user['Id'])

            if not agent:
                # Sync this user
                sync_result = agent_manager.sync_salesforce_users([sf_user])
                if sync_result['success']:
                    agent = agent_manager.get_salesforce_agent(sf_user['Id'])

            if agent:
                # Create access token with agent info
                access_token_expires = timedelta(minutes=30)
                access_token = self.create_access_token(
                    data={
                        "sub": username,
                        "role": "agent",
                        "email": sf_user.get('Email'),
                        "name": sf_user.get('Name'),
                        "agent_id": agent['agent_id'],
                        "salesforce_id": sf_user['Id']
                    },
                    expires_delta=access_token_expires
                )
                return access_token

            return None

        except Exception as e:
            print(f"Error authenticating Salesforce user: {e}")
            return None

    def authenticate_user_with_provider(self, username_or_email: str, password: str, use_salesforce: bool = False) -> Optional[str]:
        """
        Authenticate user with option to use Salesforce
        """
        if use_salesforce:
            return self.authenticate_salesforce_user(username_or_email, password)
        return self.authenticate_user(username_or_email, password)

    def _authenticate_local_agent(self, username_or_email: str, password: str) -> Optional[str]:
        """Authenticate agent credentials stored in local agent database."""
        try:
            from agents.agent_manager import AgentTeamManager
            agent_manager = AgentTeamManager()
            agents = agent_manager.get_all_agents()

            matched_agent = None
            for agent_id, agent in agents.items():
                identifiers = {
                    (agent.get("email") or "").strip().lower(),
                    (agent.get("salesforce_username") or "").strip().lower()
                }
                if username_or_email.strip().lower() in identifiers:
                    matched_agent = {"agent_id": agent_id, **agent}
                    break

            if not matched_agent:
                return None

            stored_hash = matched_agent.get("hashed_password")
            stored_password = matched_agent.get("password")
            is_valid = False
            if stored_hash:
                is_valid = self.verify_password(password, stored_hash)
            elif stored_password is not None:
                is_valid = stored_password == password

            if not is_valid:
                return None

            return self.create_access_token(
                data={
                    "sub": matched_agent.get("salesforce_username") or matched_agent.get("email"),
                    "role": "agent",
                    "email": matched_agent.get("email"),
                    "name": matched_agent.get("name", "Agent"),
                    "agent_id": matched_agent.get("agent_id"),
                    "team": matched_agent.get("team"),
                    "salesforce_id": matched_agent.get("salesforce_id")
                },
                expires_delta=timedelta(minutes=30)
            )
        except Exception as e:
            print(f"Error authenticating local agent: {e}")
            return None

    def authorize_role(self, required_role: str, user: Dict = Depends(get_current_user)):
        """Check if user has required role"""
        user_role = user.get("role")
        if user_role != required_role and user_role != "admin":
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )
        return user

    def authorize_permission(self, required_permission: str, user: Dict = Depends(get_current_user)):
        """Check if user has required permission"""
        user_permissions = user.get("permissions", [])
        if required_permission not in user_permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required: {required_permission}"
            )
        return user

# Dependency functions for FastAPI
def get_auth_manager():
    return AuthManager()

def get_current_user(auth_manager: AuthManager = Depends(get_auth_manager),
                    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    return auth_manager.get_current_user(credentials)