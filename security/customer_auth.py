import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import hashlib
import secrets
from enum import Enum

class UserRole(str, Enum):
    CUSTOMER = "customer"
    AGENT = "agent"
    ADMIN = "admin"

class CustomerAuthManager:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.customers_file = os.path.join(self.base_dir, "customers.json")
        self._init_database()

    def _init_database(self):
        """Initialize customer database"""
        if not os.path.exists(self.customers_file):
            default_customers = {
                "CUST_001": {
                    "username": "rajesh_customer",
                    "email": "rajesh.customer@email.com",
                    "password_hash": self._hash_password("Password123!"),
                    "name": "Rajesh Kumar",
                    "phone": "+91-9876543210",
                    "company": "Tech Solutions Pvt Ltd",
                    "status": "active",
                    "created_date": "2025-01-15",
                    "last_login": datetime.now().isoformat(),
                    "tickets_count": 3,
                    "total_spent": 15000,
                    "tier": "silver"
                },
                "CUST_002": {
                    "username": "priya_customer",
                    "email": "priya.customer@email.com",
                    "password_hash": self._hash_password("SecurePass456!"),
                    "name": "Priya Singh",
                    "phone": "+91-9876543211",
                    "company": "Digital Services",
                    "status": "active",
                    "created_date": "2025-02-01",
                    "last_login": datetime.now().isoformat(),
                    "tickets_count": 5,
                    "total_spent": 25000,
                    "tier": "gold"
                },
                "CUST_003": {
                    "username": "amit_customer",
                    "email": "amit.customer@email.com",
                    "password_hash": self._hash_password("MyPassword789!"),
                    "name": "Amit Patel",
                    "phone": "+91-9876543212",
                    "company": "Startup Inc",
                    "status": "active",
                    "created_date": "2025-01-20",
                    "last_login": datetime.now().isoformat(),
                    "tickets_count": 2,
                    "total_spent": 8000,
                    "tier": "bronze"
                }
            }
            with open(self.customers_file, 'w') as f:
                json.dump(default_customers, f, indent=2)

    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(16)
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}${pwd_hash.hex()}"

    def _verify_password(self, password: str, pwd_hash: str) -> bool:
        """Verify password against hash"""
        try:
            salt, stored_hash = pwd_hash.split('$')
            pwd_check = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return pwd_check.hex() == stored_hash
        except:
            return False

    def register_customer(self, email: str, username: str, password: str, name: str, phone: str) -> Tuple[bool, str]:
        """Register new customer"""
        try:
            with open(self.customers_file, 'r') as f:
                customers = json.load(f)

            # Check if email exists
            for customer_data in customers.values():
                if customer_data['email'] == email or customer_data['username'] == username:
                    return False, "Email or username already registered"

            # Create new customer
            customer_id = f"CUST_{len(customers) + 1:03d}"
            customers[customer_id] = {
                "username": username,
                "email": email,
                "password_hash": self._hash_password(password),
                "name": name,
                "phone": phone,
                "company": "",
                "status": "active",
                "created_date": datetime.now().isoformat(),
                "last_login": None,
                "tickets_count": 0,
                "total_spent": 0,
                "tier": "bronze"
            }

            with open(self.customers_file, 'w') as f:
                json.dump(customers, f, indent=2)

            return True, customer_id

        except Exception as e:
            return False, str(e)

    def login_customer(self, email: str, password: str) -> Tuple[bool, Optional[Dict]]:
        """Authenticate customer"""
        try:
            with open(self.customers_file, 'r') as f:
                customers = json.load(f)

            # Find customer by email
            for customer_id, customer_data in customers.items():
                if customer_data['email'] == email:
                    # Verify password
                    if self._verify_password(password, customer_data['password_hash']):
                        # Update last login
                        customer_data['last_login'] = datetime.now().isoformat()
                        with open(self.customers_file, 'w') as f:
                            json.dump(customers, f, indent=2)

                        return True, {
                            "customer_id": customer_id,
                            "name": customer_data['name'],
                            "email": customer_data['email'],
                            "tier": customer_data['tier'],
                            "status": customer_data['status']
                        }

            return False, "Invalid credentials"

        except Exception as e:
            return False, str(e)

    def get_customer(self, customer_id: str) -> Optional[Dict]:
        """Get customer details"""
        try:
            with open(self.customers_file, 'r') as f:
                customers = json.load(f)

            if customer_id in customers:
                data = customers[customer_id].copy()
                data.pop('password_hash', None)  # Don't return password hash
                return data

            return None

        except Exception as e:
            print(f"Error getting customer: {e}")
            return None

    def update_customer(self, customer_id: str, updates: Dict) -> bool:
        """Update customer details"""
        try:
            with open(self.customers_file, 'r') as f:
                customers = json.load(f)

            if customer_id not in customers:
                return False

            # Don't allow password change here (separate endpoint)
            allowed_fields = ['name', 'phone', 'company']
            for field in allowed_fields:
                if field in updates:
                    customers[customer_id][field] = updates[field]

            with open(self.customers_file, 'w') as f:
                json.dump(customers, f, indent=2)

            return True

        except Exception as e:
            print(f"Error updating customer: {e}")
            return False

    def get_all_customers(self) -> Dict:
        """Get all customers (admin)"""
        try:
            with open(self.customers_file, 'r') as f:
                customers = json.load(f)

            # Remove password hashes
            for customer_data in customers.values():
                customer_data.pop('password_hash', None)

            return customers

        except Exception as e:
            print(f"Error getting customers: {e}")
            return {}

# Global instance
customer_auth_manager = CustomerAuthManager()
