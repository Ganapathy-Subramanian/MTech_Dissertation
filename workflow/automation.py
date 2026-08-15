import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from integration.salesforce import salesforce

class WorkflowEngine:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.workflows_file = os.path.join(self.base_dir, "workflows.json")
        self._init_workflows()

    def _init_workflows(self):
        """Initialize workflow configurations"""
        default_workflows = {
            "Billing & Payments": {
                "auto_actions": ["send_payment_link", "schedule_followup"],
                "escalation_time": 3600,  # 1 hour
                "priority": "High",
                "department": "Finance"
            },
            "Technical Support": {
                "auto_actions": ["create_support_ticket", "notify_engineering"],
                "escalation_time": 1800,  # 30 minutes
                "priority": "High",
                "department": "Engineering"
            },
            "Account & Password": {
                "auto_actions": ["send_password_reset", "verify_identity"],
                "escalation_time": 900,  # 15 minutes
                "priority": "Medium",
                "department": "Support"
            },
            "Refund & Returns": {
                "auto_actions": ["initiate_refund_process", "schedule_callback"],
                "escalation_time": 7200,  # 2 hours
                "priority": "Medium",
                "department": "Customer Success"
            },
            "Complaints & Feedback": {
                "auto_actions": ["escalate_to_supervisor", "schedule_urgent_callback"],
                "escalation_time": 600,  # 10 minutes
                "priority": "Critical",
                "department": "Management"
            }
        }

        if not os.path.exists(self.workflows_file):
            with open(self.workflows_file, 'w') as f:
                json.dump(default_workflows, f, indent=2)

    def get_actions_for_category(self, category: str) -> List[Dict[str, Any]]:
        """Get automated actions for a ticket category"""
        try:
            with open(self.workflows_file, 'r') as f:
                workflows = json.load(f)

            workflow = workflows.get(category, {})
            actions = workflow.get('auto_actions', [])

            # Convert action strings to structured actions
            structured_actions = []
            for action in actions:
                structured_actions.append({
                    "action": action,
                    "status": "pending",
                    "timestamp": datetime.now().isoformat(),
                    "description": self._get_action_description(action)
                })

            return structured_actions
        except Exception as e:
            print(f"Error getting workflow actions: {e}")
            return []

    def _get_action_description(self, action: str) -> str:
        """Get human-readable description for actions"""
        descriptions = {
            "send_payment_link": "Send secure payment link to customer",
            "schedule_followup": "Schedule automated follow-up in 24 hours",
            "create_support_ticket": "Create ticket in engineering system",
            "notify_engineering": "Notify on-call engineering team",
            "send_password_reset": "Send password reset email/SMS",
            "verify_identity": "Initiate identity verification process",
            "initiate_refund_process": "Start refund processing workflow",
            "schedule_callback": "Schedule callback with customer success rep",
            "escalate_to_supervisor": "Escalate to supervisor immediately",
            "schedule_urgent_callback": "Schedule urgent callback within 30 minutes"
        }
        return descriptions.get(action, action)

    def escalate_to_human(self, category: str, priority: str) -> List[Dict[str, Any]]:
        """Get escalation actions for complex tickets"""
        base_actions = self.get_actions_for_category(category)

        # Add escalation-specific actions
        escalation_actions = [
            {
                "action": "assign_to_agent",
                "status": "pending",
                "timestamp": datetime.now().isoformat(),
                "description": f"Assign to human agent (Priority: {priority})",
                "priority": priority
            },
            {
                "action": "create_crm_ticket",
                "status": "pending",
                "timestamp": datetime.now().isoformat(),
                "description": "Create detailed ticket in Salesforce CRM"
            }
        ]

        return base_actions + escalation_actions

    def execute_workflow_action(self, action_id: str, ticket_id: str, ticket_data: Dict = None) -> Dict[str, Any]:
        """Execute a specific workflow action with Salesforce integration"""
        # Default ticket_data if not provided
        if ticket_data is None:
            ticket_data = {}

        action_results = {
            "send_payment_link": self._execute_send_payment_link(ticket_data),
            "create_support_ticket": self._execute_create_support_ticket(ticket_data),
            "send_password_reset": self._execute_send_password_reset(ticket_data),
            "create_crm_ticket": self._execute_create_crm_ticket(ticket_data),
            "assign_to_agent": self._execute_assign_to_agent(ticket_id, ticket_data)
        }

        result = action_results.get(action_id, {
            "status": "completed",
            "result": f"Action {action_id} executed successfully"
        })

        # Log the execution
        self._log_action_execution(action_id, ticket_id, result)

        return result

    def _execute_send_payment_link(self, ticket_data: Dict) -> Dict[str, Any]:
        """Execute send payment link action"""
        # In real implementation, integrate with payment system
        return {
            "status": "completed",
            "result": "Payment link sent successfully",
            "payment_link": "https://billing.example.com/pay/12345"
        }

    def _execute_create_support_ticket(self, ticket_data: Dict) -> Dict[str, Any]:
        """Execute create support ticket action"""
        # In real implementation, integrate with ticketing system
        return {
            "status": "completed",
            "result": "Support ticket created in engineering system",
            "support_ticket_id": f"ENG_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }

    def _execute_send_password_reset(self, ticket_data: Dict) -> Dict[str, Any]:
        """Execute send password reset action"""
        # In real implementation, integrate with identity management system
        return {
            "status": "completed",
            "result": "Password reset email sent",
            "reset_link": "https://account.example.com/reset/12345"
        }

    def _execute_create_crm_ticket(self, ticket_data: Dict) -> Dict[str, Any]:
        """Execute create CRM ticket action using Salesforce integration"""
        sf_ticket_id = salesforce.create_ticket(ticket_data)

        if sf_ticket_id:
            return {
                "status": "completed",
                "result": f"Salesforce ticket created: {sf_ticket_id}",
                "salesforce_ticket_id": sf_ticket_id
            }
        else:
            return {
                "status": "failed",
                "result": "Failed to create Salesforce ticket"
            }

    def _execute_assign_to_agent(self, ticket_id: str, ticket_data: Dict) -> Dict[str, Any]:
        """Execute assign to agent action"""
        priority = ticket_data.get("priority", "Medium")

        # Auto-assign based on priority and availability (mock logic)
        agent_id = self._get_available_agent(priority)

        if agent_id:
            success = salesforce.assign_ticket(ticket_id, agent_id)
            if success:
                return {
                    "status": "completed",
                    "result": f"Ticket assigned to agent {agent_id}",
                    "assigned_agent": agent_id
                }

        return {
            "status": "pending",
            "result": "No available agents at this time"
        }

    def _get_available_agent(self, priority: str) -> Optional[str]:
        """Get an available agent based on priority (mock implementation)"""
        # In real implementation, check agent availability, workload, skills, etc.
        agents = {
            "Critical": ["agent_supervisor_1", "agent_lead_1"],
            "High": ["agent_senior_1", "agent_senior_2", "agent_lead_1"],
            "Medium": ["agent_mid_1", "agent_mid_2", "agent_senior_1"],
            "Low": ["agent_junior_1", "agent_junior_2", "agent_mid_1"]
        }

        available_agents = agents.get(priority, ["agent_general_1"])
        return available_agents[0] if available_agents else None

    def _log_action_execution(self, action_id: str, ticket_id: str, result: Dict):
        """Log workflow action execution"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action_id": action_id,
            "ticket_id": ticket_id,
            "result": result
        }

        log_file = os.path.join(self.base_dir, "workflow_logs.json")
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []

            logs.append(log_entry)

            # Keep only last 1000 entries
            if len(logs) > 1000:
                logs = logs[-500:]

            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"Error logging workflow action: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current workflow status and metrics"""
        try:
            log_file = os.path.join(self.base_dir, "workflow_logs.json")
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)

                # Calculate metrics
                total_actions = len(logs)
                completed_actions = len([l for l in logs if l['result'].get('status') == 'completed'])
                failed_actions = len([l for l in logs if l['result'].get('status') == 'failed'])

                # Recent activity (last 24 hours)
                recent_logs = [l for l in logs if datetime.fromisoformat(l['timestamp']) > datetime.now() - timedelta(hours=24)]

                return {
                    "total_actions_executed": total_actions,
                    "success_rate": completed_actions / total_actions if total_actions > 0 else 0,
                    "recent_activity": len(recent_logs),
                    "active_workflows": self._get_active_workflows(),
                    "salesforce_integration": salesforce._ensure_authenticated()
                }
        except Exception as e:
            print(f"Error getting workflow status: {e}")

        return {"error": "Unable to retrieve workflow status"}

    def _get_active_workflows(self) -> int:
        """Get count of currently active workflows"""
        # In a real implementation, this would check active tickets
        # For now, return a mock count
        return 5  # Mock active workflows

    def create_custom_workflow(self, category: str, config: Dict[str, Any]):
        """Create or update custom workflow for a category"""
        try:
            with open(self.workflows_file, 'r') as f:
                workflows = json.load(f)

            workflows[category] = config

            with open(self.workflows_file, 'w') as f:
                json.dump(workflows, f, indent=2)

            return {"message": f"Workflow for {category} updated successfully"}
        except Exception as e:
            return {"error": f"Failed to update workflow: {e}"}