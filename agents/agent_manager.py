import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

class AgentStatus(str, Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    ON_BREAK = "on_break"

class TeamType(str, Enum):
    SUPPORT = "support"
    BILLING = "billing"
    TECHNICAL = "technical"
    ESCALATION = "escalation"

class AgentTeamManager:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.agents_file = os.path.join(self.base_dir, "agent_database.json")
        self.teams_file = os.path.join(self.base_dir, "team_database.json")
        self._init_databases()

    def _init_databases(self):
        """Initialize agent and team databases"""
        if not os.path.exists(self.agents_file):
            # Start with empty agent database - admin will create agents
            default_agents = {}
            with open(self.agents_file, 'w') as f:
                json.dump(default_agents, f, indent=2)

        if not os.path.exists(self.teams_file):
            default_teams = {
                "support": {
                    "name": "Customer Support",
                    "description": "General customer support queries",
                    "agents": [],
                    "skills": ["billing", "account", "general"],
                    "capacity": 50
                },
                "technical": {
                    "name": "Technical Support",
                    "description": "Technical issues and troubleshooting",
                    "agents": [],
                    "skills": ["technical_support", "refund", "orders"],
                    "capacity": 40
                },
                "billing": {
                    "name": "Billing Team",
                    "description": "Billing and payment issues",
                    "agents": [],
                    "skills": ["billing", "payments", "refund"],
                    "capacity": 35
                },
                "escalation": {
                    "name": "Escalation Team",
                    "description": "Complex issues requiring senior attention",
                    "agents": [],
                    "skills": ["complex_issues", "escalation", "vip"],
                    "capacity": 25
                },
                "claims": {
                    "name": "Claims Team",
                    "description": "Insurance claims, policy coverage, policy changes, emergency services",
                    "agents": [],
                    "skills": ["claims", "policy", "emergency", "coverage"],
                    "capacity": 60
                }
            }
            with open(self.teams_file, 'w') as f:
                json.dump(default_teams, f, indent=2)

    CATEGORY_TO_TEAM_MAP = {
        # ── Billing team ──────────────────────────────────────────────────────
        "Billing & Payments": "billing",
        "Billing": "billing",
        "Refund & Returns": "billing",
        # ── Technical team ───────────────────────────────────────────────────
        "Technical Support": "technical",
        "Technical": "technical",
        "Account & Password": "technical",
        # ── Claims team ──────────────────────────────────────────────────────
        "Claims": "claims",
        "Policy & Coverage": "claims",
        "Policy Changes": "claims",
        "Emergency Services": "claims",
        # ── Support team ─────────────────────────────────────────────────────
        "General Inquiry": "support",
        "Support": "support",
        # ── Escalation team ──────────────────────────────────────────────────
        "Complaints & Feedback": "escalation",
        "Refund & Returns": "escalation",
        "Escalation": "escalation",
    }

    def get_best_agent(self, category: str, priority: str) -> Optional[Dict]:
        """Get best available agent based on category (mapped to team) and priority"""
        with open(self.agents_file, 'r') as f:
            agents = json.load(f)

        # 1. Map category to a specific team (Case-insensitive & Partial Match)
        target_team = "support" # Default
        cat_lower = category.lower()
        
        for cat_key, team_val in self.CATEGORY_TO_TEAM_MAP.items():
            if cat_key.lower() in cat_lower or cat_lower in cat_key.lower():
                target_team = team_val
                break
        
        # 2. Filter available agents in that team
        team_agents = []
        for aid, adata in agents.items():
            agent_team = adata.get('team', '').lower()
            if adata.get('status') == 'available':
                if agent_team == target_team.lower() or agent_team == category.lower():
                    team_agents.append((aid, adata))

        # 3. Fallback: If no agent in that team, look for ANY available agent
        if not team_agents:
            available_agents = [(aid, adata) for aid, adata in agents.items() if adata['status'] == 'available']
        else:
            available_agents = team_agents

        if not available_agents:
            return None

        # 4. Sort by rating and active tickets (prefer less busy, higher rated)
        available_agents.sort(key=lambda x: (-x[1].get('rating', 0), x[1].get('active_tickets', 0)))

        agent_id, agent_data = available_agents[0]
        return {
            "agent_id": agent_id,
            "name": agent_data['name'],
            "email": agent_data['email'],
            "team": agent_data['team'],
            "rating": agent_data.get('rating', 5.0),
            "salesforce_id": agent_data.get('salesforce_id'),
            "salesforce_contact_id": agent_data.get('salesforce_contact_id')
        }

    def assign_ticket(self, agent_id: str, ticket_id: str) -> bool:
        """Assign ticket to agent"""
        try:
            with open(self.agents_file, 'r') as f:
                agents = json.load(f)

            if agent_id not in agents:
                return False

            agents[agent_id]['active_tickets'] += 1

            with open(self.agents_file, 'w') as f:
                json.dump(agents, f, indent=2)

            return True
        except Exception as e:
            print(f"Error assigning ticket: {e}")
            return False

    def complete_ticket(self, agent_id: str) -> bool:
        """Mark ticket as completed for agent"""
        try:
            with open(self.agents_file, 'r') as f:
                agents = json.load(f)

            if agent_id not in agents:
                return False

            agents[agent_id]['active_tickets'] = max(0, agents[agent_id]['active_tickets'] - 1)
            agents[agent_id]['resolved_tickets'] += 1

            with open(self.agents_file, 'w') as f:
                json.dump(agents, f, indent=2)

            return True
        except Exception as e:
            print(f"Error completing ticket: {e}")
            return False

    def get_all_agents(self) -> Dict:
        """Get all agents"""
        with open(self.agents_file, 'r') as f:
            return json.load(f)

    def get_all_teams(self) -> Dict:
        """Get all teams"""
        with open(self.teams_file, 'r') as f:
            return json.load(f)

    def get_team_agents(self, team: str) -> List[Dict]:
        """Get all agents in a team"""
        with open(self.agents_file, 'r') as f:
            agents = json.load(f)

        return [
            {"id": aid, **adata}
            for aid, adata in agents.items()
            if adata.get('team') == team
        ]

    def update_agent_status(self, agent_id: str, status: str) -> bool:
        """Update agent status"""
        try:
            with open(self.agents_file, 'r') as f:
                agents = json.load(f)

            if agent_id not in agents:
                return False

            agents[agent_id]['status'] = status

            with open(self.agents_file, 'w') as f:
                json.dump(agents, f, indent=2)

            return True
        except Exception as e:
            print(f"Error updating agent status: {e}")
            return False

    def create_agent(self, name: str, email: str, team: str, skills: List[str] = None) -> Dict:
        """Create new agent"""
        try:
            with open(self.agents_file, 'r') as f:
                agents = json.load(f)

            # Generate agent ID safely by finding the maximum existing ID
            if not agents:
                next_id = 1
            else:
                try:
                    # Parse AGENT_XXX numbers
                    max_id = max(
                        int(aid.split('_')[1]) 
                        for aid in agents.keys() 
                        if aid.startswith('AGENT_') and len(aid.split('_')) == 2 and aid.split('_')[1].isdigit()
                    )
                    next_id = max_id + 1
                except (ValueError, IndexError):
                    next_id = len(agents) + 1
                    
            agent_id = f"AGENT_{str(next_id).zfill(3)}"

            new_agent = {
                "name": name,
                "email": email,
                "team": team,
                "status": "offline",  # Start as offline until they log in
                "skills": skills or [],
                "active_tickets": 0,
                "resolved_tickets": 0,
                "rating": 5.0,  # Start with perfect rating
                "rating_count": 0,
                "total_rating_sum": 0,
                "joined_date": datetime.now().strftime("%Y-%m-%d")
            }

            agents[agent_id] = new_agent

            # Update team — map category name → team key in team_database
            team_key = self.CATEGORY_TO_TEAM_MAP.get(team, "support")
            with open(self.teams_file, 'r') as f:
                teams = json.load(f)

            if team_key in teams:
                if agent_id not in teams[team_key]['agents']:
                    teams[team_key]['agents'].append(agent_id)
                with open(self.teams_file, 'w') as f:
                    json.dump(teams, f, indent=2)

            with open(self.agents_file, 'w') as f:
                json.dump(agents, f, indent=2)

            return {"agent_id": agent_id, **new_agent}
        except Exception as e:
            print(f"Error creating agent: {e}")
            return None

    def update_agent_rating(self, agent_id: str, new_rating: int) -> bool:
        """Update agent rating based on customer feedback"""
        try:
            with open(self.agents_file, 'r') as f:
                agents = json.load(f)

            if agent_id not in agents:
                # If agent_id is not in local DB, it might be a SF ID. 
                # We should try to find the local agent.
                agent = self.get_salesforce_agent(agent_id)
                if agent:
                    agent_id = agent['agent_id']
                else:
                    return False

            # Initialize fields if they don't exist (for existing agents)
            if 'rating_count' not in agents[agent_id]:
                agents[agent_id]['rating_count'] = 0
            if 'total_rating_sum' not in agents[agent_id]:
                agents[agent_id]['total_rating_sum'] = 0

            agents[agent_id]['rating_count'] += 1
            agents[agent_id]['total_rating_sum'] += new_rating
            agents[agent_id]['rating'] = round(agents[agent_id]['total_rating_sum'] / agents[agent_id]['rating_count'], 1)

            with open(self.agents_file, 'w') as f:
                json.dump(agents, f, indent=2)

            return True
        except Exception as e:
            print(f"Error updating agent rating: {e}")
            return False

    def delete_agent(self, agent_id: str) -> bool:
        """Delete agent"""
        try:
            with open(self.agents_file, 'r') as f:
                agents = json.load(f)

            if agent_id not in agents:
                return False

            agent_data = agents[agent_id]
            team = agent_data.get('team')

            # Remove from agents
            del agents[agent_id]

            # Remove from team
            if team:
                with open(self.teams_file, 'r') as f:
                    teams = json.load(f)

                if team in teams and agent_id in teams[team]['agents']:
                    teams[team]['agents'].remove(agent_id)

                    with open(self.teams_file, 'w') as f:
                        json.dump(teams, f, indent=2)

            with open(self.agents_file, 'w') as f:
                json.dump(agents, f, indent=2)

            return True
        except Exception as e:
            print(f"Error deleting agent: {e}")
            return False

    def update_agent(self, agent_id: str, agent_data: Dict[str, Any]) -> bool:
        """Update agent with provided data"""
        try:
            with open(self.agents_file, 'r') as f:
                agents = json.load(f)

            if agent_id not in agents:
                return False

            # Get old team
            old_team = agents[agent_id].get('team')
            
            # Update the agent fields
            for key, value in agent_data.items():
                if key in ['name', 'email', 'team', 'skills', 'status']:
                    agents[agent_id][key] = value

            # Handle team change if applicable
            new_team = agent_data.get('team')
            if new_team and new_team != old_team:
                with open(self.teams_file, 'r') as f:
                    teams = json.load(f)

                # Remove from old team
                if old_team in teams and agent_id in teams[old_team]['agents']:
                    teams[old_team]['agents'].remove(agent_id)

                # Add to new team
                if new_team in teams:
                    if agent_id not in teams[new_team]['agents']:
                        teams[new_team]['agents'].append(agent_id)

                with open(self.teams_file, 'w') as f:
                    json.dump(teams, f, indent=2)

            with open(self.agents_file, 'w') as f:
                json.dump(agents, f, indent=2)

            return True
        except Exception as e:
            print(f"Error updating agent: {e}")
            return False

    def sync_salesforce_users(self, salesforce_users: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Sync Salesforce users with local agent database
        """
        try:
            with open(self.agents_file, 'r') as f:
                agents = json.load(f)

            synced_count = 0
            updated_count = 0

            for sf_user in salesforce_users:
                # Check if agent already exists by email
                existing_agent_id = None
                for agent_id, agent_data in agents.items():
                    if agent_data.get('email') == sf_user.get('Email'):
                        existing_agent_id = agent_id
                        break

                # Determine team based on Salesforce role/profile
                team = self._determine_team_from_salesforce(sf_user)

                if existing_agent_id:
                    # Update existing agent
                    agents[existing_agent_id].update({
                        'name': sf_user.get('Name', ''),
                        'salesforce_id': sf_user.get('Id'),
                        'is_active': sf_user.get('IsActive', True),
                        'team': team,
                        'last_sync': datetime.now().isoformat()
                    })
                    updated_count += 1
                else:
                    # Create new agent
                    agent_id = f"SF_{sf_user.get('Id')}"
                    agents[agent_id] = {
                        'name': sf_user.get('Name', ''),
                        'email': sf_user.get('Email', ''),
                        'salesforce_id': sf_user.get('Id'),
                        'team': team,
                        'status': 'offline',  # Start as offline
                        'skills': self._get_skills_for_team(team),
                        'active_tickets': 0,
                        'resolved_tickets': 0,
                        'rating': 5.0,
                        'joined_date': datetime.now().strftime('%Y-%m-%d'),
                        'is_active': sf_user.get('IsActive', True),
                        'last_sync': datetime.now().isoformat()
                    }
                    synced_count += 1

                    # Add to team
                    self._add_agent_to_team(agent_id, team)

            with open(self.agents_file, 'w') as f:
                json.dump(agents, f, indent=2)

            return {
                'success': True,
                'synced': synced_count,
                'updated': updated_count,
                'total_agents': len(agents)
            }

        except Exception as e:
            print(f"Error syncing Salesforce users: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _determine_team_from_salesforce(self, sf_user: Dict[str, Any]) -> str:
        """
        Determine team based on Salesforce user role/profile
        """
        role_name = sf_user.get('UserRole', {}).get('Name', '').lower()
        profile_name = sf_user.get('Profile', {}).get('Name', '').lower()

        # Map Salesforce roles/profiles to teams
        if 'technical' in role_name or 'technical' in profile_name:
            return 'technical'
        elif 'billing' in role_name or 'billing' in profile_name:
            return 'billing'
        elif 'escalation' in role_name or 'senior' in profile_name or 'manager' in profile_name:
            return 'escalation'
        else:
            return 'support'  # Default team

    def _get_skills_for_team(self, team: str) -> List[str]:
        """
        Get default skills for a team
        """
        skills_map = {
            'support': ['billing', 'account', 'general'],
            'technical': ['technical_support', 'troubleshooting', 'orders'],
            'billing': ['billing', 'payments', 'refunds'],
            'escalation': ['complex_issues', 'escalation', 'vip_support']
        }
        return skills_map.get(team, [])

    def _add_agent_to_team(self, agent_id: str, team: str):
        """
        Add agent to team
        """
        try:
            with open(self.teams_file, 'r') as f:
                teams = json.load(f)

            if team in teams and agent_id not in teams[team]['agents']:
                teams[team]['agents'].append(agent_id)

                with open(self.teams_file, 'w') as f:
                    json.dump(teams, f, indent=2)
        except Exception as e:
            print(f"Error adding agent to team: {e}")

    def get_salesforce_agent(self, salesforce_id: str) -> Optional[Dict[str, Any]]:
        """
        Get agent by Salesforce ID
        """
        agents = self.get_all_agents()
        for agent_id, agent_data in agents.items():
            # Match by known SF user id
            if agent_data.get('salesforce_id') == salesforce_id:
                return {'agent_id': agent_id, **agent_data}
            # Match by contact id (sometimes stored) or username/email
            if agent_data.get('salesforce_contact_id') == salesforce_id:
                return {'agent_id': agent_id, **agent_data}
            if agent_data.get('salesforce_username') == salesforce_id:
                return {'agent_id': agent_id, **agent_data}
            # Also allow passing an email to match
            if agent_data.get('email') == salesforce_id:
                return {'agent_id': agent_id, **agent_data}
        return None

    def update_agent_salesforce_info(
        self,
        agent_id: str,
        salesforce_user_id: Optional[str] = None,
        salesforce_contact_id: Optional[str] = None,
        salesforce_username: Optional[str] = None
    ) -> bool:
        """Persist Salesforce metadata for an agent record."""
        try:
            with open(self.agents_file, 'r') as f:
                agents = json.load(f)

            if agent_id not in agents:
                return False

            if salesforce_user_id:
                agents[agent_id]['salesforce_id'] = salesforce_user_id
            if salesforce_contact_id:
                agents[agent_id]['salesforce_contact_id'] = salesforce_contact_id
            if salesforce_username:
                agents[agent_id]['salesforce_username'] = salesforce_username

            with open(self.agents_file, 'w') as f:
                json.dump(agents, f, indent=2)
            return True
        except Exception as e:
            print(f"Error updating Salesforce metadata: {e}")
            return False

    def update_agent_auth_info(
        self,
        agent_id: str,
        login_username: Optional[str] = None,
        hashed_password: Optional[str] = None
    ) -> bool:
        """Persist local login credentials for an agent."""
        try:
            with open(self.agents_file, 'r') as f:
                agents = json.load(f)

            if agent_id not in agents:
                return False

            if login_username:
                agents[agent_id]['salesforce_username'] = login_username
            if hashed_password:
                agents[agent_id]['hashed_password'] = hashed_password

            with open(self.agents_file, 'w') as f:
                json.dump(agents, f, indent=2)
            return True
        except Exception as e:
            print(f"Error updating agent auth info: {e}")
            return False
    def reroute_ticket(self, from_agent_id: str, ticket_id: str, to_agent_id: Optional[str] = None,
                      category: Optional[str] = None, ticket_text: Optional[str] = None) -> bool:
        """Reroute a ticket from one agent to another.

        If `to_agent_id` is not provided, select the best available agent for `category`.
        Optionally provide `ticket_text` and `category` so the system can record
        the correction for self-learning (triggers retrain asynchronously).
        """
        try:
            with open(self.agents_file, 'r') as f:
                agents = json.load(f)

            # Resolve Salesforce owner IDs to local agent IDs if necessary
            original_from = from_agent_id
            if from_agent_id and from_agent_id not in agents:
                try:
                    sf_agent = self.get_salesforce_agent(from_agent_id)
                    if sf_agent:
                        from_agent_id = sf_agent['agent_id']
                except Exception:
                    pass

            if to_agent_id and to_agent_id not in agents:
                try:
                    sf_agent = self.get_salesforce_agent(to_agent_id)
                    if sf_agent:
                        to_agent_id = sf_agent['agent_id']
                except Exception:
                    pass

            # If target agent resolves to same as source, treat as successful no-op
            if to_agent_id and from_agent_id and to_agent_id == from_agent_id:
                # write back current agents and return True (no change)
                with open(self.agents_file, 'w') as f:
                    json.dump(agents, f, indent=2)
                print(f"[REROUTE] No-op: from_agent == to_agent ({to_agent_id}) for ticket {ticket_id}")
                return True

            # Decrement from_agent active count if present
            if from_agent_id in agents:
                agents[from_agent_id]['active_tickets'] = max(0, agents[from_agent_id].get('active_tickets', 0) - 1)

            # If no explicit target agent, pick best by category
            if not to_agent_id:
                if not category:
                    # If no category, fall back to any available agent
                    available = [aid for aid, a in agents.items() if a.get('status') == 'available']
                    to_agent_id = available[0] if available else None
                else:
                    best = self.get_best_agent(category, priority="normal")
                    to_agent_id = best['agent_id'] if best else None

            if not to_agent_id or to_agent_id not in agents:
                # Write back agents file and return failure
                with open(self.agents_file, 'w') as f:
                    json.dump(agents, f, indent=2)
                return False

            # Assign to target agent
            agents[to_agent_id]['active_tickets'] = agents[to_agent_id].get('active_tickets', 0) + 1

            with open(self.agents_file, 'w') as f:
                json.dump(agents, f, indent=2)

            # Update ticket owner in Salesforce/local mock so UI reflects assignment
            try:
                from integration.salesforce import salesforce
                # Prefer using the agent's Salesforce user id when available so SF OwnerId is updated
                sf_owner = agents[to_agent_id].get('salesforce_id') if agents.get(to_agent_id) else None
                try:
                    if sf_owner:
                        # update SF with the 005... owner id but preserve local owner id mapping
                        salesforce.update_ticket_status(ticket_id, 'Assigned', {'owner_id': sf_owner})
                        # ensure local mock stores the local agent id for dashboard/UI
                        if ticket_id in salesforce.mock_tickets:
                            salesforce.mock_tickets[ticket_id]['owner_id'] = to_agent_id
                            salesforce._save_mock_tickets()
                    else:
                        # no SF owner available, update local mock with local agent id
                        salesforce.update_ticket_status(ticket_id, 'Assigned', {'owner_id': to_agent_id})
                except Exception:
                    pass
            except Exception:
                pass

            # Optionally record correction so model learns from reroute
            if ticket_text and category:
                try:
                    # Import here to avoid circular imports at module load
                    from models.auto_retrain import SelfLearningWrapper
                    import threading
                    # also update penalty/reward points: penalize at reroute time
                    try:
                        from models.points_manager import points_manager
                        # Deduct 2 points at reroute time for the corrected category
                        points_manager.adjust_points(category, -2)
                    except Exception:
                        pass

                    def _record_and_retrain():
                        wrapper = SelfLearningWrapper(model=None)
                        wrapper.add_correction(ticket_text, category, original_category=None)

                    threading.Thread(target=_record_and_retrain, daemon=True).start()
                except Exception:
                    pass

            return True
        except Exception as e:
            print(f"Error rerouting ticket: {e}")
            return False

# Global instance
agent_team_manager = AgentTeamManager()
