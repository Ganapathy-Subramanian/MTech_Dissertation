import json
import os
from agents.agent_manager import AgentTeamManager

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
AM = AgentTeamManager()

def test_reroute_assigns_ticket(tmp_path, monkeypatch):
    agents_file = AM.agents_file
    # Backup original
    with open(agents_file, 'r', encoding='utf-8') as f:
        orig = f.read()

    try:
        test_agents = {
            "AGENT_001": {"name":"Alice","email":"a@example.com","team":"support","status":"available","active_tickets":0},
            "AGENT_002": {"name":"Bob","email":"b@example.com","team":"billing","status":"available","active_tickets":0}
        }
        with open(agents_file, 'w', encoding='utf-8') as f:
            json.dump(test_agents, f, indent=2)

        # Call reroute without specifying to_agent_id; category billing should pick AGENT_002
        success = AM.reroute_ticket(from_agent_id="AGENT_001", ticket_id="TICKET-123", to_agent_id=None, category="Billing & Payments", ticket_text="My payment failed")
        assert success
        with open(agents_file, 'r', encoding='utf-8') as f:
            updated = json.load(f)
        assert updated['AGENT_002']['active_tickets'] == 1
    finally:
        # restore
        with open(agents_file, 'w', encoding='utf-8') as f:
            f.write(orig)
