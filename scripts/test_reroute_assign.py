import sys
sys.path.append('.')
from integration.salesforce import salesforce
from agents.agent_manager import agent_team_manager

# create local mock ticket
ticket_id = 'TICKET-REROUTE-001'
salesforce.mock_tickets[ticket_id] = {
    'id': ticket_id,
    'subject': 'Reroute test',
    'description': 'Test',
    'status': 'New',
    'priority': 'Medium',
    'category': 'Complaints & Feedback',  # ⚠️ FIXED: Using valid 10-class category
    'customer_id': 'CUST1'
}

print('Before owner:', salesforce.mock_tickets[ticket_id].get('owner_id'))
res = agent_team_manager.reroute_ticket(from_agent_id='AGENT_001', ticket_id=ticket_id, to_agent_id=None, category='Complaints & Feedback', ticket_text='test')  # ⚠️ FIXED
print('Reroute result:', res)
print('After owner:', salesforce.mock_tickets[ticket_id].get('owner_id'))
print('Agent AGENT_001:', agent_team_manager.get_all_agents().get('AGENT_001'))
print('Agent AGENT_004:', agent_team_manager.get_all_agents().get('AGENT_004'))
