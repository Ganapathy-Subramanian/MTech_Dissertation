import sys
sys.path.append('.')
from agents.agent_manager import agent_team_manager

print('Before:', agent_team_manager.get_all_agents().get('AGENT_001'))
res = agent_team_manager.reroute_ticket(from_agent_id='AGENT_001', ticket_id='TICKET-XYZ', to_agent_id='AGENT_001', category='Account & Password', ticket_text='sample')
print('Result:', res)
print('After:', agent_team_manager.get_all_agents().get('AGENT_001'))
