import sys
sys.path.append('.')
from integration.salesforce import SalesforceIntegration
from models.points_manager import points_manager
import json

sf = SalesforceIntegration()
# create a mock ticket
ticket_id = 'TICKET-TEST-001'
sf.mock_tickets[ticket_id] = {
    'id': ticket_id,
    'subject': 'Test ticket',
    'description': 'CUSTOMER QUESTION: Test\n\n',
    'status': 'New',
    'priority': 'Medium',
    'category': 'Emergency Services',
    'customer_id': 'CUST_TEST',
    'created_date': '2026-01-01T00:00:00',
    'last_modified': '2026-01-01T00:00:00'
}
# ensure no points yet
print('Before points:', points_manager.get_summary())
# update status to Resolved
ok = sf.update_ticket_status(ticket_id, 'Resolved')
print('Update ok:', ok)
print('After points:', points_manager.get_summary())
# show persisted file
with open('models/penalty_rewards.json','r') as f:
    print('File contents:', f.read())
