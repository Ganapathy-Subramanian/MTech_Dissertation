from fastapi.testclient import TestClient
from main_enhanced import app
client = TestClient(app)

print("=== Testing HTML page serving ===")
for page in ['login.html', 'index.html', 'customer-dashboard.html', 'agent-dashboard.html', 'admin-dashboard.html']:
    r = client.get(f'/{page}')
    print(f'{page}: {r.status_code}')
    if r.status_code == 200:
        if '/static/style.css' in r.text:
            print(f'  ✓ Contains /static/style.css reference')
        if '/static/auth.js' in r.text or '/static/customer-dashboard.js' in r.text or '/static/admin-dashboard.js' in r.text or '/static/agent-dashboard.js' in r.text or '/static/app.js' in r.text:
            print(f'  ✓ Contains /static/JS reference')

print("\n=== Testing CORS headers ===")
r = client.options('/admin/login')
print(f'OPTIONS /admin/login: {r.status_code}')
if 'access-control-allow-origin' in r.headers:
    print(f'  ✓ Has CORS header: {r.headers["access-control-allow-origin"]}')
