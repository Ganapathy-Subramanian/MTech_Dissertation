from fastapi.testclient import TestClient
from main_enhanced import app
client = TestClient(app)
for path in ['/admin/login','/agent/login','/customer/login']:
    r = client.post(path, json={'email':'admin@company.com','password':'Admin123!'})
    print(path, r.status_code, r.text)
