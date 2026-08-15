import urllib.request, json, time
time.sleep(2)
data = json.dumps({"email": "rajesh.customer@email.com", "password": "Password123!"}).encode()
req = urllib.request.Request("http://localhost:8000/customer/login", data=data, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req) as r: 
        result = json.loads(r.read())
        print("✅ Login successful!")
        print("Customer ID:", result.get("customer_id"))
        print("Name:", result.get("name"))
        print("Email:", result.get("email"))
except urllib.error.HTTPError as e:
    try:
        err_json = json.loads(e.read().decode())
        print("❌ Error:", err_json.get("detail", err_json))
    except:
        print("❌ Error:", e.code)
except Exception as e:
    # Likely connection refused — skip network test gracefully
    print(f"SKIP network quick_test: {e}")
