import requests
import numpy as np
import time

# Note: Port updated to 8000 to match your Rust main.rs
BASE_URL = "http://127.0.0.1:8000"
DIM = 128

def run_performance_test(n_queries=100):
    print(f"\n📊 Running Speed Test: {n_queries} queries...")
    query_vectors = np.random.rand(n_queries, DIM).astype(np.float32)
    latencies = []

    for v in query_vectors:
        start = time.perf_counter()
        # FIX: Changed key from "vector" to "query" to match Rust struct
        requests.post(f"{BASE_URL}/search", json={"query": v.tolist()})
        latencies.append(time.perf_counter() - start)
    
    avg_ms = (sum(latencies) / n_queries) * 1000
    p99_ms = np.percentile(latencies, 99) * 1000
    print(f"✅ Avg Latency: {avg_ms:.2f}ms | p99 Latency: {p99_ms:.2f}ms")

def run_recall_test(n_test=50):
    print("🎯 Testing Recall Accuracy...")
    # 1. Generate test data
    test_data = np.random.rand(200, DIM).astype(np.float32)
    
    # 2. Insert into Rust
    print(f"📥 Inserting {len(test_data)} vectors...")
    for i, v in enumerate(test_data):
        resp = requests.post(f"{BASE_URL}/insert", json={"id": i, "vector": v.tolist()})
        if resp.status_code != 200:
            print(f"❌ Insert failed for ID {i}: {resp.text}")
            return

    # 3. Check accuracy
    print(f"🔍 Validating search results for {n_test} samples...")
    matches_count = 0
    for i in range(n_test):
        query = test_data[i]
        # FIX: Changed key to "query"
        response = requests.post(f"{BASE_URL}/search", json={"query": query.tolist()})
        
        if response.status_code == 200:
            try:
                data = response.json()
                # FIX: Your Rust handler returns { "matches": [ { "id": X, "score": Y } ] }
                if data['matches'] and data['matches'][0]['id'] == i:
                    matches_count += 1
            except (ValueError, KeyError, IndexError) as e:
                print(f"❌ Response parse error: {e} | Raw: {response.text}")
        else:
            print(f"❌ Search failed: {response.status_code} {response.text}")
            
    recall = (matches_count / n_test) * 100
    print(f"✅ Recall@1: {recall:.2f}% (Found {matches_count}/{n_test})")

if __name__ == "__main__":
    try:
        run_recall_test()
        run_performance_test()
    except requests.exceptions.ConnectionError:
        print("❌ FAILED: Could not connect to Rust server. Is it running on http://127.0.0.1:8000?")



# netstat -ano | findstr :8000 ( use this for cheecking TCP calls)