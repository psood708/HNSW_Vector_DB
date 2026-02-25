import requests
import numpy as np
import time

BASE_URL = "http://127.0.0.1:8080"
DIM = 128

def run_performance_test(n_queries=1000):
    print(f"📊 Running Speed Test: {n_queries} queries...")
    query_vectors = np.random.rand(n_queries, DIM).astype(np.float32)
    latencies = []

    for v in query_vectors:
        start = time.perf_counter()
        requests.post(f"{BASE_URL}/search", json={"vector": v.tolist()})
        latencies.append(time.perf_counter() - start)
    
    avg_ms = (sum(latencies) / n_queries) * 1000
    p99_ms = np.percentile(latencies, 99) * 1000
    print(f"✅ Avg Latency: {avg_ms:.2f}ms | p99 Latency: {p99_ms:.2f}ms")

def run_recall_test(n_test=50):
    print("🎯 Testing Recall Accuracy...")
    # 1. Get 50 vectors from our local "ground truth"
    test_data = np.random.rand(500, DIM).astype(np.float32)
    
    # 2. Insert into Rust
    for i, v in enumerate(test_data):
        requests.post(f"{BASE_URL}/insert", json={"id": i, "vector": v.tolist()})

    # 3. Check accuracy
    matches = 0
    for i in range(n_test):
        query = test_data[i]
        response = requests.post(f"{BASE_URL}/search", json={"vector": query.tolist()}).json()
        if response.get('id') == i:
            matches += 1
            
    recall = (matches / n_test) * 100
    print(f"✅ Recall@1: {recall}%")

if __name__ == "__main__":
    # Ensure Rust server is running!
    run_recall_test()
    run_performance_test()