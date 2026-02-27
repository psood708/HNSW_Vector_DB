import requests
import numpy as np
import time

BASE_URL = "http://127.0.0.1:8000"
DIM = 128

def brute_force_search(query, dataset):
    """The 'Gold Standard' - 100% accurate but slow."""
    # Using dot product for normalized vectors as a proxy for cosine similarity
    similarities = np.dot(dataset, query)
    return np.argmax(similarities)

def run_benchmark(n_vectors=10000, n_queries=50):
    print(f"🚀 Starting Benchmark: {n_vectors} vectors, {n_queries} queries")
    
    # 1. Prepare Data
    data = np.random.rand(n_vectors, DIM).astype(np.float32)
    # Normalize for easier Cosine Similarity comparison
    data = data / np.linalg.norm(data, axis=1, keepdims=True)
    
    # 2. Insert Phase
    print("📥 Indexing...")
    for i, v in enumerate(data):
        requests.post(f"{BASE_URL}/insert", json={"id": i, "vector": v.tolist()})

    # 3. Accuracy & Speed Phase
    print("🧪 Running Tests...")
    hnsw_latencies = []
    brute_latencies = []
    matches = 0

    for i in range(n_queries):
        query = data[i]
        
        # --- Test HNSW (Your Rust Engine) ---
        start = time.perf_counter()
        resp = requests.post(f"{BASE_URL}/search", json={"query": query.tolist(), "k": 1})
        hnsw_latencies.append(time.perf_counter() - start)
        
        # --- Test Brute Force (Python Baseline) ---
        start_bf = time.perf_counter()
        true_neighbor_id = brute_force_search(query, data)
        brute_latencies.append(time.perf_counter() - start_bf)

        # --- Compare ---
        if resp.status_code == 200:
            result = resp.json()
            if result['matches']:
                found_id = result['matches'][0]['id']
                if found_id == true_neighbor_id:
                    matches += 1
                else:
                    if i < 5: # Debug first few failures
                        print(f"DEBUG: Expected ID {i} (type {type(i)}), Found ID {found_id} (type {type(found_id)})")
                        print(f"  Mismatch at {i}: HNSW found {found_id}, BF found {true_neighbor_id}")
        else:
            print(f"❌ Search Error: {resp.text}")

    # 4. Final Metrics
    recall = (matches / n_queries) * 100
    avg_hnsw = (sum(hnsw_latencies) / n_queries) * 1000
    avg_brute = (sum(brute_latencies) / n_queries) * 1000
    speedup = avg_brute / avg_hnsw

    print("\n" + "="*30)
    print(f"📊 FINAL RESULTS")
    print(f"🎯 Recall@1: {recall:.2f}%")
    print(f"⚡ Avg HNSW Latency: {avg_hnsw:.2f}ms")
    print(f"🐢 Avg Brute Force: {avg_brute:.2f}ms")
    print(f"🚀 Speedup Factor: {speedup:.1f}x faster than Linear Scan")
    print("="*30)

if __name__ == "__main__":
    run_benchmark()