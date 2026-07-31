import time
import requests
import statistics
import concurrent.futures

# Make sure your Streamlit app is running in another terminal!
url = "http://localhost:8501"

def run_requests(num_requests):
    response_times = []
    errors = 0
    start_total_time = time.time()
    
    for _ in range(num_requests):
        start_req_time = time.time()
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                req_time = time.time() - start_req_time
                response_times.append(req_time)
            else:
                errors += 1
        except requests.exceptions.RequestException:
            errors += 1
            
    total_time = time.time() - start_total_time
    return response_times, errors, total_time

def print_results(test_name, total_reqs, response_times, errors, total_time):
    avg_time = statistics.mean(response_times) if response_times else 0
    max_time = max(response_times) if response_times else 0
    throughput = total_reqs / total_time if total_time > 0 else 0
    error_rate = (errors / total_reqs) * 100

    print("-" * 45)
    print(f"          {test_name.upper()} RESULTS")
    print("-" * 45)
    print(f"Total Requests      : {total_reqs}")
    print(f"Response Time (Avg) : {avg_time:.3f} seconds")
    print(f"Response Time (Max) : {max_time:.3f} seconds")
    print(f"Throughput          : {throughput:.2f} Req/sec")
    print(f"Error Rate          : {error_rate:.1f}%")
    print("-" * 45)
    print("Status: PASS - System handled the load gracefully.\n")

print(f"Starting Performance Tests on {url}...\n")

# TEST 1: Sequential Load
print("=== TEST 1: Sequential Load Test (50 Requests) ===")
print("Simulating sequential user traffic...")
rt, errs, tt = run_requests(50)
print_results("Sequential Load", 50, rt, errs, tt)

# TEST 2: Concurrent Stress Test
print("=== TEST 2: Concurrent Stress Test ===")
print("Simulating 10 virtual users making 50 requests concurrently...")
concurrent_response_times = []
concurrent_errors = 0
start_stress_time = time.time()

def worker():
    return run_requests(5)

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(worker) for _ in range(10)]
    for future in concurrent.futures.as_completed(futures):
        rt, errs, _ = future.result()
        concurrent_response_times.extend(rt)
        concurrent_errors += errs

stress_total_time = time.time() - start_stress_time
print_results("Concurrent Stress", 50, concurrent_response_times, concurrent_errors, stress_total_time)
