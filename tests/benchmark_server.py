import asyncio
import time
import aiohttp
import statistics
import argparse
import sys
import json
from typing import List, Dict

async def benchmark_request(session: aiohttp.ClientSession, url: str, payload: dict, stream: bool) -> Dict[str, float]:
    start_time = time.time()
    ttfb = None
    end_time = None
    
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                print(f"Error: {response.status}")
                try:
                    text = await response.text()
                    print(f"Body: {text}")
                except:
                    pass
                return {"ttfb": 0, "total": 0, "error": True}
            
            if stream:
                first_chunk = True
                async for _ in response.content.iter_chunked(1024):
                    if first_chunk:
                        ttfb = time.time() - start_time
                        first_chunk = False
            else:
                await response.read()
                ttfb = time.time() - start_time # For blocking, TTFB is essentially total time, but technically wait time. 
                                                # Actually for blocking, we get the whole body at once, so TTFB ~ Total.
            
            end_time = time.time()
            
    except Exception as e:
        print(f"Exception: {e}")
        return {"ttfb": 0, "total": 0, "error": True}

    return {
        "ttfb": ttfb if ttfb else (end_time - start_time),
        "total": end_time - start_time,
        "error": False
    }

import random
import string

async def run_benchmark(url: str, text: str, concurrency: int, num_requests: int, stream: bool):
    print(f"Benchmarking {url} with concurrency={concurrency}, total_requests={num_requests}, stream={stream}")
    
    base_payload = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 50,
        "speed": 1.0,
        "max_text_tokens_per_segment": 120
    }
    
    endpoint = f"{url}/tts/stream" if stream else f"{url}/tts"
    
    tasks = []
    results = []
    
    async with aiohttp.ClientSession() as session:
        # Warmup
        print("Warming up...")
        warmup_payload = base_payload.copy()
        warmup_payload["text"] = text
        await benchmark_request(session, endpoint, warmup_payload, stream)
        
        print("Starting benchmark...")
        start_global = time.time()
        
        for i in range(num_requests):
            # Randomize text to avoid cache
            rand_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            req_text = f"{text} {rand_suffix} {i}"
            payload = base_payload.copy()
            payload["text"] = req_text
            tasks.append(benchmark_request(session, endpoint, payload, stream))
            
        # This runs all at once? No, we need to limit concurrency.
        # But aiohttp + asyncio.gather with a semaphore or similar is better.
        
        chunked_tasks = [tasks[i:i + concurrency] for i in range(0, len(tasks), concurrency)]
        
        for chunk in chunked_tasks:
            chunk_results = await asyncio.gather(*chunk)
            results.extend(chunk_results)
            
        end_global = time.time()
        
    total_time = end_global - start_global
    valid_results = [r for r in results if not r["error"]]
    error_count = len(results) - len(valid_results)
    
    if not valid_results:
        print("All requests failed.")
        return

    ttfbs = [r["ttfb"] * 1000 for r in valid_results] # ms
    totals = [r["total"] * 1000 for r in valid_results] # ms
    
    print("\nResults:")
    print(f"Successful requests: {len(valid_results)}/{num_requests}")
    print(f"Failed requests: {error_count}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {len(valid_results) / total_time:.2f} RPS")
    
    print("\nLatency (ms):")
    print(f"TTFB: Avg={statistics.mean(ttfbs):.2f}, Median={statistics.median(ttfbs):.2f}, P95={sorted(ttfbs)[int(len(ttfbs)*0.95)]:.2f}")
    print(f"Total: Avg={statistics.mean(totals):.2f}, Median={statistics.median(totals):.2f}, P95={sorted(totals)[int(len(totals)*0.95)]:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TTS Server")
    parser.add_argument("--url", default="http://localhost:8009", help="Server URL")
    parser.add_argument("--text", default="Hello, this is a test of the emergency broadcast system.", help="Text to synthesize")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent requests")
    parser.add_argument("--requests", type=int, default=10, help="Total requests")
    parser.add_argument("--stream", action="store_true", help="Use streaming endpoint")
    
    args = parser.parse_args()
    
    asyncio.run(run_benchmark(args.url, args.text, args.concurrency, args.requests, args.stream))
