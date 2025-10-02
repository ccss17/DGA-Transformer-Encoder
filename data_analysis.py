#!/usr/bin/env python3
"""
Domain Length Analysis Script

Analyzes domain length distribution from the raw DGA training data.
Outputs percentile statistics used to determine the optimal max_len parameter.

Usage:
    python data_analysis.py
    pixi run python data_analysis.py
"""

import gzip
import json
from pathlib import Path


def analyze_domain_lengths(data_path: str = "data/raw/dga-training-data-encoded.json.gz"):
    """Analyze domain length distribution from raw data."""
    
    data_file = Path(data_path)
    
    if not data_file.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please run data preparation first or check the path.")
        return
    
    print(f"Analyzing domain lengths from: {data_path}\n")
    
    # Read and parse JSONL data (one JSON per line)
    print("Loading data...")
    lengths = []
    data_items = []
    
    with gzip.open(data_file, "rt", encoding="utf-8") as f:
        for line in f:
            # Skip comment lines
            if line.startswith("#"):
                continue
            
            # Parse JSON line
            item = json.loads(line.strip())
            domain = item["domain"]
            
            # Store length (domains are already without dots/TLDs)
            lengths.append(len(domain))
            data_items.append(item)
    
    # Sort for percentile calculation
    lengths.sort()
    n = len(lengths)
    
    print(f"Loaded {n:,} domains\n")
    
    # Calculate percentiles
    def percentile(data, p):
        """Calculate the p-th percentile of data."""
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = k - f
        if f + 1 < len(data):
            return data[f] + c * (data[f + 1] - data[f])
        else:
            return data[f]
    
    percentiles = [10, 25, 50, 75, 90, 95, 99, 99.9]
    
    print("=" * 50)
    print("Domain Length Distribution")
    print("=" * 50)
    
    for p in percentiles:
        val = percentile(lengths, p)
        print(f"{p:>5.1f}th percentile: {val:>6.1f} chars")
    
    print(f"\n{'Minimum':<20}: {min(lengths)} chars")
    print(f"{'Maximum':<20}: {max(lengths)} chars")
    print(f"{'Mean':<20}: {sum(lengths) / len(lengths):.1f} chars")
    
    print("\n" + "=" * 50)
    print("Coverage Analysis for max_len=64")
    print("=" * 50)
    
    # Count domains that fit within max_len=64
    within_64 = sum(1 for length in lengths if length <= 64)
    coverage = (within_64 / n) * 100
    
    print(f"Domains ≤ 64 chars: {within_64:,} / {n:,} ({coverage:.2f}%)")
    print(f"Domains > 64 chars: {n - within_64:,} ({100 - coverage:.2f}%)")
    
    # Show some examples of long domains
    print("\n" + "=" * 50)
    print("Examples of domains > 64 chars (will be truncated)")
    print("=" * 50)
    
    long_domains = [(item["domain"], len(item["domain"])) 
                    for item in data_items 
                    if len(item["domain"]) > 64]
    
    if long_domains:
        for domain, length in sorted(long_domains, key=lambda x: x[1], reverse=True)[:10]:
            print(f"{length:>3} chars: {domain}")
    else:
        print("No domains exceed 64 characters!")
    
    print("\n" + "=" * 50)
    print("Recommendation")
    print("=" * 50)
    print(f"max_len=64 covers {coverage:.2f}% of domains")
    print("Good balance between coverage and computational efficiency")
    if coverage >= 99.9:
        print("No truncation occurs - all domains fit within max_len=64")
    else:
        print("Long domains (>64 chars) will be truncated but are very rare")


if __name__ == "__main__":
    analyze_domain_lengths()
