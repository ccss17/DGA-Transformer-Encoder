from pathlib import Path

from src.dga_transformer_encoder.prepare_data import stream_dataset

def percentile(data, p):
    """Calculate the p-th percentile of data"""
    k = (len(data) - 1) * p / 100
    f = int(k)
    c = k - f
    if f + 1 < len(data):
        return data[f] + c * (data[f + 1] - data[f])
    else:
        return data[f]


def analyze_domain_lengths(data_path: str = "data/raw/dga-training-data-encoded.json.gz"):
    """Analyze domain length distribution from raw data."""
    
    data_file = Path(data_path)
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print("Loading data...")
    lengths = []
    domains = []
    
    for domain, _ in stream_dataset(str(data_file)):
        lengths.append(len(domain))
        domains.append(domain)
    
    # Sort for percentile calculation
    lengths.sort()
    n = len(lengths)
    
    print(f"Loaded {n:,} domains\n")
    
    percentiles = [10, 25, 50, 75, 90, 95, 99, 99.9]
    
    print(f"{"=" * 50}\nDomain Length Distribution\n{"=" * 50}")
    
    for p in percentiles:
        val = percentile(lengths, p)
        print(f"{p:>5.1f}th percentile: {val:>6.1f} chars")
    
    print(f"\n{'Minimum':<20}: {min(lengths)} chars")
    print(f"{'Maximum':<20}: {max(lengths)} chars")
    print(f"{'Mean':<20}: {sum(lengths) / len(lengths):.1f} chars")
    
    print(f"\n{"=" * 50}\nCoverage Analysis for max_len=64\n{"=" * 50}")
    
    # Count domains that fit within max_len=64
    within_64 = sum(1 for length in lengths if length <= 64)
    coverage = (within_64 / n) * 100
    
    print(f"Domains <= 64 chars: {within_64:,} / {n:,} ({coverage:.2f}%)")
    print(f"Domains > 64 chars: {n - within_64:,} ({100 - coverage:.2f}%)")
    
    print(f"\n{"=" * 50}\nExamples of domains > 64 chars (will be truncated)\n{"=" * 50}")
    
    long_domains = [(domain, len(domain)) for domain in domains if len(domain) > 64]
    
    if long_domains:
        for domain, length in sorted(long_domains, key=lambda x: x[1], reverse=True)[:10]:
            print(f"{length:>3} chars: {domain}")
    else:
        print("No domains exceed 64 characters!")
    
    print(f"\n{"=" * 50}\nRecommendation\n{"=" * 50}")
    print(f"max_len=64 covers {coverage:.2f}% of domains")
    print("Good balance between coverage and computational efficiency")
    if coverage >= 99.9:
        print("No truncation occurs - all domains fit within max_len=64")
    else:
        print("Long domains (>64 chars) will be truncated but are very rare")


if __name__ == "__main__":
    analyze_domain_lengths()
