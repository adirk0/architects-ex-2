import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def mean(values):
    return sum(values) / len(values) if values else 0


def calculate_metrics(results):

    total = len(results)

    relevance = sum(
        r["relevant"]
        for r in results
    ) / total

    hallucination = sum(
        r["hallucination"]
        for r in results
    ) / total

    citation = mean([
        r["citation_accuracy"]
        for r in results
    ])

    latency = mean([
        r["latency_ms"]
        for r in results
        if r.get("latency_ms") is not None
    ])

    return {
        "questions": total,
        "relevance": relevance,
        "hallucination_rate": hallucination,
        "citation_accuracy": citation,
        "avg_latency_ms": latency,
    }


def breakdown(results, key):

    groups = defaultdict(list)

    for r in results:
        groups[r.get(key)].append(r)

    output = {}

    for name, items in groups.items():
        output[name] = calculate_metrics(items)

    return output


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        required=True,
        help="evaluation_results.jsonl"
    )

    args = parser.parse_args()

    results = load_jsonl(args.results)

    print("\n=== Overall Metrics ===")

    metrics = calculate_metrics(results)

    for k, v in metrics.items():

        if isinstance(v, float):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")


    print("\n=== By Domain ===")

    for domain, metrics in breakdown(results, "domain").items():

        print(f"\n{domain}")

        for k, v in metrics.items():

            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")


    print("\n=== By Difficulty ===")

    for difficulty, metrics in breakdown(results, "difficulty").items():

        print(f"\n{difficulty}")

        for k, v in metrics.items():

            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":

    main()