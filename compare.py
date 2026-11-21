"""Compare two document retrieval runs against binary relevance judgments."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean


METRICS = ("recall", "mrr", "ndcg")


def ids(value, location, *, allow_empty=True):
    if not isinstance(value, list) or not all(isinstance(x, str) and x.strip() for x in value):
        raise ValueError(f"{location} must be a list of nonempty document IDs")
    if not value and not allow_empty:
        raise ValueError(f"{location} must include a relevant document")
    if len(value) != len(set(value)):
        raise ValueError(f"{location} contains duplicate document IDs")
    return value


def score(ranking, relevant, k):
    hits = [i for i, doc in enumerate(ranking[:k], start=1) if doc in relevant]
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return {
        "recall": len(hits) / len(relevant),
        "mrr": 1 / hits[0] if hits else 0.0,
        "ndcg": sum(1 / math.log2(i + 1) for i in hits) / ideal,
    }


def compare(data, k):
    if k < 1:
        raise ValueError("k must be positive")
    if not isinstance(data, dict) or not isinstance(data.get("queries"), dict) or not data["queries"]:
        raise ValueError("queries must be a nonempty object keyed by query ID")
    queries = data["queries"]
    for name in ("baseline", "candidate"):
        run = data.get(name)
        if not isinstance(run, dict):
            raise ValueError(f"{name} must map query IDs to ranked document lists")
        unknown = set(run) - set(queries)
        if unknown:
            raise ValueError(f"{name} contains unknown query IDs: {sorted(unknown)}")

    rows = []
    for query_id, query in queries.items():
        if not query_id.strip() or not isinstance(query, dict):
            raise ValueError("each query must have a nonempty ID and an object value")
        question = query.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{query_id} needs a question")
        relevant = set(ids(query.get("relevant"), query_id, allow_empty=False))
        row = {"id": query_id, "question": question}
        for name in ("baseline", "candidate"):
            ranking = ids(data[name].get(query_id, []), f"{name}.{query_id}")
            row[name] = score(ranking, relevant, k)
        row["ndcg_delta"] = row["candidate"]["ndcg"] - row["baseline"]["ndcg"]
        rows.append(row)

    summary = {
        name: {metric: mean(row[name][metric] for row in rows) for metric in METRICS}
        for name in ("baseline", "candidate")
    }
    return {"k": k, "summary": summary,
            "queries": sorted(rows, key=lambda row: (row["ndcg_delta"], row["id"]))}


def render(report):
    k = report["k"]
    print(f"{len(report['queries'])} queries · top {k} results\n")
    print(f"{'Metric':<14} {'Baseline':>8} {'Candidate':>10} {'Change':>9}")
    for metric in METRICS:
        before = report["summary"]["baseline"][metric]
        after = report["summary"]["candidate"][metric]
        label = f"{metric.upper()}@{k}"
        print(f"{label:<14} {before:>8.3f} {after:>10.3f} {after-before:>+9.3f}")
    print("\nPer-query change in nDCG (largest drops first)")
    for row in report["queries"]:
        print(f"{row['ndcg_delta']:+.3f}  {row['id']}  {row['question']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    demo = Path(__file__).resolve().parent / "examples" / "financial-docs.json"
    parser.add_argument("input", nargs="?", type=Path, default=demo)
    parser.add_argument("-k", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="emit machine-readable scores")
    parser.add_argument("--fail-on-drop", action="store_true",
                        help="exit 1 if any query loses nDCG at the chosen cutoff")
    args = parser.parse_args()
    try:
        report = compare(json.loads(args.input.read_text(encoding="utf-8")), args.k)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        render(report)
    return int(args.fail_on_drop and any(row["ndcg_delta"] < -1e-12 for row in report["queries"]))


if __name__ == "__main__":
    raise SystemExit(main())
