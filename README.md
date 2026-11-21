# Retrieval Regression Check

Compare two ranked search results for the same questions and find queries that got worse. Use it when changing a document index, chunking strategy, or reranker in a retrieval-augmented generation (RAG) system.

## Try it

Requires Python 3.8 or later. The demo runs locally with the standard library.

```sh
python3 compare.py
```

The included example has five fictional financial-document questions and two hand-written sets of rankings. At a cutoff of three, average nDCG rises from **0.600 to 0.926**. One question still regresses. For “When does the promotional interest rate expire?”, a general rates document displaces the relevant terms from first to second. Recall stays at 1 for that question while nDCG falls from 1 to 0.631.

## Compare your own runs

Copy [the example JSON](examples/financial-docs.json) and replace its queries, judgments, and results. Use the same stable query and document IDs across both runs. Each query has a question and at least one relevant document. Both runs map query IDs to lists of document IDs, best result first. Missing queries count as empty rankings. Duplicate document IDs in a single list and unknown query IDs are rejected.

```sh
python3 compare.py my-runs.json -k 5
python3 compare.py my-runs.json -k 5 --json > report.json
python3 compare.py my-runs.json -k 5 --fail-on-drop
```

The last command prints the report and exits with code 1 when any query loses nDCG at that cutoff. It exits with code 0 when the check passes. Invalid inputs exit with code 2.

| Metric | Question it answers |
| --- | --- |
| Recall@k | What fraction of all known relevant documents made the top k? |
| MRR@k | How early did the first relevant result appear, averaged across questions? |
| nDCG@k | How close was the result order to placing every relevant document first, up to k? |

All queries carry equal weight. Relevance is binary. A result absent from the relevant-ID list receives zero credit, so the report depends on the quality and completeness of the judgments. The demo numbers illustrate the calculation on synthetic rankings.

## Run the checks

```sh
python3 -m unittest -v
```
