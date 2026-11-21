import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

from compare import compare, score

ROOT = Path(__file__).resolve().parent


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "examples/financial-docs.json").read_text())

    def test_rank_discount_and_cutoff(self):
        result = score(["other", "a", "b"], {"a", "b"}, 2)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["mrr"], 0.5)
        discount = 1 / math.log2(3)
        self.assertAlmostEqual(result["ndcg"], discount / (1 + discount))

    def test_missing_query_scores_zero(self):
        del self.data["candidate"]["q1"]
        rows = compare(self.data, 3)["queries"]
        self.assertEqual(rows[0]["id"], "q1")
        self.assertEqual(rows[0]["candidate"], {"mrr": 0, "recall": 0, "ndcg": 0})

    def test_reject_duplicate_results(self):
        self.data["candidate"]["q1"] = ["a", "a"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compare(self.data, 3)

    def test_identical_rankings(self):
        self.data["candidate"] = self.data["baseline"]
        self.assertTrue(all(row["ndcg_delta"] == 0 for row in compare(self.data, 3)["queries"]))

    def test_cli_catches_drop_despite_average_gain(self):
        run = subprocess.run([sys.executable, str(ROOT / "compare.py"), "--json", "--fail-on-drop"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 1)
        report = json.loads(run.stdout)
        self.assertGreater(report["summary"]["candidate"]["ndcg"], report["summary"]["baseline"]["ndcg"])
        self.assertEqual(report["queries"][0]["id"], "q1")


if __name__ == "__main__":
    unittest.main()
