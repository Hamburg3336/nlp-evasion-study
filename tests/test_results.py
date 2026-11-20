"""Offline repository and failure-path tests; no GPU, network, or dataset needed."""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_results import compute_metrics, safe_path, verify


class PublishedResultsTests(unittest.TestCase):
    def test_published_results(self):
        result = verify(ROOT)
        self.assertEqual(result["prediction_files"], 34)

    def test_invalid_label_arrays(self):
        for gold, pred, labels in [
            ([], [], ["a"]),
            (["a"], [], ["a"]),
            (["x"], ["a"], ["a"]),
            (["a"], ["a"], ["a", "a"]),
        ]:
            with self.assertRaises(ValueError):
                compute_metrics(gold, pred, labels)

    def test_perfect_and_missing_class(self):
        score = compute_metrics(["a"], ["a"], ["a", "b"])
        self.assertEqual(score["accuracy"], 1.0)
        self.assertEqual(score["macro_f1"], 0.5)
        self.assertEqual(score["per_class"]["b"]["support"], 0)

    def test_reject_escaping_path(self):
        with self.assertRaises(ValueError):
            safe_path(ROOT, "../outside")

    def test_launcher_stops_at_failure(self):
        spec = importlib.util.spec_from_file_location("study", ROOT / "run_study.py")
        study = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(study)
        with patch.object(study.subprocess, "run") as child:
            child.return_value.returncode = 7
            self.assertEqual(study.main(["--stage", "all"]), 7)
            self.assertEqual(child.call_count, 1)

    def test_launcher_dry_run_does_not_execute(self):
        spec = importlib.util.spec_from_file_location("study", ROOT / "run_study.py")
        study = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(study)
        with patch.object(study.subprocess, "run") as child:
            self.assertEqual(study.main(["--stage", "supplement", "--dry-run"]), 0)
            child.assert_not_called()


if __name__ == "__main__":
    unittest.main()
