"""Regression checks for split isolation, probabilities, and failure propagation."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config as C
import numpy as np
import pandas as pd
from evaluate import compute_metrics, evaluate_and_save
from protocol import model_name, validate_splits


class ProtocolTests(unittest.TestCase):
    def test_reject_empty_or_mismatched_metric_inputs(self):
        for gold, pred in [([], []), (["a"], []), (["unknown"], ["a"])]:
            with self.assertRaises(ValueError):
                compute_metrics(gold, pred, labels=["a", "b"])

    def test_reject_nonfinite_probabilities(self):
        with self.assertRaises(ValueError):
            evaluate_and_save(
                "invalid", ["a"], ["a"], np.array([[np.nan, 0.0]]), class_names=["a", "b"]
            )

    def test_config_import_does_not_initialize_directories(self):
        spec = importlib.util.spec_from_file_location("isolated_config", ROOT / "src/config.py")
        module = importlib.util.module_from_spec(spec)
        with patch.object(Path, "mkdir") as mkdir:
            spec.loader.exec_module(module)
            mkdir.assert_not_called()

    def test_reject_cross_split_pairs(self):
        fit = pd.DataFrame(dict(text_a=["q"], text_b=["a"], group=["g1"]))
        val = pd.DataFrame(dict(text_a=["v"], text_b=["a"], group=["g2"]))
        test = pd.DataFrame(dict(text_a=["q"], text_b=["a"], group=["g3"]))
        with self.assertRaisesRegex(ValueError, "identical"):
            validate_splits(fit, val, test)

    def test_reject_probability_prediction_mismatch(self):
        with self.assertRaisesRegex(ValueError, "argmax"):
            evaluate_and_save(
                "invalid", ["a"], ["a"], np.array([[0.1, 0.9]]), class_names=["a", "b"]
            )

    def test_windows_model_name(self):
        self.assertEqual(model_name(r"D:\models\MiniLM"), "MiniLM")
        self.assertEqual(model_name("sentence-transformers/MiniLM"), "MiniLM")

    def test_stops_before_dependent_stage(self):
        spec = importlib.util.spec_from_file_location("run_all", ROOT / "run_all.py")
        run = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run)
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(C, "OUT", Path(temp)),
            patch.object(run, "commands", return_value=[("fail.py", []), ("must_not_run.py", [])]),
            patch.object(run.subprocess, "run") as child,
        ):
            child.return_value.returncode = 7
            self.assertEqual(run.main(["--stage", "encoder"]), 7)
            self.assertEqual(child.call_count, 1)

    def test_target_val_independent_and_grouped(self):
        import transfer

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            train = pd.DataFrame(
                [
                    dict(
                        text_a=f"q{i // 2}",
                        text_b=f"a{i}",
                        label=["yes", "no"][i % 2],
                        split="train",
                    )
                    for i in range(100)
                ]
            )
            test = pd.DataFrame(
                [dict(text_a="heldout", text_b="answer", label="yes", split="test")]
            )
            train.to_csv(out / "transfer_qnli_train.csv", index=False)
            test.to_csv(out / "transfer_qnli_test.csv", index=False)
            with patch.object(C, "DATA_PROC", out), patch.object(C, "METRICS", out):
                fit, val, test, _ = transfer.prepare_target("qnli")
            self.assertEqual(len(fit) + len(val), 100)
            self.assertFalse(set(fit.text_a) & set(val.text_a))
            self.assertNotIn("heldout", set(val.text_a))


if __name__ == "__main__":
    unittest.main()
