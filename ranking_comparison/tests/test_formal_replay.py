import json
from pathlib import Path
import tempfile
import unittest

from nearest_neighbor_baselines.formal_replay import _qccr_parameters


class FormalReplayTests(unittest.TestCase):
    def test_qccr_parameters_accept_formal_nested_artifact(self):
        parameters = _qccr_parameters(
            {
                "parameters": {
                    "beta0": 1.0,
                    "beta_positive": 2.0,
                    "beta_negative": 3.0,
                    "duplicate_penalty": 0.5,
                }
            }
        )
        self.assertEqual(parameters["beta0"], 1.0)
        self.assertEqual(parameters["duplicate_penalty"], 0.5)

    def test_qccr_parameters_reject_invalid_constraints(self):
        with self.assertRaises(ValueError):
            _qccr_parameters(
                {
                    "beta0": 0.0,
                    "beta_positive": 0.0,
                    "beta_negative": 1.0,
                    "duplicate_penalty": 0.0,
                }
            )


if __name__ == "__main__":
    unittest.main()

