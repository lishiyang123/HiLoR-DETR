import json
from pathlib import Path
import unittest


class ProtocolTests(unittest.TestCase):
    def test_protocol_locks_candidate_pool_and_formal_asset_boundary(self):
        root = Path(__file__).resolve().parents[1]
        protocol = json.loads((root / "protocol" / "protocol.json").read_text(encoding="utf-8"))
        self.assertTrue(protocol["formal_assets_policy"]["read_only"])
        self.assertTrue(protocol["formal_assets_policy"]["integration_requires_user_approval"])
        self.assertEqual(protocol["detector"]["candidate_pool"], "all 300 queries x 5 classes")
        self.assertEqual(protocol["detector"]["final_selection"], "deterministic global Top-300 query-class candidates")
        self.assertEqual(protocol["gilg_original"]["target"], "COCO-style one-to-one true-positive label at IoU=0.50")


if __name__ == "__main__":
    unittest.main()
