"""Regression checks for runnable-source and optional-result download boundaries."""
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('download_groups_under_test', ROOT / 'prepare_downloads.py')
downloads = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloads)


class DownloadGroupTests(unittest.TestCase):
    def test_cdw_difficulty_cdf_is_code(self):
        self.assertEqual(downloads.group_for(
            'main_host/source/FINAL_CODE/artifacts/ufd/cdw_difficulty_cdf_train640.json'), 'code')

    def test_btc_difficulty_cdf_is_code(self):
        self.assertEqual(downloads.group_for(
            'main_host/source/FINAL_CODE/artifacts/ufd/btc_difficulty_cdf_train640.json'), 'code')

    def test_other_artifact_json_remains_results(self):
        self.assertEqual(downloads.group_for('analysis/artifacts/summary.json'), 'results')

    def test_selected_weight_remains_weights(self):
        self.assertEqual(downloads.group_for('main_host/checkpoints/seed0/best.pth'), 'weights')

    def test_train_capture_remains_features(self):
        self.assertEqual(downloads.group_for('second_host/artifacts/seed0_train_capture.pt'), 'features')

    def test_predictions_remain_results(self):
        self.assertEqual(downloads.group_for('main_host/predictions.json'), 'results')


if __name__ == '__main__':
    unittest.main()
