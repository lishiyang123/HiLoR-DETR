"""Configuration/JSON binding checks with small synthetic files only."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_mapping_entry_test', SOURCE / 'reproduce.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class ProfileMappingTests(unittest.TestCase):
    def fixture(self, root):
        parameters = dict(beta0=0.125, beta_positive=1.1, beta_negative=0.9, duplicate_penalty=0.2)
        path = root / 'fit/qccr_params.json'
        path.parent.mkdir()
        path.write_text(json.dumps(dict(parameters=parameters)))
        config = {'PostProcessor': dict(qccr_enabled=True, **{'qccr_' + k: v for k, v in parameters.items()})}
        profile = {'inference_assets': [{'path': 'fit/qccr_params.json'}]}
        return config, profile, path

    def test_matching_mapping_is_accepted_without_rewriting_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, profile, _ = self.fixture(root)
            before = copy.deepcopy(config)
            entry.validate_profile_mapping(config, profile, root)
            self.assertEqual(config, before)

    def test_each_coefficient_mismatch_is_rejected(self):
        for name in ('beta0', 'beta_positive', 'beta_negative', 'duplicate_penalty'):
            with self.subTest(coefficient=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config, profile, _ = self.fixture(root)
                config['PostProcessor']['qccr_' + name] += 0.01
                before = copy.deepcopy(config)
                with self.assertRaisesRegex(RuntimeError, name):
                    entry.validate_profile_mapping(config, profile, root)
                self.assertEqual(config, before)

    def test_missing_mapping_file_has_a_download_hint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, profile, path = self.fixture(root)
            path.unlink()
            with self.assertRaisesRegex(RuntimeError, 'results archive'):
                entry.validate_profile_mapping(config, profile, root)

    def test_mapping_asset_must_be_unambiguous(self):
        for assets in ([], [{'path': 'a/qccr_params.json'}, {'path': 'b/qccr_params.json'}]):
            with self.subTest(assets=assets):
                with self.assertRaisesRegex(RuntimeError, 'one bound'):
                    entry.validate_profile_mapping({'PostProcessor': {'qccr_enabled': True}},
                                                   {'inference_assets': assets}, Path('/release'))

    def test_nonfinite_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, profile, path = self.fixture(root)
            value = json.loads(path.read_text())
            value['parameters']['beta0'] = float('nan')
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, 'beta0'):
                entry.validate_profile_mapping(config, profile, root)

    def test_native_postprocessing_needs_no_mapping(self):
        entry.validate_profile_mapping({'PostProcessor': {'qccr_enabled': False}}, {}, Path('/release'))


if __name__ == '__main__':
    unittest.main()
