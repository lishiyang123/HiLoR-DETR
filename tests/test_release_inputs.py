"""Small input-diagnostic tests; no model, weights or GPU required."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier = load('_release_input_tests', ROOT / 'verify_release.py')


def record(root, name, value=b'x', present=True):
    if present:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    return dict(path=name, bytes=len(value), sha256=hashlib.sha256(value).hexdigest())


def inventory(root, rows):
    payload = dict(algorithm='sha256', file_count=len(rows),
                   total_bytes=sum(row['bytes'] for row in rows),
                   files=sorted(rows, key=lambda row: row['path']))
    (root / verifier.MANIFEST_NAME).write_text(json.dumps(payload), encoding='utf-8')


class ReleaseInputTests(unittest.TestCase):
    def test_missing_paths_are_reported_before_hashing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory(root, [record(root, 'main_host/model.pth', present=False)])
            with patch.object(verifier, 'sha256_file', side_effect=AssertionError('hashing ran early')):
                with self.assertRaisesRegex(RuntimeError, r'\[weights\]') as caught:
                    verifier.verify_manifest(root)
            self.assertIn('main_host/model.pth', str(caught.exception))
            self.assertIn('DISTRIBUTION.md#downloads-by-task', str(caught.exception))

    def test_missing_cdf_is_code_not_results(self):
        root = Path('/release')
        text = verifier.missing_input_message(root, [
            'main_host/source/FINAL_CODE/artifacts/ufd/cdw_difficulty_cdf_train640.json'])
        self.assertIn('[code]', text)
        self.assertNotIn('[results]', text)

    def test_unexpected_paths_are_reported_before_hashing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record(root, 'extra.txt')
            inventory(root, [])
            with patch.object(verifier, 'sha256_file', side_effect=AssertionError('hashing ran early')):
                with self.assertRaisesRegex(RuntimeError, 'Unexpected release files') as caught:
                    verifier.verify_manifest(root)
            self.assertIn('extra.txt', str(caught.exception))

    def test_unchanged_file_still_passes_integrity_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory(root, [record(root, 'code.txt', b'unchanged')])
            verifier.verify_manifest(root)

    def test_changed_file_still_fails_sha_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory(root, [record(root, 'code.txt', b'ab')])
            (root / 'code.txt').write_bytes(b'cd')
            with self.assertRaisesRegex(RuntimeError, 'Release file changed: code.txt'):
                verifier.verify_manifest(root)

    def test_external_weight_sha_check_is_not_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external = record(root, 'optional.pt', b'ab')
            inventory(root, [])
            (root / 'optional.pt').write_bytes(b'cd')
            with patch.object(verifier, 'external_asset_records', return_value={str(root / 'optional.pt'): external}):
                with self.assertRaisesRegex(RuntimeError, 'External asset SHA mismatch'):
                    verifier.verify_manifest(root)

    def test_full_missing_list_is_available(self):
        names = ['main_host/model%d.pth' % i for i in range(25)]
        preview = verifier.missing_input_message(Path('/release'), names)
        complete = verifier.missing_input_message(Path('/release'), names, limit=None)
        self.assertIn('--all-missing', preview)
        self.assertNotIn('more;', complete)
        self.assertTrue(all(name in complete for name in names))

    def test_cli_missing_inputs_have_no_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory(root, [record(root, 'missing.pth', present=False)])
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'verify_release.py'),
                                     'verify', '--release', str(root)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('missing.pth', result.stderr)
            self.assertNotIn('Traceback', result.stderr)

    def test_study_check_names_missing_capture_before_hashing(self):
        with patch.dict(sys.modules, {'verify_release': verifier}):
            studies = load('_study_input_tests', ROOT / 'reviewer_studies.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation = 'main_host/datasets/CDW1630/annotations/instances_test2017.json'
            capture = 'ranking_comparison/state/captures/gsa_seed0_test_capture.pt'
            rows = [record(root, annotation, b'{}'), record(root, capture, present=False)]
            inventory(root, rows)
            with patch.multiple(studies, ROOT=root, PRIMARY=root / 'main_host',
                                NN=root / 'ranking_comparison'):
                with patch.object(studies, 'sha', side_effect=AssertionError('hashing ran early')):
                    with self.assertRaisesRegex(RuntimeError, 'gsa_seed0_test_capture.pt'):
                        studies.check_bound_inputs()


if __name__ == '__main__':
    unittest.main()
