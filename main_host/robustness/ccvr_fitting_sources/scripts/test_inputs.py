import json
from pathlib import Path
import tempfile
import unittest
from build_inputs import capture_file, outside_release, bound_file
from common import sha256


class InputTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_capture_discovery(self):
        p = self.root / 'seed0_train_capture.pt'; p.write_bytes(b'capture')
        self.assertEqual(capture_file(0, 'train', self.root, None, None), p.resolve())

    def test_ambiguous_capture_is_rejected(self):
        for prefix in ('a', 'b'):
            (self.root / (prefix + '_seed0_train_capture.pt')).write_bytes(b'capture')
        with self.assertRaises(RuntimeError):
            capture_file(0, 'train', self.root, None, None)

    def test_pipeline_capture_layout(self):
        p = self.root / 'seed1/capture/fresh_val_capture.pt'
        p.parent.mkdir(parents=True)
        p.write_bytes(b'capture')
        self.assertEqual(capture_file(1, 'val', self.root, None, None), p.resolve())

    def test_duplicate_pipeline_layout_is_rejected(self):
        p = self.root / 'seed1/capture/fresh_val_capture.pt'
        p.parent.mkdir(parents=True)
        p.write_bytes(b'capture')
        (self.root / 'seed1/val_capture.pt').write_bytes(b'other capture')
        with self.assertRaises(RuntimeError):
            capture_file(1, 'val', self.root, None, None)

    def test_relative_map(self):
        p = self.root / 'capture.pt'; p.write_bytes(b'capture')
        self.assertEqual(capture_file(0, 'train', None,
                                     {'0': {'train': 'capture.pt'}}, self.root), p.resolve())

    def test_release_and_existing_outputs_rejected(self):
        with self.assertRaises(RuntimeError):
            outside_release(self.root / 'inside', self.root)
        p = self.root / 'exists'; p.touch()
        with self.assertRaises(RuntimeError):
            outside_release(p, self.root / 'published')

    def test_wrong_binding_rejected(self):
        p = self.root / 'asset'; p.write_bytes(b'original')
        record = {'path': 'asset', 'sha256': sha256(p)}
        self.assertEqual(bound_file(self.root, self.root, record, 'asset')['sha256'], sha256(p))
        p.write_bytes(b'changed')
        with self.assertRaises(RuntimeError):
            bound_file(self.root, self.root, record, 'asset')


if __name__ == '__main__':
    unittest.main()
