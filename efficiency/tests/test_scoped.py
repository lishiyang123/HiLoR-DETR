import os
import sys
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from scoped_common import expand, trim_retired_branches, assert_tree_equal, parameters, roots
from measure_training import summarize_epochs, timed_epoch
import run_scoped
from measure_paired_latency import balanced_order, identical_detector_config, summarize, MODES


class RepVggBlock(nn.Module):
    def __init__(self, fused=True):
        super().__init__()
        self.conv1 = nn.Linear(3, 3)
        self.conv2 = nn.Linear(3, 3)
        if fused:
            self.conv = nn.Linear(3, 3)

    def forward(self, x):
        return self.conv(x) if hasattr(self, 'conv') else self.conv1(x) + self.conv2(x)


class ScopedTests(unittest.TestCase):
    def test_only_fused_retired_branches_removed(self):
        block = RepVggBlock(); x = torch.randn(1, 3)
        before = block(x); original = parameters(block)
        removed = trim_retired_branches(block)
        assert_tree_equal(before, block(x))
        self.assertEqual(original - parameters(block), sum(r['parameters'] for r in removed))
        self.assertEqual(len(removed), 2)
        self.assertFalse(hasattr(block, 'conv1'))

    def test_unfused_branches_not_removed(self):
        block = RepVggBlock(False)
        self.assertEqual(trim_retired_branches(block), [])
        self.assertTrue(hasattr(block, 'conv1'))

    def test_paths_can_be_relocated(self):
        value = {'path': '${PRIMARY_ROOT}/checkpoints/head.pt'}
        self.assertEqual(expand(value, {'PRIMARY_ROOT': Path('/new/release')})['path'], '/new/release/checkpoints/head.pt')

    def test_unknown_token_fails(self):
        with self.assertRaises(ValueError):
            expand('${UNKNOWN}/file', {})

    def test_training_scope_mean_and_peak(self):
        rows = [{'training_time_s': 10, 'peak_allocated_MiB': 100}, {'training_time_s': 20, 'peak_allocated_MiB': 200}]
        result = summarize_epochs(rows)
        self.assertEqual(result['mean_training_epoch_time_s'], 15)
        self.assertEqual(result['peak_training_allocated_memory_MiB'], 200)
        self.assertFalse(result['validation_included'])
        self.assertFalse(result['checkpoint_writes_included'])

    def test_epoch_wrapper_preserves_result(self):
        rows = []
        args = (False, None, None, None, [1, 2], None, 'cuda', 25)
        with patch('torch.cuda.synchronize'), patch('torch.cuda.reset_peak_memory_stats'), patch('torch.cuda.max_memory_allocated', return_value=1024**2), patch('measure_training.time.perf_counter', side_effect=[10, 15]):
            result = timed_epoch(lambda *a, **k: {'loss': 2}, args, {}, rows)
        self.assertEqual(result, {'loss': 2})
        self.assertEqual(rows[0], {'epoch': 25, 'batches': 2, 'training_time_s': 5, 'peak_allocated_MiB': 1})

    def test_smoke_not_formal_epoch_result(self):
        value = summarize_epochs([{'training_time_s': 1, 'peak_allocated_MiB': 2}], smoke=True)
        self.assertEqual(value['status'], 'SMOKE_ONLY')

    def test_table_does_not_import_old_timings(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'RESULTS.json').write_text(json.dumps({'rows': [{'Model': 'DEIM', 'Params_M': 1, 'Detector_GFLOPs': 2}]}))
            out = root / 'new'; (out / 'deim').mkdir(parents=True)
            (out / 'deim/MEASUREMENT.json').write_text(json.dumps({'status': 'MEASURED', 'model': 'DEIM', 'parameters': {'total_M': 1}, 'latency': {'forward_ms': 10, 'postprocess_ms': 1, 'total_ms': 11}}))
            with patch.object(run_scoped, 'PACKAGE', root):
                run_scoped.export_inference_tables(out, {'complexity_models': [], 'latency_models': ['deim']})
            self.assertIn('11', (out / 'LATENCY_TABLE.csv').read_text())
            self.assertNotIn('FPS', (out / 'LATENCY_TABLE.csv').read_text())

    def test_table_rejects_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / 'RESULTS.json').write_text(json.dumps({'rows': []}))
            out = root / 'new'; (out / 'hilor').mkdir(parents=True)
            (out / 'hilor/MEASUREMENT.json').write_text(json.dumps({'status': 'SMOKE_ONLY'}))
            with patch.object(run_scoped, 'PACKAGE', root), self.assertRaises(RuntimeError):
                run_scoped.export_inference_tables(out, {'complexity_models': ['hilor'], 'latency_models': []})

    def test_paired_order_is_balanced_for_every_image(self):
        for image in range(12):
            orders = [balanced_order(image, repeat) for repeat in range(3)]
            for order in orders:
                self.assertEqual(set(order), set(MODES))
            for position in range(3):
                self.assertEqual({order[position] for order in orders}, set(MODES))

    def test_only_postprocessor_and_output_may_differ(self):
        identical_detector_config({'model': {'layers': 3}, 'PostProcessor': False, 'output_dir': 'a'},
                                  {'model': {'layers': 3}, 'PostProcessor': True, 'output_dir': 'b'})
        with self.assertRaises(RuntimeError):
            identical_detector_config({'model': {'layers': 3}}, {'model': {'layers': 4}})

    def test_paired_summary_does_not_force_forward_equal(self):
        samples = [{'mode': m, 'repeat': 1, 'forward_ms': f, 'postprocess_ms': p,
                    'total_ms': f+p, 'synchronized_wall_ms': f+p+1}
                   for m,f,p in [('deim',10,1),('gca',11,1),('hilor',12,2)]]
        result, diag = summarize(samples, 1)
        self.assertEqual(result['gca']['forward_ms'], 11)
        self.assertEqual(result['hilor']['forward_ms'], 12)
        self.assertEqual(result['hilor']['total_ms'], 14)
        self.assertEqual(diag['gca_minus_hilor_forward_ms'], -1)

    def test_multiprocessing_tmp_directory_stays_short(self):
        with patch.dict(os.environ, {'EFFICIENCY_TMP_ROOT': '/root/autodl-tmp/.eff_tmp'}):
            root = run_scoped.short_tmp_root()
        socket_name = str(root / 'pymp-12345678' / 'listener-12345678')
        self.assertLess(len(os.fsencode(socket_name)), 108)
        self.assertNotIn('efficiency', str(root))

    def test_long_tmp_override_is_rejected(self):
        with patch.dict(os.environ, {'EFFICIENCY_TMP_ROOT': '/tmp/' + 'x' * 100}), self.assertRaises(ValueError):
            run_scoped.short_tmp_root()


if __name__ == '__main__':
    unittest.main()
