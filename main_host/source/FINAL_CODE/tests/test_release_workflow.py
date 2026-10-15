"""CPU-only release entry tests; no detector forward, training or fitting."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[4]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = load(ROOT / 'reproduce_pipeline.py', '_pipeline_test')
        cls.main_entry = load(ROOT / 'main_host/source/FINAL_CODE/reproduce.py', '_entry_test')

    def test_fixed_tuple(self):
        self.assertEqual(self.pipeline.SETTINGS, dict(lambda_reg=.001, head_seed=0, positive_weight=4, tau=0., beta=2., topk=300))

    def test_main_checkpoint_specific_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            checkpoint, mapping, head = base / 'checkpoint', base / 'qccr.json', base / 'head'
            checkpoint.write_bytes(b'new checkpoint')
            head.write_bytes(b'new head')
            sha = self.main_entry.sha256
            parameters = dict(beta0=0., beta_positive=1., beta_negative=1., duplicate_penalty=0.)
            mapping.write_text(json.dumps(dict(source_checkpoint_sha256=sha(checkpoint), parameters=parameters)))
            binding = dict(host='deim', dataset='cdw', detector_seed=1, checkpoint_sha256=sha(checkpoint),
                test_used_for_fitting_or_selection=False, settings=self.pipeline.SETTINGS,
                qccr=dict(path=mapping.name, sha256=sha(mapping)), umr=dict(path=head.name, sha256=sha(head)))
            path = base / 'FIT_BINDING.json'
            profile = dict(dataset='cdw', seed=1, method='full')
            path.write_text(json.dumps(binding))
            actual, files = self.main_entry.external_ccvr(path, checkpoint, profile)
            self.assertEqual(actual, parameters)
            self.assertEqual(files['umr'], head)
            for key, value in [('checkpoint_sha256', '0' * 64), ('test_used_for_fitting_or_selection', True), ('detector_seed', 0)]:
                changed = dict(binding, **{key: value})
                path.write_text(json.dumps(changed))
                with self.assertRaises(RuntimeError):
                    self.main_entry.external_ccvr(path, checkpoint, profile)

    def test_main_training_configs(self):
        for dataset in ('cdw', 'btc'):
            for method in ('native', 'gccs', 'full'):
                args = SimpleNamespace(host='deim', dataset=dataset, method=method, seed=2, output=Path('/tmp/fresh-config-only'))
                _, _, data = self.pipeline.config_for(args)
                self.assertFalse(data['use_amp'])
                self.assertEqual(data['seed'], 2)
                self.assertIn('train', data['train_dataloader']['dataset']['ann_file'])
                self.assertIn('val', data['val_dataloader']['dataset']['ann_file'])
                self.assertNotIn('resume', data)

    def test_rt_training_config_protocol(self):
        for method in ('native', 'gccs', 'full'):
            args = SimpleNamespace(host='rtdetr', dataset='cdw', method=method, seed=1, output=Path('/tmp/fresh-rt-config-only'))
            source, _, data = self.pipeline.config_for(args)
            self.assertEqual(data['epoches'], 102)
            self.assertEqual(data['train_dataloader']['batch_size'], 4)
            self.assertEqual(data['val_dataloader']['batch_size'], 8)
            self.assertFalse(data['use_amp'])
            self.assertTrue(data['use_ema'])
            if method != 'native':
                self.assertEqual(data['HiLoRSetCriterion']['ufd_alpha_u_max'], 0.)
                self.assertEqual(data['HiLoRSetCriterion']['ufd_lambda_max'], .75)
                self.assertEqual(data['HiLoRSetCriterion']['lcct_warmup_epoch'], 20)
                self.assertEqual(data['HiLoRSetCriterion']['lcct_ramp_epochs'], 5)
            else:
                self.assertIn('RT-DETR_official', str(source))

    def test_rt_new_checkpoint_requires_own_fit(self):
        wrapper = self.pipeline.rt_wrapper()
        weight = ROOT / 'second_host/runs/seed0/best_val.pth'
        with self.assertRaises(ValueError):
            wrapper.plan(0, 'full', weight)
        with self.assertRaises(ValueError):
            wrapper.plan(0, 'gccs', None, Path('/tmp/unbound.json'))
        source, actual, data = wrapper.plan(0, 'gccs', weight)
        self.assertEqual(actual, weight.resolve())
        self.assertFalse(data['use_amp'])

    def test_fixed_status_preserves_negative_diagnostic(self):
        source = ROOT / 'main_host/source/FINAL_CODE'
        fixed = self.pipeline.load_module(source / 'tools/calibration/fixed_protocol/fix_qccr.py', '_fixed_status_test')
        freeze = self.pipeline.load_module(source / 'tools/calibration/common/freeze_qccr.py', '_freeze_status_test')
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            parameters = dict(beta0=0., beta_positive=1., beta_negative=1., duplicate_penalty=0.)
            diagnostic = dict(decision='OOF_VAL_NO_GO', deltas_vs_pd=dict(AP=-.001, AP90=-.002), metrics=dict(AP=.7))
            summary = dict(decision='OOF_VAL_NO_GO', candidate_decisions={'fixed': diagnostic})
            fits = dict(candidates={'fixed': dict(lambda_reg=.001, full_val_fit=dict(parameters=parameters))})
            for name, value in [('summary', summary), ('fits', fits), ('policy', dict(qccr_lambda_reg=.001))]:
                (base / (name + '.json')).write_text(json.dumps(value))
            path = base / 'fixed.json'
            self.pipeline.call_main(fixed, ['--summary', base / 'summary.json', '--fits', base / 'fits.json', '--policy', base / 'policy.json', '--output', path])
            result = json.loads(path.read_text())
            self.assertEqual(result['decision'], 'FIXED_PROTOCOL_READY')
            self.assertEqual(result['diagnostic_only_oof'], diagnostic)
            checkpoint = base / 'checkpoint'
            checkpoint.write_bytes(b'new checkpoint')
            arguments = ['--oof-summary', path, '--fits', base / 'fits.json', '--checkpoint', checkpoint,
                         '--expected-checkpoint-sha', self.pipeline.digest(checkpoint), '--dataset-name', 'CDW1630',
                         '--source-method', 'GCCS', '--output', base / 'artifact.json']
            self.pipeline.call_main(freeze, arguments)
            self.assertEqual(json.loads((base / 'artifact.json').read_text())['selection_decision'], 'FIXED_PROTOCOL_READY')
            result['decision'] = 'OOF_VAL_GO'
            path.write_text(json.dumps(result))
            self.pipeline.call_main(freeze, arguments)
            self.assertEqual(json.loads((base / 'artifact.json').read_text())['selection_decision'], 'OOF_VAL_GO')
            path.write_text(json.dumps(summary))
            with self.assertRaises(RuntimeError):
                self.pipeline.call_main(freeze, arguments)


if __name__ == '__main__':
    unittest.main()
