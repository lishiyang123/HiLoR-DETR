"""CPU regressions for current public entry points, not archived run registries."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys
from argparse import Namespace
from unittest.mock import patch

from _bootstrap import ROOT


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pipeline = load(ROOT.parent / 'reproduce_pipeline.py', '_workflow_pipeline')
training = load(ROOT / 'tools/training.py', '_workflow_training')
entry = load(ROOT.parent / 'main_host/source/FINAL_CODE/reproduce.py', '_workflow_main')


class PublicWorkflowTest(unittest.TestCase):
    def test_saved_paths_relocate_when_installed_under_a_different_parent(self):
        wrapper = load(ROOT / 'reproduce_second_host.py', '_relocation_regression')
        with tempfile.TemporaryDirectory() as directory:
            installed = Path(directory) / 'installed-code'
            with patch.object(wrapper, 'RELEASE', installed), patch.object(wrapper, 'HERE', installed / 'second_host'):
                for relative in ('second_host/artifacts/ufd/difficulty_cdf_train640.json',
                                 'main_host/datasets/CDW1630/train2017',
                                 'benchmarks/02_RT_DETR_R18/configs/model.yml'):
                    saved = wrapper.OLD + '/saved_parent/HiLoR-DETR/' + relative
                    self.assertEqual(wrapper.relocate(saved), str(installed / relative))
                    nested = wrapper.OLD + '/author-work/rt/runtime/' + relative
                    self.assertEqual(wrapper.relocate(nested), str(installed / relative))
                self.assertEqual(wrapper.relocate('/another/new/output'), '/another/new/output')
                self.assertEqual(wrapper.relocate('relative/file.json'), 'relative/file.json')

    def test_native_control_uses_native_not_gca_configuration(self):
        args = Namespace(host='deim', dataset='cdw', method='native-ccvr', seed=0,
                         output=Path('/tmp/unused_fresh_control'))
        source, init, config = pipeline.config_for(args, capture=True)
        self.assertFalse(config['DEIMCriterion']['ufd_enabled'])
        self.assertFalse(config['DEIMCriterion']['lcct_enabled'])
        self.assertFalse(config['PostProcessor']['qccr_enabled'])
        self.assertFalse(config['PostProcessor']['umr_enabled'])
        self.assertIn('val2017', config['val_dataloader']['dataset']['ann_file'])

    def test_control_binding_requires_its_own_value_head(self):
        with tempfile.TemporaryDirectory() as value:
            folder = Path(value)
            checkpoint = folder / 'model.pth'; checkpoint.write_bytes(b'checkpoint')
            qccr = folder / 'qccr.json'
            qccr.write_text(json.dumps(dict(source_checkpoint_sha256=entry.sha256(checkpoint), parameters={})))
            binding = dict(host='deim', dataset='cdw', detector_seed=0,
                checkpoint_sha256=entry.sha256(checkpoint), settings=pipeline.SETTINGS,
                test_used_for_fitting_or_selection=False,
                qccr=dict(path=qccr.name, sha256=entry.sha256(qccr)))
            path = folder / 'FIT_BINDING.json';path.write_text(json.dumps(binding))
            profile = dict(dataset='cdw', seed=0, method='deim_ccvr')
            with self.assertRaises(KeyError):
                entry.external_ccvr(path, checkpoint, profile)
            head = folder / 'value.pt';head.write_bytes(b'head')
            binding['umr'] = dict(path=head.name, sha256=entry.sha256(head))
            path.write_text(json.dumps(binding))
            _, paths = entry.external_ccvr(path, checkpoint, profile)
            self.assertEqual(paths['umr'], head)
            binding['checkpoint_sha256'] = '0' * 64
            path.write_text(json.dumps(binding))
            with self.assertRaisesRegex(RuntimeError, 'another detector'):
                entry.external_ccvr(path, checkpoint, profile)

    def test_val_tracker_retains_earlier_exact_tie(self):
        tracker = training.BestValTracker()
        self.assertTrue(tracker.consider(3, .7))
        self.assertFalse(tracker.consider(4, .7))
        self.assertEqual(tracker.epoch, 3)
        with self.assertRaises(training.TrainingGuardError):
            tracker.consider(5, float('nan'))


if __name__ == '__main__':
    unittest.main()
