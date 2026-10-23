"""Validate the supported RT workflow's current inputs without GPU or writes."""
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    entry = load(HERE / 'reproduce_second_host.py', '_rt_input_entry')
    cache = {}
    original = entry.digest
    def digest(path):
        path = Path(path).resolve()
        if path not in cache:
            cache[path] = original(path)
        return cache[path]
    entry.digest = digest
    index = json.loads((HERE / 'CURRENT_RESULT_MANIFEST.json').read_text())
    count = 0
    for mode in ('native', 'gccs', 'full'):
        for seed in (0, 1, 2):
            source, checkpoint, config = entry.plan(seed, mode)
            row = index['configurations'][mode][str(seed)]
            if digest(checkpoint) != row['checkpoint_sha256']:
                raise RuntimeError('Current result/checkpoint mismatch')
            for name in ('coco_bbox_eval.pth', 'predictions.json'):
                if not (RELEASE / row['evaluation'] / name).is_file():
                    raise FileNotFoundError('Published evaluation missing: ' + name)
            if not source.is_dir():
                raise FileNotFoundError(source)
            if mode != 'native':
                cdf = Path(config['HiLoRSetCriterion']['ufd_cdf_path'])
                if not cdf.is_file():
                    raise FileNotFoundError('Train difficulty statistics missing: ' + str(cdf))
            count += 1
    pipeline = load(RELEASE / 'reproduce_pipeline.py', '_rt_input_pipeline')
    from argparse import Namespace
    for method in ('native', 'gccs'):
        _, _, data = pipeline.config_for(Namespace(host='rtdetr', dataset='cdw',
            method=method, seed=0, output=Path('/tmp/unused_rt_input_check')))
        if (data['epoches'] != 102 or data['train_dataloader']['batch_size'] != 4
                or data['val_dataloader']['batch_size'] != 8
                or data['use_amp'] is not False or data['use_ema'] is not True):
            raise RuntimeError('Training protocol differs from the published settings')
    print(f'RT INPUTS: PASS ({count} mode/seed cases; Native/GCCS training settings)')


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
