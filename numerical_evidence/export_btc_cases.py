"""Export a few qualitative candidates from released BTC seed-0 checkpoints.

This author-side helper never changes release files or computes aggregate AP.
CPU preview inference is recorded explicitly; it is not an old prediction cache.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=['deim', 'full'], required=True)
    parser.add_argument('--split', choices=['test1', 'test2'], required=True)
    parser.add_argument('--files', nargs='+', required=True)
    args = parser.parse_args()
    release, output = args.release.resolve(), args.output.resolve()
    if release == output or release in output.parents:
        raise RuntimeError('Use an author directory outside the release')
    output.mkdir(parents=True, exist_ok=False)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    source = release / 'main_host/source/FINAL_CODE'
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('published_entry', source / 'reproduce.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    import yaml
    import torch
    from engine.core import YAMLConfig
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    key = f'btc_{args.method}_seed0_{args.split}'
    profile = json.loads((source / 'PROFILES.json').read_text())[key]
    checkpoint = release / profile['checkpoint']
    if digest(checkpoint) != profile['checkpoint_sha256']:
        raise RuntimeError('Checkpoint differs from released profile')
    for asset in profile.get('inference_assets', []):
        if digest(release / asset['path']) != asset['sha256']:
            raise RuntimeError('Fitted asset differs from released profile')
    data = entry.expand(yaml.safe_load((source / profile['config']).read_text()), release, source)
    data.update(device='cpu', use_amp=False, seed=0)
    data['val_dataloader']['num_workers'] = 0
    config = output / 'runtime.yml'
    config.write_text(yaml.safe_dump(data, sort_keys=False))
    cfg = YAMLConfig(str(config))
    model = cfg.model.cpu().eval()
    weights = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    state = weights['ema']['module']
    model.load_state_dict(state, strict=True)
    del state, weights
    post = cfg.postprocessor.cpu().eval()
    dataset = cfg.val_dataloader.dataset
    annotations = json.loads(Path(data['val_dataloader']['dataset']['ann_file']).read_text())
    by_file = {im['file_name']: im for im in annotations['images']}
    by_id = {image_id: i for i, image_id in enumerate(dataset.ids)}
    binding = dict(profile=key, method=args.method, detector_seed=0,
                   checkpoint_sha256=profile['checkpoint_sha256'],
                   fitted_assets=profile.get('inference_assets', []),
                   source=str(source), device='cpu', precision='FP32', batch_size=1,
                   input_size=[640, 640], topk=300, filtering_before_export=False,
                   role='post-hoc qualitative preview from released weights',
                   model_or_parameter_selection=False, formal_metrics_modified=False)
    (output / 'BINDING.json').write_text(json.dumps(binding, indent=2))
    for name in args.files:
        im = by_file[name]
        image_path = Path(data['val_dataloader']['dataset']['img_folder']) / name
        tensor, target = dataset[by_id[im['id']]]
        start = time.monotonic()
        with torch.inference_mode():
            raw = model(tensor.unsqueeze(0))
            prediction = post(raw, target['orig_size'].unsqueeze(0))[0]
        rows = []
        for box, score, label in zip(prediction['boxes'].tolist(), prediction['scores'].tolist(), prediction['labels'].tolist()):
            x1, y1, x2, y2 = box
            rows.append(dict(image_id=im['id'], category_id=label,
                             bbox=[x1, y1, x2-x1, y2-y1], score=score))
        payload = dict(image=im, image_sha256=digest(image_path), detections=rows,
                       ground_truth=[a for a in annotations['annotations'] if a['image_id'] == im['id']],
                       categories=annotations['categories'], binding=binding)
        (output / (Path(name).stem + '.json')).write_text(json.dumps(payload, indent=2))
        print(f"EXPORTED {key} {name}: {len(rows)} detections, {time.monotonic()-start:.1f}s", flush=True)


if __name__ == '__main__':
    main()
