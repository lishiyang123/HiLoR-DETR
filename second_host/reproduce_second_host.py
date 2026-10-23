#!/usr/bin/env python3
"""Portable Test entry for evaluating saved RT-DETR checkpoints.

Uses the released YAML settings and FP32/EMA inference. Bbox evaluation defaults
to faster-coco-eval 1.7.2, also used by the training/Val pipeline. An explicit
pycocotools backend is available for backend comparisons. New evaluations write
to a separate output directory and never overwrite published results.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import sys

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent


def merge(base, child):
    for key, value in child.items():
        if key == '__include__':
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def yaml_closure(path):
    import yaml
    data = yaml.safe_load(path.read_text()) or {}
    result = {}
    for inc in data.get('__include__', []):
        p = Path(inc)
        p = Path(relocate(str(p)))
        merge(result, yaml_closure(p if p.is_absolute() else path.parent / p))
    return merge(result, data)


def relocate(value):
    if isinstance(value, dict):
        return {k: relocate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v) for v in value]
    if isinstance(value, str):
        return value.replace('${RELEASE_ROOT}', str(RELEASE))
    return value


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def plan(seed, mode, new_checkpoint=None, new_binding=None):
    sota = RELEASE / 'benchmarks/02_RT_DETR_R18'
    if mode == 'native':
        source = sota / 'source/RT-DETR_official/rtdetr_pytorch'
        config = sota / 'configs/cdw1630_rtdetr_r18_frozen_test_v1.yml'
        checkpoint = sota / ('runs/seed%d/best_val.pth' % seed)
    else:
        source = HERE / 'source/RT-DETR_HiLoR/rtdetr_pytorch'
        config = HERE / ('configs/hilor/generated/seed%d/qccr_umr.yml' % seed) if mode == 'full' else HERE / 'configs/hilor/cdw1630_rtdetr_r18_gsa.yml'
        checkpoint = HERE / ('runs/seed%d/best_val.pth' % seed)
    if new_checkpoint is not None:
        checkpoint = new_checkpoint.resolve(strict=True)
    else:
        saved = {row['path']: row for row in json.loads((RELEASE / 'RELEASE_MANIFEST.sha256.json').read_text())['files']}
        relative = checkpoint.relative_to(RELEASE).as_posix()
        if digest(checkpoint) != saved[relative]['sha256']:
            raise RuntimeError('Checkpoint SHA mismatch')
    if new_binding is not None and (new_checkpoint is None or mode != 'full'):
        raise ValueError('New binding requires --checkpoint and --mode full')
    if new_checkpoint is not None and mode == 'full' and new_binding is None:
        raise ValueError('New full-method evaluation requires checkpoint-specific --binding')
    data = relocate(yaml_closure(config))
    primary = RELEASE / 'main_host/datasets/CDW1630'
    loader = data['val_dataloader']
    loader['dataset']['img_folder'] = str(primary / 'test2017')
    loader['dataset']['ann_file'] = str(primary / 'annotations/instances_test2017.json')
    assert loader['batch_size'] == 8 and not loader['shuffle']
    assert data['use_amp'] is False and data['use_ema'] is True
    if mode == 'full':
        binding = new_binding.resolve(strict=True) if new_binding else HERE / ('artifacts/inference/seed%d/qccr_umr_binding.json' % seed)
        spec = data['HiLoRRTDETRPostProcessor']['artifact_manifest']
        if new_binding:
            spec.update(path=str(binding), sha256=digest(binding))
        if digest(binding) != spec['sha256']:
            raise RuntimeError('Inference binding SHA mismatch')
        document = json.loads(binding.read_text())
        if document['detector_checkpoint_sha256'] != digest(checkpoint) or document['detector_seed'] != seed:
            raise RuntimeError('Inference binding belongs to another checkpoint/seed')
        if new_binding is not None:
            data['HiLoRRTDETRPostProcessor']['expected_detector_checkpoint_sha256'] = document['detector_checkpoint_sha256']
    return source, checkpoint, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--mode', choices=('native', 'gsa', 'full'), required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--checkpoint', type=Path, help='New checkpoint; default replays paper weights')
    parser.add_argument('--binding', type=Path, help='Checkpoint-specific newly fitted RT CCVR binding')
    parser.add_argument('--evaluator', choices=('faster-coco-eval', 'pycocotools'),
                        default='faster-coco-eval', help='Test backend only; never changes training or Val')
    args = parser.parse_args()
    source, checkpoint, data = plan(args.seed, args.mode, args.checkpoint, args.binding)
    if args.preflight_only:
        print('RELOCATED RT-DETR CONFIG/INPUT PREFLIGHT: PASS (no inference)')
        return
    if args.output is None:
        parser.error('--output required for GPU evaluation')
    output = args.output.resolve()
    if output.exists() or output == RELEASE or RELEASE in output.parents:
        raise ValueError('Use a new output directory outside this frozen release')
    # Deterministic Test execution; training and fitting settings are unchanged.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['OMP_NUM_THREADS'] = '1'
    import torch
    import yaml
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for checkpoint evaluation; use --preflight-only')
    output.mkdir(parents=True)
    data['output_dir'] = str(output)
    config = output / 'relocated_config.yml'
    config.write_text(yaml.safe_dump(data, sort_keys=False))
    sys.path.insert(0, str(source))
    from src.core import YAMLConfig
    from src.data import get_coco_api_from_dataset
    from src.misc import dist
    from src.solver import TASKS
    from src.solver import det_engine
    if args.evaluator == 'faster-coco-eval':
        sys.path.insert(0, str(RELEASE))
        from coco_test_evaluation import faster_bbox_evaluator, save_predictions
        det_engine.CocoEvaluator = faster_bbox_evaluator(det_engine.CocoEvaluator)
    evaluate = det_engine.evaluate
    os.chdir(HERE)
    dist.set_seed(args.seed)
    cfg = YAMLConfig(str(config), resume=str(checkpoint), tuning=None,
                     use_amp=False, output_dir=str(output))
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.eval()
    solver.postprocessor.to(solver.device)
    base_ds = get_coco_api_from_dataset(solver.val_dataloader.dataset)
    model = solver.ema.module if solver.ema else solver.model
    stats, evaluator = evaluate(model, solver.criterion, solver.postprocessor,
        solver.val_dataloader, base_ds, solver.device, output)
    (output / 'METRICS.json').write_text(json.dumps(stats, indent=2) + '\n')
    torch.save(evaluator.coco_eval['bbox'].eval, output / 'coco_bbox_eval.pth')
    import importlib.metadata
    metadata = dict(evaluator=args.evaluator, evaluator_version=importlib.metadata.version(args.evaluator),
                    checkpoint_sha256=digest(checkpoint), seed=args.seed, mode=args.mode,
                    weights='ema.module', precision='FP32', evaluation_batch=8,
                    training_or_fitting_changed=False, archived_results_overwritten=False)
    metadata['runtime'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(solver.device),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'])
    metadata['runtime']['cpu_threads'] = torch.get_num_threads()
    if args.evaluator == 'faster-coco-eval':
        metadata['predictions_sha256'] = digest(save_predictions(evaluator, output))
    (output / 'EVALUATION_BACKEND.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('RELOCATED CHECKPOINT EVALUATION: COMPLETE; archived results unchanged')


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
