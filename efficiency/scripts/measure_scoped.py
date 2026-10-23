"""One-process-per-detector CPU check, deployment complexity or real-Test timing."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from scoped_common import (
    PACKAGE, assert_tree_equal, bindings, fp32_tree, load_detector, make_cfg,
    parameters, refresh_manifest, roots, sha256, thread_setup, trim_retired_branches, write_json,
)


def sample_image(data_root, split='val'):
    import numpy as np
    import torch
    from PIL import Image
    ann = json.loads((Path(data_root) / 'annotations' / ('instances_' + split + '2017.json')).read_text())
    row = sorted(ann['images'], key=lambda x: int(x['id']))[0]
    with Image.open(Path(data_root) / (split + '2017') / row['file_name']) as handle:
        image = handle.convert('RGB')
        if image.size != (640, 640):
            raise RuntimeError('Expected an already resized 640x640 image')
        array = np.asarray(image).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div(255).unsqueeze(0)


def check_output(raw, result):
    import torch
    assert tuple(raw['pred_logits'].shape) == (1, 300, 5)
    assert tuple(raw['pred_boxes'].shape) == (1, 300, 4)
    assert len(result) == 1
    assert tuple(result[0]['scores'].shape) == (300,)
    for key in ('scores', 'boxes'):
        assert torch.isfinite(result[0][key]).all(), 'Nonfinite postprocessor output'


def cpu_check(args):
    import torch
    cfg, row, bound, config = make_cfg(args.model, args.asset_root, args.output / 'effective_configs')
    detector, post, weight_source = load_detector(cfg, row)
    image = sample_image(bound['data']['root'])
    with torch.inference_mode():
        detector = detector.deploy().float().eval()
        before = detector(image)
        removed = trim_retired_branches(detector)
        after = detector(image)
        assert_tree_equal(before, after)
        result = post(fp32_tree(after), torch.tensor([[640, 640]]))
        check_output(after, result)
    ids = {id(p) for p in detector.parameters()}
    pp = sum(p.numel() for p in post.parameters() if id(p) not in ids)
    # Check train-only criterion construction without allocating training activations.
    training_constructor = None
    if args.model in ('deim', 'gca'):
        training_constructor = type(cfg.criterion).__name__
    from calflops import calculate_flops  # Import check; no CPU latency is reported.
    report = {'status': 'CPU_PASS_GPU_PENDING', 'model': row['label'],
              'weight_source': weight_source, 'strict_checkpoint_load': True,
              'checkpoint_sha256': row['checkpoint_sha256'], 'config_sha256': sha256(config),
              'detector_parameters': parameters(detector), 'postprocessor_parameters': pp,
              'fitted_scalars': row['fitted_scalars'], 'retired_branch_removal_bit_exact': True,
              'retired_branches': removed, 'real_val_image_forward_and_top300': 'PASS',
              'criterion_construction': training_constructor, 'cuda_checked': False,
              'test_inference_run': False}
    write_json(args.output / (args.model + '.json'), report)
    print(f"CPU CHECK {args.model}: PASS; strict weights, deployed forward and postprocessor; GPU pending", flush=True)


def gpu_measure(args):
    import torch
    from benchmark_runtime import load_test_index, prepare_image, profile_flops
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('This protocol requires an available RTX 4090')
    cfg, row, bound, config = make_cfg(args.model, args.asset_root, args.output / 'effective_configs')
    detector, post, weight_source = load_detector(cfg, row)
    detector = detector.deploy().float().eval()
    removed = trim_retired_branches(detector)
    detector = detector.cuda().eval()
    post = post.cuda().eval()  # Includes UMR MLP; no CPU/CUDA split.
    ids = {id(p) for p in detector.parameters()}
    pp = sum(p.numel() for p in post.parameters() if id(p) not in ids)
    output = args.output / args.model
    if output.exists():
        raise RuntimeError('Refusing to overwrite a measured run: ' + str(output))
    output.mkdir(parents=True)
    result = {'status': 'SMOKE_ONLY' if args.task == 'smoke' else 'MEASURED',
              'model': row['label'], 'seed': 0, 'checkpoint_sha256': row['checkpoint_sha256'],
              'source': row['source'], 'effective_config_sha256': sha256(config),
              'weight_source': weight_source, 'input_shape': [1, 3, 640, 640],
              'environment': {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda},
              'parameters': {'detector': parameters(detector), 'postprocessor': pp,
                             'fitted_scalars': row['fitted_scalars'],
                             'total_M': (parameters(detector) + pp + row['fitted_scalars']) / 1e6,
                             'scope': 'registered parameters of official deploy graph after retired RepVGG branches and UFD head removal'},
              'removed_retired_branches': removed}
    if args.model in bound['complexity_models']:
        with torch.inference_mode():
            result['flops'] = profile_flops(detector)
        assert result['flops']['postprocessor_included'] is False
        if result['flops']['profiler_version'] != '0.3.2':
            raise RuntimeError('Expected calflops 0.3.2')
    if args.task == 'smoke':
        # Both R18 configurations must execute their complete postprocessor too,
        # although only their complexity is requested in the formal experiment.
        image = sample_image(bound['data']['root']).cuda()
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=False):
            raw = detector(image)
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=False):
            prediction = post(fp32_tree(raw), torch.tensor([[640, 640]], device='cuda'))
        check_output(raw, prediction)
        result['gpu_forward_and_full_postprocessor_smoke'] = 'PASS'
    if args.model in bound['latency_models']:
        rows = load_test_index(Path(bound['data']['annotation']), Path(bound['data']['root']) / 'test2017')
        if sha256(bound['data']['annotation']) != bound['data']['test_annotation_sha256']:
            raise RuntimeError('Test annotation SHA differs from prepared inputs')
        warmup, repeats = (3, 1) if args.task == 'smoke' else (50, 3)
        timed_rows = rows[:8] if args.task == 'smoke' else rows
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        samples = []
        repeat_means = []
        def one(image_row, timed):
            cpu, hw = prepare_image(image_row)
            image = cpu.cuda()
            sizes = torch.tensor([list(hw)], dtype=torch.int64, device='cuda')
            torch.cuda.synchronize()
            start, middle, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            start.record()
            with torch.cuda.amp.autocast(enabled=False):
                raw = detector(image)
            middle.record()
            with torch.cuda.amp.autocast(enabled=False):
                prediction = post(fp32_tree(raw), sizes)
            end.record()
            end.synchronize()
            check_output(raw, prediction)
            if not timed:
                return None
            f, p, t = start.elapsed_time(middle), middle.elapsed_time(end), start.elapsed_time(end)
            assert abs(f + p - t) <= max(0.05, t * 0.01)
            return {'image_id': image_row['image_id'], 'forward_ms': f, 'postprocess_ms': p, 'total_ms': t}
        with torch.inference_mode():
            for r in rows[:warmup]:
                one(r, False)
            for repeat in range(repeats):
                values = []
                for r in timed_rows:
                    sample = one(r, True)
                    sample['repeat'] = repeat + 1
                    samples.append(sample)
                    values.append(sample['total_ms'])
                repeat_means.append(statistics.mean(values))
                print(f"{args.model}: repeat {repeat+1}/{repeats} complete", flush=True)
        cv = 100 * statistics.stdev(repeat_means) / statistics.mean(repeat_means) if repeats > 1 else 0
        result['latency'] = {k: statistics.mean(s[k] for s in samples) for k in ('forward_ms', 'postprocess_ms', 'total_ms')}
        result['latency'].update({'warmup': warmup, 'images_per_repeat': len(timed_rows), 'repeats': repeats,
                                  'repeat_mean_total_ms': repeat_means, 'repeat_latency_cv_percent': cv,
                                  'precision': 'FP32 detector and complete postprocessing; autocast disabled',
                                  'matmul_allow_tf32': False, 'cudnn_allow_tf32': True,
                                  'role': 'individual diagnostic; use --stage latency for the paired publication table',
                                  'scope': 'forward + FP32 conversion + complete postprocessing + box scaling; IO/preprocess/H2D/evaluation excluded'})
        if args.task != 'smoke' and cv > 5:
            result['status'] = 'UNSTABLE_REMEASURE'
        with (output / 'LATENCY_SAMPLES.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
            writer.writeheader()
            writer.writerows(samples)
    write_json(output / 'MEASUREMENT.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result['status'] == 'UNSTABLE_REMEASURE':
        raise RuntimeError('Repeat latency CV exceeds 5%; measurements saved but not accepted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['deim', 'gca', 'hilor', 'r18', 'r18_hilor'])
    parser.add_argument('--task', required=True, choices=['cpu-check', 'smoke', 'formal'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--asset-root')
    args = parser.parse_args()
    thread_setup()
    if args.task == 'cpu-check':
        cpu_check(args)
    else:
        gpu_measure(args)
