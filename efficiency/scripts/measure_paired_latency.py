"""Balanced, within-process latency comparison; no profiling or model changes."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

from scoped_common import (
    fp32_tree, load_detector, make_cfg, parameters, sha256, thread_setup,
    trim_retired_branches, write_json,
)
from measure_scoped import check_output

MODES = ('deim', 'gccs', 'hilor')
ORDERS = tuple(itertools.permutations(MODES))


def balanced_order(image_index, repeat_index):
    # Each mode occupies each position equally over six consecutive images.
    # All modes are run for every image; no candidate-ranking result is selected.
    base = ORDERS[image_index % len(ORDERS)]
    shift = repeat_index % len(MODES)
    return base[shift:] + base[:shift]


def identical_detector_config(native, full):
    def model_part(cfg):
        return {key: value for key, value in cfg.items() if key not in ('PostProcessor', 'output_dir')}
    if model_part(native) != model_part(full):
        raise RuntimeError('GCCS and full HiLoR do not share the same detector configuration')


def summarize(samples, repeats):
    result = {}
    for mode in MODES:
        values = [row for row in samples if row['mode'] == mode]
        means = [statistics.mean(row['total_ms'] for row in values if row['repeat'] == r + 1)
                 for r in range(repeats)]
        cv = 100 * statistics.stdev(means) / statistics.mean(means) if repeats > 1 else 0.0
        result[mode] = {key: statistics.mean(row[key] for row in values)
                        for key in ('forward_ms', 'postprocess_ms', 'total_ms', 'synchronized_wall_ms')}
        result[mode].update({'repeat_mean_total_ms': means, 'repeat_latency_cv_percent': cv})
    a, b = result['gccs']['forward_ms'], result['hilor']['forward_ms']
    diagnostic = {'gccs_minus_hilor_forward_ms': a - b,
                  'absolute_relative_forward_gap_percent': 100 * abs(a - b) / ((a + b) / 2),
                  'role': 'measurement diagnostic, not an algorithm speedup or accuracy selection'}
    return result, diagnostic


def gpu_state():
    command = ['nvidia-smi', '--query-gpu=name,utilization.gpu,clocks.current.graphics,clocks.current.memory,power.draw,temperature.gpu,pstate', '--format=csv,noheader']
    try:
        return subprocess.check_output(command, text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return 'unavailable'


def environment_record(torch):
    """Record the effective device and precision settings outside timed regions."""
    cpu = next((line.split(':', 1)[1].strip()
                for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')), 'unavailable')
    device = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=driver_version,uuid,power.limit',
         '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
    cgroup = {}
    for name in ('cpu.max', 'memory.max'):
        path = Path('/sys/fs/cgroup') / name
        cgroup[name] = path.read_text().strip() if path.exists() else 'unavailable'
    return {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'python': platform.python_version(), 'os': platform.platform(),
            'cpu': cpu, 'cgroup_limits': cgroup,
            'gpu_driver_uuid_power_limit_w': device,
            'torch_num_threads': torch.get_num_threads(),
            'torch_num_interop_threads': torch.get_num_interop_threads(),
            'thread_environment': {key: os.environ.get(key) for key in
                ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic}


def main(args):
    import torch
    from benchmark_runtime import load_test_index, prepare_image
    thread_setup()
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Paired inference timing requires RTX 4090')
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'PAIRED_LATENCY.json').exists() or any((args.output / m).exists() for m in MODES):
        raise RuntimeError('Existing latency measurements are not overwritten')
    configs, rows, models, posts, weights = {}, {}, {}, {}, {}
    for name in ('deim', 'gccs'):
        cfg, row, bound, config = make_cfg(name, args.asset_root, args.output / 'effective_configs')
        detector, post, weight_source = load_detector(cfg, row)
        detector = detector.deploy().float().eval()
        trim_retired_branches(detector)
        models[name], posts[name] = detector.cuda().eval(), post.cuda().eval()
        configs[name], rows[name], weights[name] = config, row, weight_source
    full_cfg, full_row, full_bound, full_config = make_cfg('hilor', args.asset_root, args.output / 'effective_configs')
    identical_detector_config(json.loads((Path(__file__).resolve().parents[1] / 'configs/templates/gccs.json').read_text()),
                              json.loads((Path(__file__).resolve().parents[1] / 'configs/templates/hilor.json').read_text()))
    if rows['gccs']['checkpoint_sha256'] != full_row['checkpoint_sha256'] or rows['gccs']['source'] != full_row['source']:
        raise RuntimeError('GCCS and HiLoR must use the identical checkpoint and source')
    if sha256(full_row['head']['path']) != full_row['head']['sha256']:
        raise RuntimeError('Frozen MLP value-head SHA mismatch')
    post = full_cfg.postprocessor.float().eval().cuda()
    if not post.qccr_enabled or not post.umr_enabled or parameters(post.umr.head) != 14729:
        raise RuntimeError('Full HiLoR must use the bound formal MLP QCCR–UMR pipeline')
    if (post.umr.topk, post.umr.tau, post.umr.beta) != (300, 0.0, 2.0):
        raise RuntimeError('Frozen UMR tuple differs')
    # One physical detector instance serves both modes. Only postprocessing differs.
    models['hilor'], posts['hilor'] = models['gccs'], post
    assert models['gccs'] is models['hilor']
    configs['hilor'], rows['hilor'], weights['hilor'] = full_config, full_row, weights['gccs']
    thread_setup()  # Reassert after detector-source imports, identically for all modes.
    # These are the main-host FP32 evaluation environment's backend settings.
    # cuDNN may use TF32 internally for float32 convolutions; record this explicitly.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    environment = environment_record(torch)
    index = load_test_index(Path(bound['data']['annotation']), Path(bound['data']['root']) / 'test2017')
    if sha256(bound['data']['annotation']) != bound['data']['test_annotation_sha256']:
        raise RuntimeError('Test annotation differs from the existing binding')
    smoke = args.task == 'smoke'
    warmup, repeats = (3, 1) if smoke else (50, 3)
    needed = index[:8] if smoke else index
    # All GPU inputs are ready before any timed interval. No IO or transfer occurs
    # between the three variants of an image; prepare_image preserves the old norm.
    prepared = []
    for image_row in needed:
        cpu, hw = prepare_image(image_row)
        prepared.append((image_row, cpu.cuda(), torch.tensor([list(hw)], device='cuda', dtype=torch.int64)))
    events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(3))

    def one(mode, prepared_image, timed):
        image_row, image, sizes = prepared_image
        torch.cuda.synchronize()
        start, middle, end = events
        wall_start = time.perf_counter()
        start.record()
        with torch.cuda.amp.autocast(enabled=False):
            raw = models[mode](image)
        middle.record()
        with torch.cuda.amp.autocast(enabled=False):
            result = posts[mode](fp32_tree(raw), sizes)
        end.record()
        end.synchronize()
        wall_ms = 1000 * (time.perf_counter() - wall_start)
        check_output(raw, result)  # Output validation is outside both timing scopes.
        if image.dtype != torch.float32 or any(
                value.is_floating_point() and value.dtype != torch.float32
                for value in raw.values() if isinstance(value, torch.Tensor)):
            raise RuntimeError('FP32 timing requires float32 inputs and detector outputs')
        if not timed:
            return None
        f, p, t = start.elapsed_time(middle), middle.elapsed_time(end), start.elapsed_time(end)
        if abs(f + p - t) > max(0.05, t * 0.01):
            raise RuntimeError('CUDA timing partition differs')
        return {'mode': mode, 'image_id': image_row['image_id'], 'forward_ms': f,
                'postprocess_ms': p, 'total_ms': t, 'synchronized_wall_ms': wall_ms}

    snapshots = {'before_warmup': gpu_state()}
    samples = []
    with torch.inference_mode():
        for i, item in enumerate(prepared[:warmup]):
            for mode in balanced_order(i, 0):
                one(mode, item, False)
        snapshots['after_warmup'] = gpu_state()
        for repeat in range(repeats):
            for i, item in enumerate(prepared):
                order = balanced_order(i, repeat)
                for position, mode in enumerate(order):
                    row = one(mode, item, True)
                    row.update({'repeat': repeat + 1, 'order_position': position + 1})
                    samples.append(row)
            snapshots['after_repeat_' + str(repeat + 1)] = gpu_state()
            print(f'Paired latency: repeat {repeat+1}/{repeats} complete, all three modes', flush=True)
    summary, forward_diagnostic = summarize(samples, repeats)
    stable = all(value['repeat_latency_cv_percent'] <= 5 for value in summary.values())
    comparable_forward = forward_diagnostic['absolute_relative_forward_gap_percent'] <= 5
    status = 'SMOKE_ONLY' if smoke else ('MEASURED' if stable and comparable_forward else 'MEASUREMENT_REVIEW_REQUIRED')
    protocol = {'schema': 'balanced-within-process-primary-latency-v1',
                'warmup_per_mode': warmup, 'images_per_repeat': len(prepared), 'repeats': repeats,
                'timer': 'CUDA events; synchronized wall time retained as a separate diagnostic',
                'precision': 'FP32 detector and complete postprocessing; autocast disabled',
                'matmul_allow_tf32': False, 'cudnn_allow_tf32': True,
                'input_shape': [1, 3, 640, 640],
                'order': 'six balanced permutations across real-image triples',
                'shared_gccs_hilor_detector_instance': True, 'flops_profiling_in_timing_process': False,
                'preloaded_gpu_inputs': True, 'postprocessor_implementation': 'existing optimized source; unchanged',
                'scope': 'forward + FP32 conversion + complete postprocessing + box scaling; IO/preprocess/H2D/evaluation excluded',
                'repeat_cv_limit_percent': 5, 'same_detector_relative_forward_gap_review_threshold_percent': 5,
                'reported_values': 'independent full-pipeline samples for each mode; no substituted/averaged shared forward value'}
    report = {'status': status, 'protocol': protocol, 'summary': summary,
              'forward_diagnostic': forward_diagnostic, 'gpu_state_snapshots': snapshots,
              'environment': environment,
              'measurement_script_sha256': sha256(__file__),
              'accuracy_evaluated': False, 'training_executed': False}
    write_json(args.output / 'PAIRED_LATENCY.json', report)
    with (args.output / 'PAIRED_LATENCY_SAMPLES.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
        writer.writeheader(); writer.writerows(samples)
    table = []
    for mode in MODES:
        detector_ids = {id(p) for p in models[mode].parameters()}
        pp = sum(p.numel() for p in posts[mode].parameters() if id(p) not in detector_ids)
        record = {'status': status, 'model': rows[mode]['label'], 'seed': 0,
                  'checkpoint_sha256': rows[mode]['checkpoint_sha256'], 'source': rows[mode]['source'],
                  'effective_config_sha256': sha256(configs[mode]), 'weight_source': weights[mode],
                  'parameters': {'detector': parameters(models[mode]), 'postprocessor': pp,
                                 'fitted_scalars': rows[mode]['fitted_scalars'],
                                 'total_M': (parameters(models[mode]) + pp + rows[mode]['fitted_scalars']) / 1e6},
                  'latency': summary[mode], 'protocol': protocol, 'environment': report['environment']}
        write_json(args.output / mode / 'MEASUREMENT.json', record)
        table.append({'Model': rows[mode]['label'], 'Forward_ms': summary[mode]['forward_ms'],
                      'Postprocess_ms': summary[mode]['postprocess_ms'], 'Total_ms': summary[mode]['total_ms']})
    with (args.output / 'LATENCY_TABLE.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    for row in table:
        print(row, flush=True)
    print('Same-detector forward diagnostic:', forward_diagnostic, '\nStatus:', status, flush=True)
    if not smoke and status != 'MEASURED':
        raise RuntimeError('All results saved; unstable/comparison-inconsistent timing requires review, not forced correction')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=['smoke', 'formal'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--asset-root')
    main(parser.parse_args())
