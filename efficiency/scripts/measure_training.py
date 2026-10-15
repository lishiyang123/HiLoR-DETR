"""Time the unmodified training engine; keep all new runs outside formal assets."""
from __future__ import annotations

import argparse
import csv
import itertools
import random
import statistics
import time
from pathlib import Path

from scoped_common import make_cfg, sha256, thread_setup, write_json


def summarize_epochs(rows, smoke=False):
    return {
        'status': 'SMOKE_ONLY' if smoke else 'MEASURED',
        'completed_training_epochs': len(rows),
        'mean_training_epoch_time_s': statistics.mean(r['training_time_s'] for r in rows),
        'peak_training_allocated_memory_MiB': max(r['peak_allocated_MiB'] for r in rows),
        'validation_included': False, 'checkpoint_writes_included': False,
        'data_loading_and_transfer_included': True,
        'scope': 'complete train_one_epoch wall time; forward/loss/backward/update/EMA/logging included',
    }


def timed_epoch(original, args, kwargs, rows):
    import torch
    epoch = args[7] if len(args) > 7 else kwargs['epoch']
    loader = args[4] if len(args) > 4 else kwargs['data_loader']
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = original(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    rows.append({'epoch': int(epoch), 'batches': len(loader), 'training_time_s': elapsed,
                 'peak_allocated_MiB': torch.cuda.max_memory_allocated() / 1024**2})
    print(f'TRAINING COST epoch={epoch} time={elapsed:.3f}s peak={rows[-1]["peak_allocated_MiB"]:.2f}MiB', flush=True)
    return result


class LimitedLoader:
    """Only used by a clearly labelled smoke test, never for an epoch statistic."""
    def __init__(self, loader, batches=3):
        self.loader, self.batches = loader, batches

    def __iter__(self):
        return itertools.islice(iter(self.loader), self.batches)

    def __len__(self):
        return min(self.batches, len(self.loader))


def cpu_check(args):
    import torch
    cfg, row, bound, config = make_cfg(args.model, args.asset_root, args.output / 'effective_configs')
    model, criterion, optimizer = cfg.model, cfg.criterion, cfg.optimizer
    train, val = cfg.train_dataloader, cfg.val_dataloader
    assert cfg.use_amp is False and cfg.use_ema is True and cfg.epoches == 102
    assert train.batch_size == 8 and val.batch_size == 8
    assert len(train.dataset) == 1144 and len(val.dataset) == 249
    assert train.num_workers == 4 and val.num_workers == 4
    assert sha256(cfg.tuning) == bound['training']['initialization_sha256']
    # Exercise the real augmentation pipeline once without a full CPU B=8 backward.
    train.set_epoch(25)
    image, target = train.dataset[0]
    assert torch.isfinite(image).all() and torch.isfinite(target['boxes']).all()
    write_json(args.output / (args.model + '.json'), {
        'status': 'CPU_PASS_GPU_PENDING', 'criterion': type(criterion).__name__,
        'optimizer': type(optimizer).__name__, 'train_images': len(train.dataset),
        'val_images': len(val.dataset), 'batch_size': train.batch_size, 'precision': 'FP32',
        'real_training_augmentation': 'PASS', 'full_backward_checked': False,
    })
    print(f'TRAINING CPU CHECK {args.model}: PASS; full CUDA backward pending', flush=True)


def gpu_training(args):
    import numpy as np
    import torch
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Training measurement requires RTX 4090')
    output = args.output / args.model
    if output.exists():
        raise RuntimeError('Refusing to overwrite a training measurement run')
    output.mkdir(parents=True)
    cfg, row, bound, config = make_cfg(args.model, args.asset_root, output / 'effective_config')
    assert sha256(cfg.tuning) == bound['training']['initialization_sha256']
    cfg.output_dir = str(output / 'training_run')
    cfg.summary_dir = None
    cfg.device = 'cuda'
    # Disable the same optional TensorBoard writer for both measured runs.
    class TimingConfig(type(cfg)):
        @property
        def writer(self):
            return None
    cfg.__class__ = TimingConfig
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    from engine.solver import TASKS
    from engine.solver import det_solver
    original = det_solver.train_one_epoch
    rows = []
    def measured(*values, **kw):
        result = timed_epoch(original, values, kw, rows)
        write_json(output / 'EPOCHS.json', rows)
        return result
    det_solver.train_one_epoch = measured
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    if args.task == 'smoke':
        solver.train()
        solver.train_dataloader.set_epoch(25)
        limited = LimitedLoader(solver.train_dataloader)
        measured(False, solver.lr_scheduler, solver.model, solver.criterion, limited,
                 solver.optimizer, solver.device, 25, max_norm=cfg.clip_max_norm,
                 print_freq=cfg.print_freq, ema=solver.ema, scaler=None, writer=None,
                 lr_warmup_scheduler=None)
    else:
        solver.fit()  # Original schedule, augmentation, EMA and Val-driven stage handling.
        if len(rows) != 102 or [r['epoch'] for r in rows] != list(range(102)):
            raise RuntimeError('A partial run cannot be reported as a 102-epoch mean')
    summary = summarize_epochs(rows, smoke=args.task == 'smoke')
    summary.update({'model': row['label'], 'seed': 0, 'protocol': bound['training'],
                    'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
                    'training_config_sha256': sha256(config), 'formal_checkpoints_modified': False})
    write_json(output / 'TRAINING_COST.json', summary)
    with (output / 'EPOCHS.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['train_deim', 'train_gccs'], required=True)
    parser.add_argument('--task', choices=['cpu-check', 'smoke', 'formal'], required=True)
    parser.add_argument('--asset-root')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    thread_setup()
    cpu_check(args) if args.task == 'cpu-check' else gpu_training(args)
