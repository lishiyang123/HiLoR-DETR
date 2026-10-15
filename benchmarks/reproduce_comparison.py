"""Plan or train a comparison model with its recorded Train/Val settings.

Default is a no-write, CPU-only plan. Execution requires --train and a fresh
external output directory. No Test data or saved paper results are consulted
for checkpoint selection.
"""
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent
MODELS = {
    'dfine': '03_D_FINE_M',
    'yolov9': '10_YOLOV9_C',
    'yolov10': '01_YOLO_SERIES/YOLOv10_M_TRAINED',
    'yolov12': '01_YOLO_SERIES/YOLOv12_M_TRAINED',
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def plan(model, seed, output):
    import yaml
    folder = HERE / MODELS[model]
    portable = load(RELEASE / 'second_host/reproduce_second_host.py', '_sota_paths')
    if model == 'dfine':
        source = folder / 'source/D-FINE_official'
        config = portable.relocate(portable.yaml_closure(folder / 'configs/cdw1630_dfine_m_formal.yml'))
        config.update(seed=seed, output_dir=str(output), use_amp=False)
        initialization = folder / 'source/dfine_m_coco.pth'
    else:
        source = folder / ('source/YOLOv9_official' if model == 'yolov9' else
                           'source/YOLOv10_official' if model == 'yolov10' else 'source/YOLOv12_official')
        filename = 'opt.yaml' if model == 'yolov9' else 'args.yaml'
        config = portable.relocate(yaml.safe_load((folder / f'runs/seed{seed}/{filename}').read_text()))
        initialization = folder / ('source/yolov9-c.pt' if model == 'yolov9' else
                                  'source/yolov10m.pt' if model == 'yolov10' else 'source/yolov12m.pt')
        config.update(seed=seed, project=str(output.parent), name=output.name,
                      exist_ok=False, resume=False)
        config['weights' if model == 'yolov9' else 'model'] = str(initialization)
    if not source.is_dir() or not initialization.is_file():
        raise FileNotFoundError('Model source or official initialization is missing')
    if model == 'dfine':
        for key in ('train_dataloader', 'val_dataloader'):
            annotation = Path(config[key]['dataset']['ann_file'])
            if not annotation.is_file() or 'test' in annotation.name.lower():
                raise RuntimeError('Training must use the recorded Train/Val annotations')
    else:
        dataset = yaml.safe_load(Path(config['data']).read_text())
        base = Path(portable.relocate(dataset['path']))
        for split in ('train', 'val'):
            if not (base / dataset[split]).is_dir() or 'test' in dataset[split].lower():
                raise RuntimeError('Recorded YOLO Train/Val images are missing or include Test')
        config['_dataset'] = dict(dataset, path=str(base))
    return folder, source, initialization, config


def select_checkpoint(output, source, seed, epoch, ap):
    if not math.isfinite(ap):
        raise RuntimeError('Nonfinite Val AP')
    target = output / 'weights/best_val_ap.pt'
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.incoming')
    shutil.copy2(source, temporary)
    os.replace(temporary, target)
    (output / 'VAL_SELECTION.json').write_text(json.dumps(dict(seed=seed,
        epoch_zero_based=epoch, val_ap=ap, checkpoint='weights/best_val_ap.pt',
        selection='maximum Val AP; earliest exact tie', test_used=False), indent=2)+'\n')


def train_yolo(model, seed, source, initialization, config, output):
    import yaml
    sys.path.insert(0, str(source))
    os.chdir(source)
    dataset = config.pop('_dataset')
    data_path = output.parent / (output.name + '_dataset.yaml')
    if data_path.exists():
        raise FileExistsError(data_path)
    data_path.write_text(yaml.safe_dump(dataset, sort_keys=False))
    config['data'] = str(data_path)
    best = -math.inf
    seen = set()
    if model == 'yolov9':
        from utils.callbacks import Callbacks
        module = load(source / 'train_dual.py', '_yolov9_training')
        module.check_amp = lambda _model: False
        callbacks = Callbacks()
        pending = {}
        def on_fit(log_values, epoch, *_):
            nonlocal best
            if epoch in seen:
                return
            seen.add(epoch)
            ap = float(log_values[6])  # 3 train losses + P/R/AP50/AP50:95
            if math.isfinite(ap) and ap > best:
                best = ap
                pending.update(epoch=epoch, ap=ap)
        def on_save(last, epoch, *_):
            if pending.get('epoch') == epoch:
                select_checkpoint(output, Path(last), seed, epoch, pending['ap'])
        callbacks.register_action('on_fit_epoch_end', callback=on_fit)
        callbacks.register_action('on_model_save', callback=on_save)
        config.update(save_dir=str(output), weights=str(initialization))
        module.main(argparse.Namespace(**config), callbacks=callbacks)
    else:
        from ultralytics import YOLO
        if model == 'yolov10':
            from ultralytics import YOLOv10
            detector = YOLOv10(str(initialization))
        else:
            detector = YOLO(str(initialization))
        from ultralytics.cfg import DEFAULT_CFG_DICT
        arguments = {k:v for k,v in config.items() if k in DEFAULT_CFG_DICT
                     and k not in ('model', 'mode', 'task', 'save_dir')}
        arguments.update(amp=False, val=True)
        def on_fit(trainer):
            nonlocal best
            epoch = int(trainer.epoch)
            if epoch in seen:
                return
            seen.add(epoch)
            ap = float(trainer.metrics['metrics/mAP50-95(B)'])
            if math.isfinite(ap) and ap > best:
                best = ap
                select_checkpoint(output, Path(trainer.last), seed, epoch, ap)
        detector.add_callback('on_fit_epoch_end', on_fit)
        detector.train(**arguments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--seed', type=int, choices=(0,1,2), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if output == RELEASE or RELEASE in output.parents or output.exists():
        parser.error('Use a new output directory outside the release')
    folder, source, initialization, config = plan(args.model, args.seed, output)
    print(json.dumps(dict(model=args.model, seed=args.seed, source=str(source),
        initialization=str(initialization), output=str(output), configuration=config,
        execution='TRAIN' if args.train else 'PLAN_ONLY'), indent=2))
    if not args.train:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required to train; omit --train for CPU planning')
    if args.model == 'dfine':
        import yaml
        output.parent.mkdir(parents=True, exist_ok=True)
        path = output.parent / (output.name + '_config.yml')
        if path.exists():
            raise FileExistsError(path)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        subprocess.run([sys.executable, '-B', str(source/'train.py'), '--config', str(path),
            '--tuning', str(initialization), '--seed', str(args.seed), '--output-dir', str(output)],
            cwd=source, check=True)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        train_yolo(args.model, args.seed, source, initialization, config, output)


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
