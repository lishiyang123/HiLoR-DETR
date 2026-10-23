#!/usr/bin/env python3
"""Fresh-run training, Train/Val fitting and evaluation, separate from paper replay.

Run one explicit stage at a time in the corresponding host environment.
No stage changes published weights, results, historical records or method settings.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MAIN = ROOT / 'main_host/source/FINAL_CODE'
RT = ROOT / 'second_host'
SETTINGS = dict(lambda_reg=.001, head_seed=0, positive_weight=4, tau=0., beta=2., topk=300)
RT_EVALUATOR = dict(name='faster-coco-eval', version='1.7.2', iou_type='bbox', max_dets=[1, 10, 100])


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def load_module(path, name):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def call_main(module, arguments):
    old = sys.argv
    try:
        sys.argv = [str(module.__file__)] + list(map(str, arguments))
        return module.main()
    finally:
        sys.argv = old


def command(script, *arguments):
    subprocess.run([sys.executable, '-B', str(script), *map(str, arguments)], check=True)


def dataset_paths(dataset, split):
    if dataset == 'cdw':
        base = ROOT / 'main_host/datasets/CDW1630'
        return base / (split + '2017'), base / ('annotations/instances_' + split + '2017.json')
    base = ROOT / 'generalization/btc/data/coco_deim'
    return base / (split + '2017' if split in ('train', 'val') else split), base / ('annotations/instances_' + split + '.json')


def rt_wrapper():
    return load_module(RT / 'reproduce_second_host.py', '_portable_rt')


def config_for(args, capture=False):
    """Materialize existing profiles; only paths, seed and output are run-specific."""
    import yaml
    if args.host == 'deim':
        module = load_module(MAIN / 'reproduce.py', '_portable_main')
        profiles = json.loads((MAIN / 'PROFILES.json').read_text())
        method = 'deim' if args.method in ('native', 'native-ccvr') else 'gsa'
        key = args.dataset + '_' + method + '_train'
        data = module.expand(yaml.safe_load((MAIN / profiles[key]['config']).read_text()), ROOT, MAIN)
        source = MAIN
        initialization = ROOT / profiles[key]['pretrained']
        if digest(initialization) != profiles[key]['pretrained_sha256']:
            raise RuntimeError('Initialization SHA mismatch')
        if capture:
            eval_profile = profiles[f'{args.dataset}_{method}_seed{args.seed}_' + ('test' if args.dataset == 'cdw' else 'test1')]
            evaluation = module.expand(yaml.safe_load((MAIN / eval_profile['config']).read_text()), ROOT, MAIN)
            data['val_dataloader'] = evaluation['val_dataloader']
    else:
        wrapper = rt_wrapper()
        native = args.method == 'native'
        config = RT / ('configs/native_off/cdw1630_rtdetr_r18_native.yml' if native else 'configs/hilor/cdw1630_rtdetr_r18_gsa.yml')
        data = wrapper.relocate(wrapper.yaml_closure(config))
        source = ROOT / 'benchmarks/02_RT_DETR_R18/source/RT-DETR_official/rtdetr_pytorch' if native and not capture else RT / 'source/RT-DETR_HiLoR/rtdetr_pytorch'
        initialization = ROOT / 'benchmarks/02_RT_DETR_R18/source/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth'
        expected = json.loads((RT / 'provenance/PROTOCOL_LOCK.json').read_text())['bindings']['upstream_lock']['coco_initialization_sha256']
        if digest(initialization) != expected:
            raise RuntimeError('RT COCO initialization SHA mismatch')
        if capture:
            data['RTDETRTransformer']['hilor_return_decoder_features'] = True
    for key in ('resume', 'tuning'):
        data.pop(key, None)
    data.update(seed=args.seed, use_amp=False, device='cuda:0', output_dir=str(args.output))
    for loader in ('train_dataloader', 'val_dataloader'):
        folder, annotation = dataset_paths(args.dataset, 'train' if loader.startswith('train') else 'val')
        data[loader]['dataset'].update(img_folder=str(folder), ann_file=str(annotation))
    if args.host == 'deim':
        data['HGNetv2']['pretrained'] = False
    else:
        data['PResNet']['pretrained'] = False
    return source, initialization, data


def save_config(output, name, data):
    import yaml
    path = output / name
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def install_rt_evaluator(source):
    """Use the shared bbox evaluator for RT Val/capture without global patches."""
    sys.path.insert(0, str(source))
    from src import data
    from src.solver import det_engine
    from coco_test_evaluation import evaluator_classes, faster_bbox_evaluator
    evaluator_classes()  # Fail before training if the pinned backend is unavailable.
    evaluator = faster_bbox_evaluator(data.CocoEvaluator)
    data.CocoEvaluator = evaluator
    det_engine.CocoEvaluator = evaluator
    return evaluator


def train(args):
    if args.host == 'deim':
        method = 'deim' if args.method in ('native', 'native-ccvr') else 'gsa'
        command(MAIN / 'reproduce.py', '--release-root', ROOT, '--profile', f'{args.dataset}_{method}_train', '--seed', args.seed, '--output', args.output)
        return
    source, initialization, data = config_for(args)
    config = save_config(args.output, 'config.yml', data)
    launcher = load_module(RT / 'tools/training.py', '_rt_training')
    plan = dict(seed=args.seed, method=args.method, val_evaluator=RT_EVALUATOR,
                config=dict(path=str(config), sha256=digest(config)),
                coco_initialization=dict(path=str(initialization), sha256=digest(initialization)),
                bindings={name + '_sha256': digest(RT / ('provenance/' + name.upper() + '.json')) for name in ('protocol_lock', 'implementation_lock')})
    write_json(args.output / 'TRAIN_RUN_MANIFEST.json', plan)

    def finalize(_root, output, run_plan, selected, elapsed):
        if output.resolve() != args.output:
            raise RuntimeError('Fresh training output mismatch')
        if digest(output / 'best_val.pth') != selected['checkpoint_sha256']:
            raise RuntimeError('Fresh best-Val checkpoint mismatch')
        completed = dict(selected, state='complete', val_evaluator=RT_EVALUATOR, training_seconds=elapsed,
                         training_run_manifest_sha256=digest(output / 'TRAIN_RUN_MANIFEST.json'))
        write_json(output / 'BEST_VAL.json', completed)
        write_json(output / 'RUN_COMPLETED.json', dict(state='complete', seed=args.seed,
                   best_val_checkpoint_sha256=selected['checkpoint_sha256'], test_used=False))
        return completed

    # Keep the recorded optimizer/epoch/EMA/best-Val loop; each fresh run owns
    # its output directory and completion record.
    launcher.finalize_training_run = finalize
    launcher.SOURCE_RELATIVE = source.relative_to(ROOT)
    launcher.install_best_val_training(source, plan)
    install_rt_evaluator(source)
    from src.core import YAMLConfig
    from src.misc import dist
    from src.solver import TASKS
    os.chdir(RT)
    dist.set_seed(args.seed)
    cfg = YAMLConfig(str(config), tuning=str(initialization), resume=None, use_amp=False)
    TASKS[cfg.yaml_cfg['task']](cfg).fit()


def capture(args):
    import torch
    source, _, data = config_for(args, capture=True)
    configs = {}
    for split in ('train', 'val'):
        config = copy.deepcopy(data)
        folder, annotation = dataset_paths(args.dataset, split)
        config['val_dataloader']['dataset'].update(img_folder=str(folder), ann_file=str(annotation))
        configs[split] = save_config(args.output, split + '.yml', config)
    if args.host == 'deim':
        command(MAIN / 'tools/calibration/common/capture_train_val.py', '--branch-root', MAIN,
                '--checkpoint', args.checkpoint, '--expected-checkpoint-sha', digest(args.checkpoint),
                '--train-config', configs['train'], '--val-config', configs['val'], '--output-dir', args.output,
                '--capture-prefix', 'fresh', '--num-classes', 5 if args.dataset == 'cdw' else 3)
        return
    install_rt_evaluator(source)
    module = load_module(RT / 'tools/capture.py', '_rt_capture')
    from src.core import YAMLConfig
    cfg = YAMLConfig(str(configs['train']))
    model = cfg.model.to('cuda:0').eval()
    state, state_source = module.checkpoint_state(args.checkpoint)
    model.load_state_dict(state, strict=True)
    post = cfg.postprocessor.to('cuda:0').eval()
    lock_path = RT / 'provenance/DATA_SPLIT_LOCK.json'
    lock = rt_wrapper().relocate(json.loads(lock_path.read_text()))
    for split in ('train', 'val'):
        module.capture_split(split=split, config_path=configs[split], model=model, postprocessor=post,
            device=torch.device('cuda:0'), checkpoint_path=args.checkpoint, checkpoint_sha=digest(args.checkpoint),
            checkpoint_state_source=state_source, training_run_evidence={'scope': 'fresh_checkpoint_capture'},
            output_dir=args.output, prefix='fresh', data_lock=lock, data_lock_sha=digest(lock_path))


def load_captures(args):
    import torch
    values = {}
    for split in ('train', 'val'):
        path = args.workdir / ('capture/fresh_' + split + '_capture.pt')
        value = torch.load(path, map_location='cpu', weights_only=False)
        if value.get('split') != split or value.get('checkpoint_sha256') != digest(args.checkpoint):
            raise RuntimeError('Capture split/checkpoint mismatch')
        _, annotation = dataset_paths(args.dataset, split)
        expected = {int(row['id']) for row in json.loads(annotation.read_text())['images']}
        ids = [int(row['image_id']) for row in value['records']]
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise RuntimeError('Capture must contain every image in its declared split exactly once')
        count = 5 if args.dataset == 'cdw' else 3
        for record in value['records']:
            if tuple(record['pred_logits'].shape) != (300, count) or tuple(record['pred_boxes'].shape) != (300, 4):
                raise RuntimeError('Capture query/class shape mismatch')
        values[split] = value
    return values


def main_train_data(module, capture_value, parameters):
    """Use the original main-host marginal labels and runtime feature builders."""
    import torch
    from engine.deim.qccr import apply_qccr, compute_competition_risk
    from engine.deim.umr import box_cxcywh_to_xyxy, build_image_features
    rows = {key: [] for key in ('features', 'classes', 'hidden', 'targets')}
    ids = []
    for record in capture_value['records']:
        raw, boxes = record['pred_logits'].float(), record['pred_boxes'].float()
        height, width = [int(v) for v in record['orig_size'].tolist()]
        absolute = box_cxcywh_to_xyxy(boxes) * torch.tensor([width, height, width, height], dtype=torch.float32)
        risk = compute_competition_risk(raw.unsqueeze(0), boxes.unsqueeze(0))[0]
        calibrated = apply_qccr(raw, risk, parameters['beta0'], parameters['beta_positive'], parameters['beta_negative'], parameters['duplicate_penalty'])
        score = torch.sigmoid(calibrated)
        labels = module.compute_marginal_labels(record['gt_boxes'], record['gt_labels'], record['gt_iscrowd'], absolute, score)
        index, features, classes, hidden = build_image_features(raw, boxes, absolute, record['hL'].float(), calibrated, score, risk, K=300)
        if not torch.equal(classes, index % module.NUM_CLASSES):
            raise RuntimeError('Feature class mapping mismatch')
        for key, value in dict(features=features, classes=classes, hidden=hidden, targets=labels[index // module.NUM_CLASSES, index % module.NUM_CLASSES]).items():
            rows[key].append(value)
        ids.append(int(record['image_id']))
    return {key: torch.stack(value) for key, value in rows.items()}, ids


def fit(args):
    import torch
    captures = load_captures(args)
    target = args.output / 'qccr_targets.pt'
    fits_path = args.output / 'qccr_fits.json'
    capture_paths = {split: args.workdir / ('capture/fresh_' + split + '_capture.pt') for split in ('train', 'val')}
    if args.host == 'deim':
        sys.path.insert(0, str(MAIN))
        builder = load_module(MAIN / 'tools/qccr/build_qccr_targets.py', '_main_targets')
        call_main(builder, ['--capture', capture_paths['val'], '--output', target, '--summary', args.output / 'targets_summary.json'])
        fitter = load_module(MAIN / 'tools/qccr/fit_qccr.py', '_main_fitter')
        fitter.LAMBDA_REG_VALUES = (.001,)
        call_main(fitter, ['--targets', target, '--output', fits_path, '--seed', 0, '--folds', 5])
        fits = json.loads(fits_path.read_text())
        trainer = load_module(MAIN / 'tools/calibration/common/train_umr_select_val.py', '_main_value_trainer')
        trainer.NUM_CLASSES = 5 if args.dataset == 'cdw' else 3
        from engine.deim.umr import UMRHead
    else:
        source = RT / 'source/RT-DETR_HiLoR/rtdetr_pytorch'
        builder = load_module(RT / 'tools/build_qccr_targets.py', '_rt_targets')
        payload = builder.build_payload(captures['val'], capture_paths['val'], digest(capture_paths['val']), source,
                  implementation_lock_sha256=digest(RT / 'provenance/IMPLEMENTATION_LOCK.json'), detector_seed=args.seed)
        torch.save(payload, target)
        fitter = load_module(RT / 'tools/fit_qccr.py', '_rt_fitter')
        fitter.LAMBDA_REG_VALUES = (.001,)
        fits = fitter.run_fit(payload, target_path=target, target_sha=digest(target), seed=0, folds=5)
        trainer = load_module(RT / 'tools/train_umr_select_val.py', '_rt_value_trainer')
        from fit_common import install_source_root, runtime_source_hashes
        apply, risk, UMRHead, features = install_source_root(source, require_umr=True)
    fits['selection_status'] = 'FIXED_PROTOCOL_DIAGNOSTICS_ONLY'
    fits['fixed_lambda_reg'] = .001
    fits['adaptive_selection_performed'] = False
    write_json(fits_path, fits)
    candidate = next(row for row in fits['candidates'].values() if row['lambda_reg'] == .001)
    parameters = candidate['full_val_fit']['parameters']
    qccr = dict(method='QCCR-v1', parameters=parameters, selected_lambda_reg=.001,
                source_checkpoint_sha256=digest(args.checkpoint), source_capture_sha256=digest(capture_paths['val']),
                selection_decision='FIXED_PROTOCOL_READY', test_used_for_selection=False)
    if args.host == 'deim':
        train_data, ids = main_train_data(trainer, captures['train'], parameters)
    else:
        train_data, ids, _, _ = trainer.build_train_data(captures['train'], parameters, apply, risk, features)
    model, history, best_epoch, best_holdout, fit_count, holdout_count = trainer.train_one(UMRHead, train_data, ids, 0, 4)
    model_path = args.output / 'umr_seed0_pw4_best.pt'
    torch.save(model.state_dict(), model_path)
    write_json(args.output / 'UMR_TRAINING.json', dict(best_epoch=best_epoch, best_holdout_qfl=best_holdout,
               fit_image_count=fit_count, holdout_image_count=holdout_count, history=history,
               weight_sha256=digest(model_path), source_checkpoint_sha256=digest(args.checkpoint),
               early_stopping_split='Train internal holdout', val_used_for_early_stopping=False))
    qccr_path = args.output / 'qccr.json'
    if args.host == 'rtdetr':
        qccr.update(dataset='CDW1630', detector_seed=args.seed,
                     source_data_split_lock_sha256=digest(RT / 'provenance/DATA_SPLIT_LOCK.json'),
                     runtime_source_sha256=runtime_source_hashes(source, require_umr=False))
    write_json(qccr_path, qccr)
    binding = dict(host=args.host, dataset=args.dataset, detector_seed=args.seed,
                   checkpoint_sha256=digest(args.checkpoint), settings=SETTINGS,
                   test_used_for_fitting_or_selection=False,
                   qccr=dict(path=qccr_path.name, sha256=digest(qccr_path)),
                   umr=dict(path=model_path.name, sha256=digest(model_path)),
                   captures={key: dict(path=str(value), sha256=digest(value)) for key, value in capture_paths.items()})
    write_json(args.output / 'FIT_BINDING.json', binding)
    if args.host == 'rtdetr':
        bind_rt(args, source, capture_paths, qccr_path, model_path)


def bind_rt(args, source, capture_paths, qccr_path, model_path):
    helper = load_module(RT / 'tools/train_umr_fixed.py', '_rt_fixed_binding')
    runtime = helper.runtime_source_hashes(source, require_umr=True)
    checkpoint_sha = digest(args.checkpoint)
    data_sha = digest(RT / 'provenance/DATA_SPLIT_LOCK.json')
    implementation_sha = digest(RT / 'provenance/IMPLEMENTATION_LOCK.json')
    lock = dict(path=str(RT / 'provenance/INFERENCE_FIXED_PROTOCOL_LOCK.json'), sha256=digest(RT / 'provenance/INFERENCE_FIXED_PROTOCOL_LOCK.json'))
    decision = helper.build_fixed_decision(detector_seed=args.seed, checkpoint_sha=checkpoint_sha,
        train_capture_sha=digest(capture_paths['train']), val_capture_sha=digest(capture_paths['val']),
        data_lock_sha=data_sha, qccr_sha=digest(qccr_path), qccr_source_capture_sha=digest(capture_paths['val']),
        runtime_hashes=runtime, model_path=model_path, model_sha=digest(model_path),
        implementation_sha=implementation_sha, fixed_lock=lock)
    decision['qccr_refitted'] = True
    selection = args.output / 'FIXED_SELECTION.json'
    write_json(selection, decision)
    binding = dict(schema='cdw1630-rtdetr-r18-qccr-umr-inference-binding-v1', format_version=2,
        host='RT-DETR-R18', dataset='CDW1630', mode='qccr-umr', detector_seed=args.seed,
        detector_checkpoint=str(args.checkpoint), detector_checkpoint_sha256=checkpoint_sha,
        test_accessed_for_fitting_or_selection=False, fit_and_selection_splits=['train', 'val'],
        implementation_lock_sha256=implementation_sha, runtime_source_sha256=runtime,
        source_data_split_lock_sha256=data_sha, source_val_capture_sha256=digest(capture_paths['val']),
        fixed_protocol_lock=lock, selection_policy=decision['selection_policy'],
        captures={key: dict(sha256=digest(value), detector_checkpoint_sha256=checkpoint_sha, data_split_lock_sha256=data_sha) for key, value in capture_paths.items()},
        qccr_artifact=dict(path=qccr_path.name, sha256=digest(qccr_path), source_capture_sha256=digest(capture_paths['val']), source_data_split_lock_sha256=data_sha, runtime_source_sha256={'hilor_qccr.py': runtime['hilor_qccr.py']}),
        umr=dict(path=model_path.name, sha256=digest(model_path), selection_path=selection.name,
                 selection_sha256=digest(selection), feature_schema='umr-25d-rtdetr-final-hidden-v1', num_classes=5,
                 num_queries=300, topk=300, tau=0., beta=2., runtime_source_sha256=runtime,
                 source_val_capture_sha256=digest(capture_paths['val'])))
    write_json(args.output / 'RT_INFERENCE_BINDING.json', binding)


def evaluate(args):
    if args.host == 'deim':
        method = {'native': 'deim', 'native-ccvr': 'deim_ccvr'}.get(args.method, args.method)
        extra = ['--ccvr-binding', args.workdir / 'fit/FIT_BINDING.json'] if method in ('full', 'deim_ccvr') else []
        command(MAIN / 'reproduce.py', '--release-root', ROOT, '--profile', f'{args.dataset}_{method}_seed{args.seed}_{args.split}',
                '--checkpoint', args.checkpoint, '--output', args.output, *extra)
    else:
        extra = ['--binding', args.workdir / 'fit/RT_INFERENCE_BINDING.json'] if args.method == 'full' else []
        command(RT / 'reproduce_second_host.py', '--seed', args.seed, '--mode', args.method,
                '--checkpoint', args.checkpoint, '--output', args.output, *extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', choices=('deim', 'rtdetr'), required=True)
    parser.add_argument('--dataset', choices=('cdw', 'btc'), required=True)
    parser.add_argument('--method', choices=('native', 'gsa', 'full', 'native-ccvr'), required=True)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--stage', choices=('train', 'capture', 'fit', 'evaluate'), required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, help='Required after training; explicitly choose best Val checkpoint, never Test')
    parser.add_argument('--split', choices=('test', 'test1', 'test2'), default='test')
    parser.add_argument('--preflight-only', action='store_true', help='Check inputs/configuration; do not train, fit, infer or write files')
    args = parser.parse_args()
    args.workdir = args.workdir.resolve()
    if args.workdir == ROOT or ROOT in args.workdir.parents:
        parser.error('Use a work directory outside the published release')
    if args.host == 'rtdetr' and args.dataset != 'cdw':
        parser.error('Second-host protocol is CDW1630 only')
    if args.method == 'native-ccvr' and (args.host != 'deim' or args.dataset != 'cdw'):
        parser.error('Native+CCVR control is declared only for the CDW DEIM host')
    if args.stage in ('capture', 'fit') and args.method not in ('full', 'native-ccvr'):
        parser.error('Capture/fit stages require full or native-ccvr')
    if args.stage == 'evaluate' and (args.dataset == 'cdw' and args.split != 'test' or args.dataset == 'btc' and args.split == 'test'):
        parser.error('CDW uses test; BTC uses test1/test2')
    if args.stage != 'train':
        if args.checkpoint is None:
            parser.error('Supply the explicitly Val-selected --checkpoint')
        args.checkpoint = args.checkpoint.resolve(strict=True)
    args.output = args.workdir / (args.stage if args.stage != 'evaluate' else 'evaluate_' + args.split)
    if args.output.exists():
        parser.error('Stage output already exists; use another work directory')
    for split in ('train', 'val') if args.stage != 'evaluate' else (args.split,):
        folder, annotation = dataset_paths(args.dataset, split)
        if not folder.is_dir() or not annotation.is_file():
            raise FileNotFoundError('Dataset input missing: ' + str(annotation))
    if args.stage in ('train', 'capture'):
        source, initialization, data = config_for(args, capture=args.stage == 'capture')
        print(json.dumps(dict(stage=args.stage, source=str(source), initialization=str(initialization),
              seed=args.seed, training_batch=data['train_dataloader'].get('total_batch_size', data['train_dataloader'].get('batch_size')),
              evaluation_batch=data['val_dataloader'].get('total_batch_size', data['val_dataloader'].get('batch_size')), amp=data['use_amp'],
              evaluator=RT_EVALUATOR if args.host == 'rtdetr' else 'main-host profile')))
        if args.host == 'rtdetr':
            from coco_test_evaluation import evaluator_classes
            evaluator_classes()
    if args.stage == 'fit':
        load_captures(args)
    if args.stage == 'evaluate':
        if args.host == 'rtdetr':
            rt_wrapper().plan(args.seed, args.method, args.checkpoint,
                args.workdir / 'fit/RT_INFERENCE_BINDING.json' if args.method == 'full' else None)
        else:
            method = {'native': 'deim', 'native-ccvr': 'deim_ccvr'}.get(args.method, args.method)
            extra = ['--ccvr-binding', args.workdir / 'fit/FIT_BINDING.json'] if method in ('full', 'deim_ccvr') else []
            command(MAIN / 'reproduce.py', '--release-root', ROOT, '--profile', f'{args.dataset}_{method}_seed{args.seed}_{args.split}', '--checkpoint', args.checkpoint,
                    '--output', args.output, '--preflight-only', *extra)
    if args.preflight_only:
        print('FRESH-RUN PREFLIGHT: PASS; no training, fitting, inference or files written')
        return
    if args.stage in ('train', 'capture'):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for detector training/capture')
    # Main training creates its own output; other stages create it here.
    if args.stage != 'evaluate' and not (args.stage == 'train' and args.host == 'deim'):
        args.output.mkdir(parents=True)
    globals()[args.stage](args)


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
