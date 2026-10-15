"""Prepare relocatable configs from existing formal assets; never edit those assets."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import yaml

from scoped_common import PACKAGE, roots, sha256, write_json, refresh_manifest


def merge(base, child):
    for key, value in child.items():
        if key == '__include__':
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_yaml(path):
    path = Path(path).resolve(strict=True)
    data = yaml.safe_load(path.read_text()) or {}
    result = {}
    for include in data.get('__include__', []):
        inc = Path(include)
        merge(result, load_yaml(inc if inc.is_absolute() else path.parent / inc))
    return merge(result, data)


def tokenize(value, mapping):
    if isinstance(value, dict):
        return {k: tokenize(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [tokenize(v, mapping) for v in value]
    if isinstance(value, str):
        for key, path in sorted(mapping.items(), key=lambda item: -len(str(item[1]))):
            text = str(path)
            if value == text or value.startswith(text + '/'):
                return '${' + key + '}' + value[len(text):]
    return value


def prepare(project=None):
    mapping = roots(project)
    primary = mapping['PRIMARY_ROOT']
    source = primary / 'source/FINAL_CODE'
    old_source = primary / 'source/FINAL_CODE'
    sota = mapping['SOTA_ROOT'] / '02_RT_DETR_R18'
    second = mapping['SECOND_ROOT']
    frozen_path = primary / 'training_records/CDW1630/Ours/seed0/calibration_evidence/FROZEN_INFERENCE.json'
    frozen = json.loads(frozen_path.read_text())
    for entry in (frozen['checkpoint'], frozen['qccr'], frozen['umr']['checkpoint']):
        path = primary / entry['path']
        if sha256(path) != entry['sha256']:
            raise RuntimeError('Primary frozen asset mismatch: ' + str(path))
    # Resolve portable paths in the release-bound source profiles.
    def fix_primary(value):
        if isinstance(value, dict):
            if value.get('gglh_enabled') not in (None, False):
                raise RuntimeError('Cannot discard an enabled historical branch')
            return {k: fix_primary(v) for k, v in value.items() if k != 'gglh_enabled'}
        if isinstance(value, list):
            return [fix_primary(v) for v in value]
        if isinstance(value, str):
            value = value.replace('${RELEASE_ROOT}', str(primary.parent)).replace('${SOURCE_ROOT}', str(source))
        if isinstance(value, str) and value.startswith(str(old_source)):
            return str(source) + value[len(str(old_source)):]
        return value

    common_path = primary / 'protocols/materialized_configs/CDW1630/cdw_common_train.yml'
    baseline = fix_primary(load_yaml(source / 'profiles/cdw_deim_train.yml'))
    gccs = fix_primary(load_yaml(source / 'profiles/cdw_gccs_train.yml'))
    for cfg in (baseline, gccs):
        cfg['use_amp'] = False  # Matches formal train.py's default CLI override.
        cfg['use_ema'] = True
        cfg['sync_bn'] = False  # Single-device measurement.
        cfg['device'] = 'cuda'
        cfg['seed'] = 0
        for split in ('train', 'val'):
            loader = cfg[split + '_dataloader']
            loader['dataset']['img_folder'] = str(mapping['DATA_ROOT'] / (split + '2017'))
            loader['dataset']['ann_file'] = str(mapping['DATA_ROOT'] / 'annotations' / ('instances_' + split + '2017.json'))
    configs = {}
    models = {}
    def add(name, label, cfg, src, flavor, checkpoint, python, full=False, fitted=0):
        config = copy.deepcopy(cfg)
        config['output_dir'] = '${PACKAGE_ROOT}/outputs/' + name
        if 'HGNetv2' in config:
            config['HGNetv2']['pretrained'] = False
        if 'PResNet' in config:
            config['PResNet']['pretrained'] = False
        for split in ('train', 'val'):
            if split + '_dataloader' in config:
                ds = config[split + '_dataloader']['dataset']
                ds['img_folder'] = str(mapping['DATA_ROOT'] / (split + '2017'))
                ds['ann_file'] = str(mapping['DATA_ROOT'] / 'annotations' / ('instances_' + split + '2017.json'))
        configs[name] = tokenize(config, mapping)
        models[name] = {
            'label': label, 'source': tokenize(str(src), mapping), 'flavor': flavor,
            'checkpoint': tokenize(str(checkpoint), mapping), 'checkpoint_sha256': sha256(checkpoint),
            'python': '${ENV_ROOT}/' + python + '/bin/python', 'full': full, 'fitted_scalars': fitted,
        }

    b0_checkpoint = primary / 'checkpoints/CDW1630/B0/seed0/best_stg1.pth'
    ours_checkpoint = primary / frozen['checkpoint']['path']
    for name, label, cfg, cp in [('deim', 'DEIM', baseline, b0_checkpoint), ('gccs', 'DEIM + GCCS', gccs, ours_checkpoint)]:
        cfg = copy.deepcopy(cfg)
        cfg['DFINETransformer']['ufd_deploy_strip'] = False  # Strict load first, then deploy.
        add(name, label, cfg, source, 'engine', cp, 'deim_cvpr2025')
    full = copy.deepcopy(gccs)
    full['DFINETransformer']['ufd_deploy_strip'] = False
    q = frozen['qccr']['parameters']
    full['PostProcessor'].update({
        'qccr_enabled': True, 'umr_enabled': True,
        'qccr_beta0': q['beta0'], 'qccr_beta_positive': q['beta_positive'],
        'qccr_beta_negative': q['beta_negative'], 'qccr_duplicate_penalty': q['duplicate_penalty'],
        'umr_checkpoint': str(primary / frozen['umr']['checkpoint']['path']),
        'umr_topk': frozen['umr']['topk'], 'umr_tau': frozen['umr']['tau'], 'umr_beta': frozen['umr']['beta'],
    })
    add('hilor', 'HiLoR-DETR', full, source, 'engine', ours_checkpoint, 'deim_cvpr2025', True, 4)
    models['hilor']['head'] = {'path': tokenize(str(primary / frozen['umr']['checkpoint']['path']), mapping), 'sha256': frozen['umr']['checkpoint']['sha256']}
    native_cfg = load_yaml(sota / 'configs/cdw1630_rtdetr_r18_frozen_test_v1.yml')
    add('r18', 'RT-DETR-R18', native_cfg, sota / 'source/RT-DETR_official/rtdetr_pytorch', 'src', sota / 'runs/seed0/best_val.pth', 'rtdetr_sota')
    adapted_cfg_path = second / 'configs/hilor/generated/seed0/qccr_umr.yml'
    adapted = load_yaml(adapted_cfg_path)
    add('r18_hilor', 'RT-DETR-R18 + HiLoR', adapted, second / 'source/RT-DETR_HiLoR/rtdetr_pytorch', 'src', second / 'runs/seed0/best_val.pth', 'rtdetr_sota', True, 4)
    secondary_path = second / 'artifacts/inference/seed0/qccr_umr_binding.json'
    secondary = json.loads(secondary_path.read_text())
    if models['r18_hilor']['checkpoint_sha256'] != secondary['detector_checkpoint_sha256']:
        raise RuntimeError('Secondary detector does not match the frozen inference binding')
    secondary_head = secondary_path.parent / secondary['umr']['path']
    if sha256(secondary_head) != secondary['umr']['sha256']:
        raise RuntimeError('Secondary value-head SHA mismatch')
    models['r18_hilor']['head'] = {'path': tokenize(str(secondary_head), mapping), 'sha256': secondary['umr']['sha256']}
    for name, cfg in [('train_deim', baseline), ('train_gccs', gccs)]:
        cfg = copy.deepcopy(cfg)
        cfg['output_dir'] = '${PACKAGE_ROOT}/outputs/' + name
        cfg['tuning'] = str(primary / 'pretrained/deim_hgnetv2_m_coco.pth')
        cfg['resume'] = None
        cfg['summary_dir'] = '${PACKAGE_ROOT}/outputs/' + name + '/tensorboard'
        configs[name] = tokenize(cfg, mapping)
        models[name] = {'label': 'DEIM' if name == 'train_deim' else 'DEIM + GCCS', 'source': '${PRIMARY_ROOT}/source/FINAL_CODE', 'flavor': 'engine', 'python': '${ENV_ROOT}/deim_cvpr2025/bin/python', 'full': False}
    for name, cfg in configs.items():
        write_json(PACKAGE / 'configs/templates' / (name + '.json'), cfg)
    binding = {
        'schema': 'scoped-efficiency-run-bindings-v1', 'seed': 0,
        'data': {'root': '${DATA_ROOT}', 'annotation': '${DATA_ROOT}/annotations/instances_test2017.json',
                 'val_annotation': '${DATA_ROOT}/annotations/instances_val2017.json',
                 'test_annotation_sha256': sha256(mapping['DATA_ROOT'] / 'annotations/instances_test2017.json')},
        'primary_frozen_inference': {'path': '${PRIMARY_ROOT}/' + str(frozen_path.relative_to(primary)), 'sha256': sha256(frozen_path)},
        'secondary_binding': {'path': '${SECOND_ROOT}/artifacts/inference/seed0/qccr_umr_binding.json', 'sha256': sha256(second / 'artifacts/inference/seed0/qccr_umr_binding.json')},
        'models': models, 'complexity_models': ['hilor', 'r18', 'r18_hilor'],
        'latency_models': ['deim', 'gccs', 'hilor'],
        'training': {'models': ['train_deim', 'train_gccs'], 'epochs': 102, 'batch_size': 8,
                     'precision': 'FP32', 'ema': True, 'mean_scope': 'all 102 complete train_one_epoch calls; validation and checkpoint writing excluded',
                     'peak_scope': 'max allocated CUDA memory over training only; model/criterion/optimizer/EMA included',
                     'initialization': 'same official COCO pretrained checkpoint',
                     'tensorboard': 'disabled for both; engine diagnostic logging retained',
                     'initialization_sha256': sha256(primary / 'pretrained/deim_hgnetv2_m_coco.pth')},
        'prepared_from': {'primary_common_config_sha256': sha256(common_path), 'secondary_generated_config_sha256': sha256(adapted_cfg_path)},
    }
    write_json(PACKAGE / 'SCOPED_BINDINGS.json', binding)
    scope_path = PACKAGE / 'SCOPE.json'
    scope = json.loads(scope_path.read_text())
    scope['status'] = 'SCOPED_RUNNERS_PREPARED_GPU_VERIFICATION_PENDING'
    scope['training_validation_scope'] = binding['training']['mean_scope']
    scope['runner'] = 'run_scoped.py; separate check, smoke, inference, training-smoke, training stages'
    write_json(scope_path, scope)
    refresh_manifest()
    print('SCOPED CONFIG PREPARATION: PASS; formal assets not modified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-root')
    prepare(parser.parse_args().asset_root)
