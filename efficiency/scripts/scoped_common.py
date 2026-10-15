"""Path bindings and deployment helpers for the scoped efficiency experiment."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def roots(project=None):
    project = Path(project or os.environ.get('EFFICIENCY_ASSET_ROOT', PACKAGE.parent)).resolve()
    release = project if (project / 'main_host').is_dir() else project / 'HiLoR-DETR'
    env_default = project.parent.parent / 'envs' if release == project else project.parent / 'envs'
    vendor_default = PACKAGE / 'vendor' if (PACKAGE / 'vendor').is_dir() else project / 'SOTA_NINE_MODEL_REAL_TEST_EFFICIENCY_STATE_v2/vendor'
    result = {
        'PROJECT_ROOT': project,
        'PACKAGE_ROOT': PACKAGE,
        'PRIMARY_ROOT': Path(os.environ.get('EFFICIENCY_PRIMARY_ROOT', release / 'main_host')).resolve(),
        'SOTA_ROOT': Path(os.environ.get('EFFICIENCY_SOTA_ROOT', release / 'benchmarks')).resolve(),
        'SECOND_ROOT': Path(os.environ.get('EFFICIENCY_SECOND_ROOT', project / 'second_host')).resolve(),
        'ENV_ROOT': Path(os.environ.get('EFFICIENCY_ENV_ROOT', env_default)).resolve(),
        'VENDOR_ROOT': Path(os.environ.get('EFFICIENCY_VENDOR_ROOT', vendor_default)).resolve(),
    }
    result['DATA_ROOT'] = Path(os.environ.get('DATA_ROOT', result['PRIMARY_ROOT'] / 'datasets/CDW1630')).resolve()
    return result


def expand(value, mapping):
    if isinstance(value, dict):
        return {k: expand(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v, mapping) for v in value]
    if isinstance(value, str):
        for key, path in mapping.items():
            value = value.replace('${' + key + '}', str(path))
        if '${' in value:
            raise ValueError('Unresolved path token: ' + value)
    return value


def bindings(project=None):
    return expand(json.loads((PACKAGE / 'SCOPED_BINDINGS.json').read_text()), roots(project))


def torch_load(path):
    import torch
    # Only user-owned, SHA-bound research checkpoints are read here.
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def thread_setup():
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def make_cfg(name, project=None, config_dir=None):
    import yaml
    bound = bindings(project)
    row = bound['models'][name]
    source = Path(row['source'])
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(roots(project)['VENDOR_ROOT']))
    if row['flavor'] == 'engine':
        from engine.core import YAMLConfig
    else:
        from src.core import YAMLConfig
    template = PACKAGE / 'configs/templates' / (name + '.json')
    cfg_data = expand(json.loads(template.read_text()), roots(project))
    config = Path(config_dir or PACKAGE / 'configs/generated') / (name + '.yml')
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.safe_dump(cfg_data, sort_keys=False), encoding='utf-8')
    return YAMLConfig(str(config)), row, bound, config


def parameters(module):
    return sum(p.numel() for p in module.parameters())


def load_detector(cfg, row):
    import torch
    path = Path(row['checkpoint'])
    if sha256(path) != row['checkpoint_sha256']:
        raise RuntimeError('Detector checkpoint SHA mismatch: ' + row['label'])
    if row.get('head') and sha256(row['head']['path']) != row['head']['sha256']:
        raise RuntimeError('Frozen value-head SHA mismatch')
    detector = cfg.model
    checkpoint = torch_load(path)
    ema = checkpoint.get('ema')
    state = ema.get('module') if isinstance(ema, dict) else None
    weight_source = 'ema.module'
    if not isinstance(state, dict):
        state = checkpoint.get('model')
        weight_source = 'model'
    if not isinstance(state, dict):
        raise RuntimeError('Checkpoint has no detector state')
    detector.load_state_dict(state, strict=True)
    del checkpoint, state
    post = cfg.postprocessor
    if bool(getattr(post, 'qccr_enabled', False)) != row['full']:
        raise RuntimeError('QCCR switch differs from measurement binding')
    if bool(getattr(post, 'umr_enabled', False)) != row['full']:
        raise RuntimeError('UMR switch differs from measurement binding')
    if row['full']:
        if getattr(post.umr, 'topk', None) != 300 or float(post.umr.beta) != 2.0 or float(post.umr.tau) != 0.0:
            raise RuntimeError('Frozen UMR inference tuple mismatch')
        head_count = parameters(post.umr.head)
        if head_count != 14729 or any(isinstance(m, torch.nn.MultiheadAttention) for m in post.umr.head.modules()):
            raise RuntimeError('Expected formal 14,729-parameter MLP, not a legacy attention head')
    return detector.float().eval(), post.float().eval(), weight_source


def trim_retired_branches(detector):
    """Only remove branches explicitly bypassed by the official fused forward."""
    removed = []
    for name, module in list(detector.named_modules()):
        if type(module).__name__ == 'RepVggBlock' and hasattr(module, 'conv'):
            for attr in ('conv1', 'conv2'):
                if hasattr(module, attr):
                    removed.append({'path': name + '.' + attr, 'parameters': parameters(getattr(module, attr))})
                    delattr(module, attr)
    return removed


def deploy(detector):
    detector = detector.deploy()
    removed = trim_retired_branches(detector)
    return detector.float().eval(), removed


def fp32_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {k: fp32_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [fp32_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(fp32_tree(v) for v in value)
    return value


def assert_tree_equal(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        if not torch.equal(left, right):
            raise AssertionError('Retired-branch removal changed a forward tensor')
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def refresh_manifest():
    files = {}
    for path in sorted(PACKAGE.rglob('*')):
        relative = path.relative_to(PACKAGE)
        if not path.is_file() or '__pycache__' in relative.parts or relative.parts[0] in ('outputs', 'checks') or path.name == 'PACKAGE_MANIFEST.json':
            continue
        if relative.parts[:2] == ('configs', 'generated'):
            continue
        files[relative.as_posix()] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    scope_path = PACKAGE / 'SCOPE.json'
    status = json.loads(scope_path.read_text()).get('status') if scope_path.exists() else 'PREPARING'
    write_json(PACKAGE / 'PACKAGE_MANIFEST.json', {'schema': 'efficiency-working-file-manifest-v2', 'files': files, 'status': status})
