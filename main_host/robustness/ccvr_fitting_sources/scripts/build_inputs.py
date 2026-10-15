#!/usr/bin/env python3
"""Resolve released assets and explicitly supplied Train/Val captures on CPU."""
import argparse
from pathlib import Path
from common import read_json, require, sha256, write_json_atomic

PACKAGE = Path('main_host/robustness/ccvr_fitting_sources')


def release_root():
    return Path(__file__).resolve().parents[4]


def outside_release(path, release):
    path = Path(path).expanduser().resolve()
    release = Path(release).resolve()
    require(path != release and release not in path.parents,
            'Run outputs must be outside the published release')
    require(not path.exists(), 'Refusing to overwrite: %s' % path)
    return path


def bound_file(child, release, record, label):
    path = (child / record['path']).resolve(strict=True)
    require(release in path.parents, 'Asset escaped the release: %s' % label)
    require(path.is_file() and sha256(path) == record['sha256'],
            'Released asset binding mismatch: %s' % label)
    return {'path': str(path), 'sha256': record['sha256']}


def capture_file(seed, split, root, mapping, map_directory):
    if mapping is not None:
        try:
            value = Path(mapping[str(seed)][split])
        except KeyError as error:
            raise RuntimeError('Capture map needs seed 0/1/2, each with train and val') from error
        path = value if value.is_absolute() else map_directory / value
        return path.resolve(strict=True)
    require(root is not None, 'Supply --captures-root or --capture-map')
    root = Path(root).resolve(strict=True)
    candidates = sorted(set(root.rglob('*seed%d*%s*capture*.pt' % (seed, split))))
    direct_paths = [root / ('seed%d' % seed) / ('%s_capture.pt' % split),
                    root / ('seed%d' % seed) / 'capture' / ('fresh_%s_capture.pt' % split)]
    candidates = sorted(set(candidates + [p for p in direct_paths if p.is_file()]))
    require(len(candidates) == 1,
            'Need exactly one seed%d %s capture; use --capture-map for ambiguous names' % (seed, split))
    return candidates[0].resolve(strict=True)


def build_input_lock(release, captures_root=None, capture_map=None, accept_regenerated=False):
    release = Path(release).resolve(strict=True)
    child = release / PACKAGE
    binding = read_json(child / 'bindings/ASSET_BINDINGS.json')
    resolved = {key: bound_file(child, release, value, key)
                for key, value in binding['release_assets'].items()}
    source = Path(resolved['qccr_source']['path']).parents[2]
    require(source == Path(resolved['umr_source']['path']).parents[2],
            'Mapping and value code must use the same source tree')
    protocol = child / 'protocol/FITTING_SOURCE_PROTOCOL.json'
    lock = {'format_version': 1, 'test_accessed': False,
            'protocol': {'path': str(protocol), 'sha256': sha256(protocol)},
            'source': {'path': str(source),
                       'qccr_sha256': resolved['qccr_source']['sha256'],
                       'umr_sha256': resolved['umr_source']['sha256']},
            'val_annotation': resolved['val_annotation'], 'seeds': {},
            'capture_scope': 'regenerated' if accept_regenerated else 'exact_paper_inputs'}
    mapping = read_json(Path(capture_map).resolve(strict=True)) if capture_map else None
    directory = Path(capture_map).resolve().parent if capture_map else None
    reference = None
    for seed in (0, 1, 2):
        entries = {key: resolved['seed%d_%s' % (seed, key)] for key in
                   ('detector_checkpoint', 'formal_umr_head', 'formal_umr_metadata',
                    'qccr_artifact', 'qccr_fits')}
        folds = read_json(Path(entries['qccr_fits']['path']))['fold_by_image']
        require(len(folds) == 249 and set(map(int, folds.values())) == set(range(5)),
                'Invalid released Val fold map')
        if reference is None:
            reference = folds
        require(folds == reference, 'Val fold maps differ across detector seeds')
        for split in ('train', 'val'):
            path = capture_file(seed, split, captures_root, mapping, directory)
            require(path.is_file(), 'Missing capture: %s' % path)
            actual = sha256(path)
            expected = binding['capture_inputs']['seed%d_%s_capture' % (seed, split)]['sha256']
            require(accept_regenerated or actual == expected,
                    'Capture differs from the exact paper input: %s. Use the original optional '
                    'capture or explicitly accept a fresh regenerated capture.' % path)
            entries[split + '_capture'] = {'path': str(path), 'sha256': actual}
        lock['seeds'][str(seed)] = entries
    lock['shared_fold_by_image'] = reference
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=release_root())
    parser.add_argument('--captures-root', type=Path)
    parser.add_argument('--capture-map', type=Path)
    parser.add_argument('--accept-regenerated-captures', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    output = outside_release(args.output, args.release)
    lock = build_input_lock(args.release, args.captures_root, args.capture_map,
                            args.accept_regenerated_captures)
    if not args.preflight_only:
        write_json_atomic(output, lock)
    print('CCVR FITTING-SOURCE INPUTS: PASS; six Train/Val captures; Test excluded')


if __name__ == '__main__':
    main()
