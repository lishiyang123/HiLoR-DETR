"""Fetch exact third-party comparison weights from their original provider."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--check-only', action='store_true', help='Do not download any files')
    args = parser.parse_args()
    root = args.root.resolve()
    records = json.loads((root / 'EXTERNAL_ASSETS.json').read_text())['files']
    for row in records:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe asset path')
        target = root / relative
        if target.exists():
            if target.is_symlink() or target.stat().st_size != row['bytes'] or sha(target) != row['sha256']:
                raise RuntimeError('Existing asset differs; not overwritten: ' + row['path'])
            print(row['model'] + ': SHA-256 PASS')
            continue
        if args.check_only:
            print(row['model'] + ': DOWNLOAD REQUIRED from ' + row['url'])
            continue
        if not row['url'].startswith('https://zenodo.org/records/17447590/files/'):
            raise ValueError('Unexpected original provider URL')
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + '.part')
        # Do not overwrite a partial file from another process or interrupted download.
        with partial.open('xb') as output:
            request = urllib.request.Request(row['url'], headers={'User-Agent': 'HiLoR-DETR-reproduction/1.0'})
            with urllib.request.urlopen(request, timeout=120) as response:
                for block in iter(lambda: response.read(4 * 1024 * 1024), b''):
                    output.write(block)
        if partial.stat().st_size != row['bytes'] or sha(partial) != row['sha256']:
            raise RuntimeError('Downloaded file failed SHA-256; .part retained for inspection')
        if target.exists():
            raise RuntimeError('Destination appeared during download; .part retained')
        partial.rename(target)
        print(row['model'] + ': DOWNLOADED / SHA-256 PASS')


if __name__ == '__main__':
    main()
