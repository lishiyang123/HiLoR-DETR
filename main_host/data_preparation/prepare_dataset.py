"""Rebuild the published CDW1630/BTC COCO data from the original YOLO/VOC files."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[1]
CDW_CATEGORIES = [dict(id=i, name=n, supercategory='none') for i, n in enumerate(
    ['Brick', 'Concrete', 'Plastic Bottle', 'Reinforcement', 'timber'])]
BTC_CATEGORIES = [dict(id=i, name=n, supercategory='construction_waste') for i, n in enumerate(
    ['brick', 'concrete', 'tile'])]
BTC_DOI = 'https://doi.org/10.17632/24d45pf8wm.1'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def raw_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Missing or unsafe raw file: {relative}')
    return path


def image_size(path):
    from PIL import Image
    with Image.open(path) as image:
        return image.size


def validate_box(box, width, height, context):
    x, y, w, h = box
    if not all(math.isfinite(v) for v in box) or w <= 0 or h <= 0:
        raise ValueError(f'Non-finite or non-positive box: {context}')
    if x < -1e-6 or y < -1e-6 or x + w > width + 1e-6 or y + h > height + 1e-6:
        raise ValueError(f'Box outside image; no silent clipping: {context}')


def yolo_annotations(path, width, height):
    annotations = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f'Expected five YOLO fields: {path}:{line_number}')
        values = [float(v) for v in fields]
        category, cx, cy, w, h = values
        if not all(math.isfinite(v) for v in values) or category != int(category) or not 0 <= category < 5:
            raise ValueError(f'Invalid class or number: {path}:{line_number}')
        box = [round((cx - w / 2) * width, 4), round((cy - h / 2) * height, 4),
               round(w * width, 4), round(h * height, 4)]
        validate_box(box, width, height, f'{path}:{line_number}')
        annotations.append(dict(category_id=int(category), bbox=box,
                                area=round(box[2] * box[3], 4), iscrowd=0, segmentation=[]))
    return annotations, 0


def voc_annotations(path, width, height):
    xml = ET.parse(path).getroot()
    if int(xml.findtext('size/width')) != width or int(xml.findtext('size/height')) != height:
        raise ValueError(f'XML/image size mismatch: {path}')
    classes = {c['name']: c['id'] for c in BTC_CATEGORIES}
    annotations, empty = [], 0
    for index, obj in enumerate(xml.findall('object'), 1):
        if len(obj) == 0 and not (obj.text or '').strip():
            empty += 1
            continue
        name = (obj.findtext('name') or '').strip()
        if name not in classes:
            raise ValueError(f'Unknown or incomplete VOC class: {path} object {index}')
        bound = obj.find('bndbox')
        if bound is None:
            raise ValueError(f'Missing VOC bounding box: {path} object {index}')
        xmin, ymin, xmax, ymax = [float(bound.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
        # Preserve the experiment's declared VOC convention, not an inclusive +1 variant.
        box = [xmin - 1.0, ymin - 1.0, xmax - xmin, ymax - ymin]
        validate_box(box, width, height, f'{path} object {index}')
        annotations.append(dict(category_id=classes[name], bbox=box,
                                area=box[2] * box[3], iscrowd=0))
    return annotations, empty


def encoded(dataset, payload):
    if dataset == 'cdw1630':
        return json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
    return (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def split_membership(manifest):
    training = manifest['splits']['train']['images']
    validation = manifest['splits']['val']['images']
    stems = [Path(row['file_name']).stem for row in training + validation]
    if len(stems) != 350 or len(set(stems)) != 350:
        raise ValueError('BTC internal split must contain 350 distinct original training images')
    ranked = sorted(stems, key=lambda stem: hashlib.sha256(
        f'BTC_INTERNAL_280_70_V1|seed=0|stem={stem}'.encode()).hexdigest())
    if {Path(row['file_name']).stem for row in validation} != set(ranked[:70]):
        raise ValueError('BTC manifest differs from the declared fixed 280/70 split')


def rebuild(dataset, raw_root, manifest):
    if dataset == 'btc':
        split_membership(manifest)
    results, image_sources, ignored = {}, {}, {}
    for split, specification in manifest['splits'].items():
        images, annotations, sources = [], [], []
        empty = 0
        annotation_id = specification['annotation_id_base'] + 1
        for position, row in enumerate(specification['images'], 1):
            image = raw_path(raw_root, row['raw_image'])
            annotation = raw_path(raw_root, row['raw_annotation'])
            for path, expected in [(image, row['image_sha256']), (annotation, row['annotation_sha256'])]:
                if sha256(path) != expected:
                    raise ValueError(f'Raw file differs from the published source version: {path}')
            width, height = image_size(image)
            image_id = specification['image_id_base'] + position
            record = dict(id=image_id, file_name=row['file_name'], width=width, height=height)
            if dataset == 'btc':
                record.update(source_xml=annotation.name, source_split=specification['source_split'])
            images.append(record)
            objects, skipped = (yolo_annotations if dataset == 'cdw1630' else voc_annotations)(annotation, width, height)
            empty += skipped
            for obj in objects:
                annotations.append(dict(id=annotation_id, image_id=image_id, **obj))
                annotation_id += 1
            sources.append((image, row['file_name']))
        payload = dict(info=specification['info'], licenses=[] if dataset == 'cdw1630' else [dict(id=1, name='CC BY 4.0')],
                       images=images, annotations=annotations, categories=CDW_CATEGORIES if dataset == 'cdw1630' else BTC_CATEGORIES)
        result = encoded(dataset, payload)
        if hashlib.sha256(result).hexdigest() != specification['coco_sha256']:
            raise ValueError(f'Rebuilt {split} annotation differs from the published COCO annotation')
        if len(annotations) != specification['annotation_count'] or empty != specification['ignored_empty_objects']:
            raise ValueError(f'Unexpected annotation/empty-node count in {split}')
        results[specification['annotation_file']] = result
        image_sources[specification['image_directory']] = sources
        ignored[split] = empty
    if dataset == 'btc':
        first = json.loads(results['instances_test1.json'])
        second = json.loads(results['instances_test2.json'])
        specification = manifest['combined_test']
        payload = dict(info=specification['info'], licenses=first['licenses'],
                       images=first['images'] + second['images'], annotations=first['annotations'] + second['annotations'],
                       categories=BTC_CATEGORIES)
        result = encoded(dataset, payload)
        if hashlib.sha256(result).hexdigest() != specification['coco_sha256']:
            raise ValueError('Combined BTC Test annotation differs from the published annotation')
        results[specification['annotation_file']] = result
        image_sources['test_combined'] = image_sources['test1'] + image_sources['test2']
    return results, image_sources, ignored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('cdw1630', 'btc'), required=True)
    parser.add_argument('--raw-root', type=Path, required=True, help='Extracted supplier data root; see README')
    parser.add_argument('--output', type=Path, help='New directory outside the release; never overwrites')
    parser.add_argument('--verify-only', action='store_true', help='Rebuild in memory and check annotation hashes for the chosen dataset, with no output files')
    args = parser.parse_args()
    raw_root = args.raw_root.resolve(strict=True)
    if not raw_root.is_dir():
        parser.error('--raw-root must be a directory')
    output = args.output.resolve() if args.output else None
    if args.verify_only and output is not None:
        parser.error('--verify-only cannot be combined with --output')
    if not args.verify_only and output is None:
        parser.error('Supply --output or --verify-only')
    if output is not None and (output.exists() or output == RELEASE or RELEASE in output.parents
                               or output == raw_root or raw_root in output.parents):
        parser.error('Choose a new output outside the release and raw data directories')
    manifest = json.loads((HERE / 'source_manifest.json').read_text(encoding='utf-8'))[args.dataset]
    try:
        results, images, ignored = rebuild(args.dataset, raw_root, manifest)
    except (ValueError, OSError, ET.ParseError) as error:
        parser.exit(1, f'DATA PREPARATION FAILED: {error}\n')
    if output is not None:
        output.mkdir(parents=True)
        (output / 'annotations').mkdir()
        for filename, contents in results.items():
            (output / 'annotations' / filename).write_bytes(contents)
        for directory, sources in images.items():
            target = output / directory
            target.mkdir()
            for source, filename in sources:
                if (target / filename).exists():
                    raise ValueError(f'Duplicate destination image: {directory}/{filename}')
                shutil.copy2(source, target / filename)
        if args.dataset == 'cdw1630':
            names = [c['name'] for c in CDW_CATEGORIES]
            (output / 'data.yaml').write_text('path: .\ntrain: train2017\nval: val2017\ntest: test2017\nnc: 5\nnames: ' + repr(names) + '\n', encoding='utf-8')
        report = dict(dataset=args.dataset, annotation_sha256={name: hashlib.sha256(content).hexdigest() for name, content in results.items()},
                      image_counts={name: len(rows) for name, rows in images.items()}, ignored_empty_voc_objects=ignored,
                      images_resized_or_reencoded=False)
        (output / 'PREPARATION_SUMMARY.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'{args.dataset.upper()} RAW-TO-COCO REBUILD: PASS')
    for filename, contents in results.items():
        print(f'{filename}: {hashlib.sha256(contents).hexdigest()}')
    print('Published annotation bytes: identical; original split and image bytes preserved')
    print('No model training, fitting, inference or GPU required')


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
