"""YOLOv8-L FP32 directory-backed, batch-8 Test inference."""
import json
from pathlib import Path
import random
import sys


def run(source,checkpoint,annotation,images,output,model_key,seed,epoch,inference):
    sys.path.insert(0,str(source))
    import torch
    import ultralytics
    from common import coco_row,normalize_names
    assert source in Path(ultralytics.__file__).resolve().parents
    if model_key=='yolov10':
        from ultralytics import YOLOv10 as Model
    else:
        from ultralytics import YOLO as Model
    assert inference['batch']==8 and inference['half'] is False
    assert inference['imgsz']==640 and inference['max_det']==300
    assert inference['conf']==.001 and inference['iou']==.7
    assert inference['augment'] is False and inference['rect'] is False and inference['agnostic_nms'] is False
    random.seed(inference['seed']);torch.manual_seed(inference['seed']);torch.cuda.manual_seed_all(inference['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    detector=Model(str(checkpoint))
    metadata=detector.ckpt
    assert metadata['epoch']==epoch and metadata['train_args']['seed']==seed
    assert metadata['train_args']['epochs']==102 and metadata['train_args']['resume'] is False
    names=normalize_names(detector.names)
    declared=json.loads(annotation.read_text())
    assert names==[r['name'] for r in sorted(declared['categories'],key=lambda r:r['id'])]
    byname={Path(row['file_name']).name:row for row in declared['images']}
    seen=set();rows=[]
    for result in detector.predict(source=str(images),stream=True,imgsz=inference['imgsz'],
        batch=inference['batch'],conf=inference['conf'],iou=inference['iou'],max_det=inference['max_det'],
        device=inference['device'],half=inference['half'],augment=inference['augment'],
        agnostic_nms=inference['agnostic_nms'],rect=inference['rect'],verbose=False,save=False,plots=False):
        name=Path(result.path).name
        assert name in byname and name not in seen
        seen.add(name)
        if result.boxes is not None:
            boxes=result.boxes.xyxy.detach().cpu().tolist()
            scores=result.boxes.conf.detach().cpu().tolist()
            classes=result.boxes.cls.detach().cpu().tolist()
            for box,score,cls in zip(boxes,scores,classes):
                rows.append(coco_row(byname[name]['id'],int(cls),box,float(score),len(names)))
    assert seen==set(byname) and rows
    dataset=detector.predictor.dataset
    assert type(dataset).__name__=='LoadImagesAndVideos' and dataset.bs==8
    rows.sort(key=lambda row:(row['image_id'],-row['score'],row['category_id'],row['bbox']))
    output.mkdir(parents=True,exist_ok=False)
    (output/'predictions.json').write_text(json.dumps(rows,separators=(',',':'),allow_nan=False)+'\n')
    (output/'runtime.json').write_text(json.dumps(dict(model=model_key,seed=seed,precision='FP32',
        inference=inference,images=len(seen),detections=len(rows),classes=names,
        loader=dict(source_is_directory=True,effective_batch=dataset.bs,name=type(dataset).__name__),
        torch=torch.__version__,cuda=torch.version.cuda,ultralytics=ultralytics.__version__,
        gpu=torch.cuda.get_device_name(0)),indent=2)+'\n')
    print(model_key,seed,'Test predictions:',len(seen),'images;',len(rows),'detections; actual batch=8',flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder = args.folder.resolve()
    selected = json.loads((folder / f'runs/seed{args.seed}/selection.json').read_text())
    runtime = json.loads((folder / f'results/test_reeval_v1/seed{args.seed}/runtime.json').read_text())
    assert selected['seed'] == runtime['seed'] == args.seed
    checkpoint = folder / f'runs/seed{args.seed}' / selected['checkpoint']
    run(folder / 'source/Ultralytics_official', checkpoint,
        args.dataset / 'annotations/instances_test2017.json', args.dataset / 'test2017',
        args.output, 'yolov8', args.seed, selected['epoch_zero_based'], runtime['inference'])


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
