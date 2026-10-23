"""Profile the selected seed-0 deployed graph; no latency or Test AP measurement."""
import argparse
import sys
from pathlib import Path

from scoped_common import bindings, load_detector, make_cfg, parameters, roots, sha256, thread_setup, trim_retired_branches, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--task', choices=['formal'], required=True)
    parser.add_argument('--asset-root', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    thread_setup()
    import torch
    from benchmark_runtime import profile_flops
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Complexity profiling requires the configured RTX 4090 environment')
    bound = bindings(args.asset_root)
    row = bound['models'][args.model]
    sys.path.insert(0, str(roots(args.asset_root)['VENDOR_ROOT']))
    output = args.output / args.model
    if output.exists():
        raise FileExistsError(output)
    assert sha256(row['checkpoint']) == row['checkpoint_sha256']
    sys.path.insert(0, row['source'])
    deployment = {}
    post_parameters = 0
    if row['flavor'] in ('engine', 'src'):
        cfg, row, bound, config = make_cfg(args.model, args.asset_root, args.output/'effective_configs')
        detector, post, weight_source = load_detector(cfg, row)
        detector = detector.deploy().float().eval()
        deployment = dict(method='official deploy', retired_branches=trim_retired_branches(detector),
                          weight_source=weight_source)
        post_parameters = parameters(post)
    elif row['flavor'] == 'yolov9':
        from models.yolo import Model
        from utils.general import non_max_suppression
        from benchmark_yolov9 import load_checkpoint, official_reparameterize, verify_deployment_equivalence
        dual = load_checkpoint(Path(row['checkpoint']))
        detector, conversion = official_reparameterize(Model, Path(row['source']), dual)
        detector, equivalent = verify_deployment_equivalence(dual, detector, non_max_suppression,
            Path(bound['data']['val_annotation']), Path(bound['data']['root'])/'val2017')
        deployment = dict(method='official YOLOv9 reparameterization', conversion=conversion,
                          selected_deployment=equivalent['selected_deployment'],
                          fusion_gate=equivalent['fusion_coco_metric_equivalence'])
        del dual
    else:
        if args.model == 'yolov10':
            from ultralytics import YOLOv10 as Model
        else:
            from ultralytics import YOLO as Model
        detector = Model(row['checkpoint']).model.float().eval()
        try:
            detector = detector.fuse(verbose=False)
        except TypeError:
            detector = detector.fuse()
        deployment = dict(method='official Conv-BN fusion')
    detector = detector.cuda().float().eval()
    with torch.inference_mode():
        flops = profile_flops(detector)
    assert flops['profiler_version'] == '0.3.2'
    count = parameters(detector)
    scalars = row.get('fitted_scalars', 0)
    report = dict(status='MEASURED', model=row['label'], seed=0, checkpoint=row['checkpoint'],
        checkpoint_sha256=row['checkpoint_sha256'], source=row['source'], deployment=deployment,
        input_shape=[1,3,640,640],
        parameters=dict(detector=count,postprocessor=post_parameters,fitted_scalars=scalars,
                        total_M=(count+post_parameters+scalars)/1e6), flops=flops,
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)))
    write_json(output/'MEASUREMENT.json', report)
    print(args.model, report['parameters']['total_M'], flops['forward_GFLOPs'], flush=True)


if __name__ == '__main__':
    main()
