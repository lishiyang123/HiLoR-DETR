# HiLoR-DETR reproduction materials

This release provides the implementation, configurations and numerical evidence
for the CDW1630 and BTC experiments. HiLoR-DETR combines training-time
GCA with inference-time
competition-conditioned value reranking (CCVR).

## Start here

- `REPRODUCTION_QUICKSTART.md`: environments, data preparation and commands.
- `ENVIRONMENT_MATRIX.md`: the environments used by each experiment.
- `DATA_AVAILABILITY.md`: dataset sources, splits and preparation instructions.
- `DISTRIBUTION.md`: separate code, results, weights and optional downloads.

Choose the level of reproduction needed:

1. **Check reported results:** use the saved predictions, evaluation arrays and
   unrounded tables. This does not require detector training or a GPU.
2. **Repeat inference:** obtain the selected detector checkpoint and its fitted
   CCVR assets, then run the host-specific evaluation entry.
3. **Train and fit again:** run the explicit training, Train/Val capture, fitting
   and evaluation stages in `reproduce_pipeline.py`, using a new work directory.

Fresh training is a statistical reproduction, not a promise of bitwise-identical
results on arbitrary hardware. Existing-paper replay and fresh training are
separate workflows; neither overwrites the supplied results.

## Package navigation

| Area | Directory |
| --- | --- |
| Main-host implementation and CDW1630 results | `main_host/` |
| Four-configuration system comparison, including Native + CCVR | `main_host/paper_tables/three_seed/` |
| BTC generalization | `generalization/btc/` |
| Comparison detectors | `benchmarks/` |
| RT-DETR-R18 second host | `second_host/` |
| Complexity, inference latency and training cost | `efficiency/` |
| Parameter sensitivity | `sensitivity/` |
| Source numbers for manuscript figures | `numerical_evidence/` |
| Three-seed paired Bootstrap | `bootstrap/` |
| Dataset statistics, condition groups and per-class analysis | `dataset_analysis/` |
| Controlled candidate-ranking comparison | `ranking_comparison/` |

The directory names are also used by configuration and asset references. Extract
the selected downloads into one directory, preserving these relative paths.
Generated publication graphics are not required; their numerical evidence is
included. Use a case-sensitive Linux/WSL filesystem when extracting the release.

Current photographic examples (F1, F2 and F7) use BTC seed 0; see
`numerical_evidence/README.md` for the exact cases and numerical replay.

The main CDW1630 package contains Native DEIM, Native + CCVR, GCA and complete
HiLoR-DETR for seeds 0/1/2. Their twelve formal evaluations and reusable input locations are
listed in `main_host/manifests/CDW1630_THREE_SEED_RELEASE.json`. See
`main_host/README.md` for shared predictions and optional raw captures.

## Method source organization

The shared CDW1630/BTC implementation is under
`main_host/source/FINAL_CODE/engine/deim/`:

| Stage | Functional file |
| --- | --- |
| GCA: auxiliary geometric supervision | `auxiliary_geometry.py` |
| GCA: classification-role adjustment | `classification_role.py` |
| CCVR: competition-conditioned mapping | `competition_mapping.py` |
| CCVR: candidate-value estimation | `candidate_value.py` |

These file boundaries separate implementation responsibilities within the two
stages; they do not define four independent proposed modules. Some configuration,
state-dictionary and fitted-artifact identifiers use `ufd`, `lcct`, `qccr` or
`umr`. Those identifiers are retained for compatibility. The second-host README
maps the corresponding RT-DETR source files to these same responsibilities.

## License

The authors' original code is released under **Apache-2.0** where applicable;
see `LICENSE`. Third-party components and modifications covered by GPL/AGPL
retain those licenses; the root license does not override them. Datasets,
pretrained weights and other third-party assets retain their respective terms.
`THIRD_PARTY_LICENSES.md` is the central source/license guide and links to the
bundled full texts. Retain upstream copyright and attribution notices.

## Code and asset availability

Public source: [HiLoR-DETR](https://github.com/lishiyang123/HiLoR-DETR).
Versioned downloads: [Releases](https://github.com/lishiyang123/HiLoR-DETR/releases).

Choose the release cited by the manuscript. Use its Git tag, automatic "Source
code" archive, or attached code archive, together with the matching result and
asset files listed in that release's `DOWNLOAD_MANIFEST.json`. The source includes
both dataset-specific Train difficulty CDFs required by the main-host GCA profiles.
Checkpoint inference also requires the selected weights, fitted CCVR assets and
dataset inputs.

Code, formal results, model weights and optional data/features are separate
downloads. `DOWNLOAD_MANIFEST.json` attached to each release maps archive parts
to their files. Each archive part can be extracted independently
into the same directory; numbered parts are not byte fragments to concatenate.
Obtain the selected detector and its mapping/value-head assets together.
See `DISTRIBUTION.md` for assembly and scope. No DOI is claimed.

## Comparison weights

YOLOv8-L and YOLO11-L use bundled three-seed trained weights and matching
Ultralytics source under `benchmarks/01_YOLO_SERIES/YOLOv8_L_TRAINED/` and
`benchmarks/01_YOLO_SERIES/YOLOv11_L_TRAINED/`.
