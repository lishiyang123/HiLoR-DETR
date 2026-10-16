# Download preparation and fixed versions

Source repository: <https://github.com/lishiyang123/HiLoR-DETR>.
Versioned code/results/weights and optional data/features:
<https://github.com/lishiyang123/HiLoR-DETR/releases>.

The attached `DOWNLOAD_MANIFEST.json` records the exact contents and checksums
of independently extractable archive parts. Extract the parts needed for the
selected workflow into one directory, preserving relative paths. Numbered
archives are complete tar files; do not concatenate them. Publication splits
large weight/feature groups into smaller attachments. No DOI is claimed.

The authors' original code is Apache-2.0 (`LICENSE`). Third-party code, datasets
and weights retain their own terms; see `THIRD_PARTY_LICENSES.md` and
`DATA_AVAILABILITY.md`. Use these same links in the root README and manuscript.

## Before public upload

- Include `THIRD_PARTY_LICENSES.md`, the root `LICENSE-*.txt` full texts and the
  original license/copyright notices retained in third-party source directories.
  GPL/AGPL-covered components and modifications retain their respective terms.
- The archive groups below describe local reproduction assets, not a declaration
  that every asset has been cleared for public redistribution.
- Public YOLOv8-L / YOLO11-L checkpoints are download-only external inputs.
  They are excluded from the release manifest and all archive groups, including
  after local restoration. `EXTERNAL_ASSETS.json` and `fetch_external_assets.py`
  provide original-provider URLs and exact SHA-256 checks. Do not upload these
  two checkpoint files as new mirrors.
- Retain the published dataset attribution, license and conversion notices for
  optional dataset downloads. Check model-specific terms before uploading other
  third-party checkpoints. A link to a publicly downloadable file is not itself
  an additional redistribution grant.

## Fixed content identifier

`RELEASE_MANIFEST.sha256.json` identifies every current release file by relative
path, size and SHA-256. `prepare_downloads.py` derives the release ID
`hilordetr-<first 16 characters of the manifest SHA-256>` from that completed
manifest. Its download inventory also records the full manifest digest. Changing
any included asset and refreshing the manifest produces a different release ID.
Record this ID alongside the actual download URL when publishing; it is not a
claim that a Git tag or DOI has already been registered.

## Archive groups

| Group | Contents | Use |
|---|---|---|
| `code` | Source, configurations, installation/data instructions, navigation, manifests | Lightweight executable/documentation download |
| `results` | Predictions, evaluation arrays, numerical tables, logs, analysis outputs and the common Test candidate pool | Offline result checking; no detector training |
| `weights` | Selected detectors, initializers, fitted value heads and comparison heads | Checkpoint inference or training initialization |
| `data` | Bundled dataset images | Optional; see `DATA_AVAILABILITY.md` for originals, licensing and exact reconstruction |
| `features` | Regenerable Train/Val feature captures | Optional; not needed for inference with already fitted CCVR |

This is transport grouping, not deletion or a change to an experiment. Every file
belongs to exactly one group. Archive members retain the original relative paths;
extract the requested groups into **the same directory**. No extra experiment
folder renaming or weight conversion takes place. Required mapping coefficients,
value-head weights, and shared Test candidates retain their existing bindings.

Within each archive, byte-identical files use standard tar hard links to an
earlier member. Every input is still checked against its own SHA-256 before a
link is written. Extraction recreates every original path and its exact bytes;
there is no cross-archive link dependency. This avoids duplicate payloads without
changing runtime paths, configurations or scientific assets. Catalog sizes report
both logical file bytes and unique payload bytes.

## Prepare a download inventory (no compression)

From the aggregate release root, using Python 3.10+:

```bash
python -B prepare_downloads.py
python -B prepare_downloads.py --output /your/new/download_plan
```

The first command prints the release ID and exact uncompressed group sizes. The
second additionally writes `DOWNLOAD_MANIFEST.json` outside the release. It does
not create archives, train, fit, infer, or alter the frozen package. The catalog
contains the group membership and original file digests; archive checksums are
added only after an actual build.

## Build archives when a publication destination is ready

```bash
python -B prepare_downloads.py --build --groups code results weights --output /your/new/submission_downloads
```

Add `data features` to `--groups` only if those optional downloads are wanted.
Compression consumes CPU and additional disk space; this is not a quick GPU
check. The destination must be new and outside the release. All input file hashes
are checked while archiving, archives contain no absolute/parent paths, and the
release is never modified. The download manifest records the actual compressed
size and SHA-256 of each built archive. Files ending in `.part` indicate an
incomplete build and must not be published.

Upload only assets approved by the public-upload checklist above, together with
an inventory of the assets actually supplied. A local complete archive is not
automatically a public-upload candidate. If an asset is obtained from its
original provider instead, keep the exact filename, source and expected digest
in the download instructions. Review the inventory before upload; provide a
smaller or split optional download if the host imposes size limits.

## Assemble and verify

This is a Linux/CUDA reproduction package. Extract into a case-sensitive Linux
filesystem (or a case-sensitive WSL workspace): upstream and result directories
may contain case-distinct filenames. Do not unpack and repack the full release
through a normal case-insensitive Windows directory, which can merge those names.

Extract groups into a single clean directory, preserving their member paths.
Recreate missing dataset images using the pinned original data and the released
data-preparation instructions. Include optional feature captures if a full
release-wide hash check, rather than checkpoint inference, is required. Then:

```bash
python -B verify_release.py verify
```

The full verifier checks all 14 packages and deliberately reports missing files
in a partial download. This is not an inference failure: install the required
groups/data for the selected task using `REPRODUCTION_QUICKSTART.md`. Retraining
remains a statistical reproduction; archived-result verification and checkpoint
inference are distinct workflows.

## External comparison input restoration

Saved results and numerical tables do not require external model downloads.
To rerun YOLOv8-L / YOLO11-L inference, run `python -B fetch_external_assets.py`
at the release root. It restores only the two cataloged original-provider
files and verifies their bytes before installing them at the expected paths.
The release verifier accepts these optional restored files only when their
sizes and SHA-256 values match; they remain excluded from archive inventories.

## v1.0.1 download assembly

Use the v1.0.1 code and results archives together. Its `DOWNLOAD_MANIFEST.json`
also lists unchanged weight/data/feature attachments at their exact v1.0.0 URLs;
these bytes are shared, not alternative experiment results. Download each selected
archive from its `download_url`, verify its SHA-256, and extract into the same
clean directory. Do not overlay the old code/results archives. Source photos
are in the unchanged optional BTC dataset files. Only qualitative figure evidence
and navigation changed; detector/fitted weights and quantitative tables did not.
