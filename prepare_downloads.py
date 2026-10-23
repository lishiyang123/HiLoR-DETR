#!/usr/bin/env python3
"""Inventory or build grouped release downloads without modifying the release."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parent
GROUPS = ("code", "results", "weights", "data", "features")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group_for(relative):
    path = PurePosixPath(relative)
    name = path.name.lower()
    parts = {part.lower() for part in path.parts}
    suffix = path.suffix.lower()
    # GCA profiles load these fixed Train statistics during initialization,
    # including checkpoint evaluation. Keep them with runnable source, not results.
    if suffix == '.json' and name.endswith('_difficulty_cdf_train640.json') and 'ufd' in parts:
        return 'code'
    if suffix in {'.zip', '.7z', '.rar', '.gz', '.tar'} and parts.intersection(
            {'datasets', 'data', 'official_assets_inbox'}):
        return 'data'
    if name in {'predictions.json', 'predictions.json.gz'} or name.endswith(('_predictions.json', '_predictions.json.gz')):
        return 'results'
    if 'ccvr_fitting_sources' in parts and name in {'oof_metrics.json', 'mapping_fits.json'}:
        return 'results'
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
        if parts.intersection({"datasets", "data", "images", "train2017", "val2017", "test2017"}):
            return "data"
    if "features" in parts and suffix in {".pt", ".pth"}:
        return "features"
    if suffix == ".pt" and ("train_capture" in name or "val_capture" in name):
        return "features"
    if suffix in {".pt", ".pth", ".ckpt", ".onnx", ".safetensors"}:
        if "eval" in name or "capture" in name or "targets" in name or "eval" in parts:
            return "results"
        return "weights"
    # Keep runnable source and configuration with the lightweight code archive,
    # even when they live inside an analysis/result subpackage.
    if suffix in {".py", ".sh", ".bat", ".ps1", ".yml", ".yaml"}:
        return "code"
    if suffix in {".csv", ".tsv", ".npz", ".npy", ".log"}:
        return "results"
    if parts.intersection({"results", "outputs", "paper_tables", "training_records",
                           "predictions", "evaluations", "evaluations_gca",
                           "evidence", "analysis", "reports", "captures", "artifacts"}):
        if suffix not in {".md", ".txt"}:
            return "results"
    return "code"


def catalog(root):
    manifest = root / "RELEASE_MANIFEST.sha256.json"
    digest = sha(manifest)
    rows = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    rows = rows + [{"path": manifest.name, "bytes": manifest.stat().st_size, "sha256": digest}]
    if len({row["path"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate path in release manifest")
    groups = {group: [] for group in GROUPS}
    for row in sorted(rows, key=lambda item: item["path"]):
        relative = PurePosixPath(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe release path: " + row["path"])
        path = root / row["path"]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"]:
            raise RuntimeError("Missing or size-changed release input: " + row["path"])
        groups[group_for(row["path"])].append(row)
    version = "hilordetr-" + digest[:16]
    def group_record(group, files):
        unique = {(r["bytes"], r["sha256"]) for r in files}
        total = sum(r["bytes"] for r in files)
        payload = sum(size for size, _ in unique)
        return {"archive": f"{version}-{group}.tar.gz", "file_count": len(files),
                "uncompressed_bytes": total, "unique_payload_bytes": payload,
                "duplicate_payload_bytes": total - payload, "files": files}

    return {"schema": "hilordetr.grouped-downloads.v1", "release_id": version,
            "release_manifest_sha256": digest, "published_url": None,
            "verification": "Inventory uses the release manifest; archive build verifies each input SHA while streaming.",
            "groups": {group: group_record(group, files) for group, files in groups.items()}}


class HashingReader:
    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        data = self.handle.read(size)
        self.digest.update(data)
        return data


def build_group(root, output, record):
    target = output / record["archive"]
    part = output / (target.name + ".part")
    seen_paths = set()
    for row in record["files"]:
        relative = PurePosixPath(row["path"])
        if (relative.is_absolute() or ".." in relative.parts or
                "\\" in row["path"] or row["path"] in seen_paths):
            raise ValueError("Unsafe or repeated archive path: " + row["path"])
        path = root / row["path"]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"]:
            raise RuntimeError("Missing or size-changed release input: " + row["path"])
        seen_paths.add(row["path"])
    written = {}
    linked_files = 0
    saved_payload_bytes = 0
    with part.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for row in record["files"]:
                path = root / row["path"]
                info = tarfile.TarInfo(row["path"])
                info.size = row["bytes"]
                info.mode = path.stat().st_mode & 0o777
                info.mtime = 0
                identity = (row["bytes"], row["sha256"])
                if identity in written:
                    # Verify the duplicate's actual bytes, not just its manifest.
                    if sha(path) != row["sha256"]:
                        raise RuntimeError("Release input changed: " + row["path"])
                    info.type = tarfile.LNKTYPE
                    info.linkname = written[identity]
                    info.size = 0
                    archive.addfile(info)
                    linked_files += 1
                    saved_payload_bytes += row["bytes"]
                    continue
                with path.open("rb") as handle:
                    reader = HashingReader(handle)
                    archive.addfile(info, reader)
                    if reader.digest.hexdigest() != row["sha256"]:
                        raise RuntimeError("Release input changed: " + row["path"])
                written[identity] = row["path"]
    if target.exists():
        raise FileExistsError(target)
    part.rename(target)
    record["archive_bytes"] = target.stat().st_size
    record["archive_sha256"] = sha(target)
    record["hardlinked_files"] = linked_files
    record["deduplicated_payload_bytes"] = saved_payload_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New external directory for the inventory/archives")
    parser.add_argument("--build", action="store_true", help="Actually compress the selected groups; may take substantial CPU/disk")
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=["code", "results", "weights"])
    args = parser.parse_args()
    root = ROOT.resolve()
    data = catalog(root)
    if args.build and args.output is None:
        parser.error("--build requires --output")
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if output == root or root in output.parents:
            parser.error("Output must be outside the frozen release")
        # Never overwrite a previous download or plan.
        output.mkdir(parents=True, exist_ok=False)
        if args.build:
            for group in dict.fromkeys(args.groups):
                print("Building", group, flush=True)
                build_group(root, output, data["groups"][group])
        (output / "DOWNLOAD_MANIFEST.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print("Inventory:", output / "DOWNLOAD_MANIFEST.json")
    print("Release ID:", data["release_id"])
    for group, record in data["groups"].items():
        print(f"{group}: {record['file_count']} files, {record['uncompressed_bytes'] / 1e6:.3f} MB logical, "
              f"{record['unique_payload_bytes'] / 1e6:.3f} MB unique payload")
    print("No public URL created. Upload/deposit requires an author-chosen destination.")


if __name__ == "__main__":
    main()
