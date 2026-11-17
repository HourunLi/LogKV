"""Download a relocatable GLiClass bundle as ordinary files, optionally with offline wheels.

Online: python unused/prepare_query_detector.py --out /data/gliclass-edge --wheels
Offline: python unused/prepare_query_detector.py --out /data/gliclass-edge --verify
Use the same Python/platform on the online and offline machines when preparing wheels.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

MODEL_ID = "knowledgator/gliclass-instruct-edge-v1.0"
MODEL_REVISION = "727be8a417f6a7718e591b025e07054c146d8139"
VERSIONS = {"gliclass": "0.1.15", "transformers": "4.57.3"}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def regular_tree(root):
    if root.is_symlink():
        raise ValueError(f"symlink directory is not supported: {root}")
    for p in root.rglob("*"):
        if p.is_symlink():
            raise ValueError(f"bundle contains a symlink: {p}")


def verify(root):
    regular_tree(root)
    manifest = json.loads((root / "bundle.json").read_text())
    for name, expected in manifest["sha256"].items():
        p = root / name
        if not p.is_file() or sha256(p) != expected:
            raise ValueError(f"missing or corrupt bundle file: {p}")
    print(f"Verified {len(manifest['sha256'])} ordinary files, revision {manifest['revision']}")


def wheel_requirements():
    """Export the installed runtime dependency closure, reusing the target's existing PyTorch."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    for name, expected in VERSIONS.items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"prepare wheels with {name}=={expected}")
    pending, found = list(VERSIONS), {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name == "torch" or name in found:
            continue
        dist = importlib.metadata.distribution(name)
        found[name] = dist.version
        for raw in dist.requires or ():
            req = Requirement(raw)
            if req.marker is None or req.marker.evaluate({"extra": ""}):
                if canonicalize_name(req.name) != "torch":
                    if not req.specifier.contains(importlib.metadata.version(req.name), prereleases=True):
                        raise ValueError(f"installed dependency does not satisfy {raw}")
                    pending.append(req.name)
    return [f"{name}=={version}" for name, version in sorted(found.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--wheels", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.out.expanduser().absolute()
    if args.verify:
        verify(root)
        return
    root.mkdir(parents=True, exist_ok=True)
    regular_tree(root)
    endpoint = args.endpoint.rstrip("/")
    request = urllib.request.Request(
        f"{endpoint}/api/models/{MODEL_ID}/revision/{urllib.parse.quote(args.revision, safe='')}?blobs=true",
        headers={"User-Agent": "LogKV-query-detector-bundle"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        info = json.load(response)
    revision = info["sha"]
    wanted = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt"}
    files = {entry["rfilename"]: entry for entry in info["siblings"] if entry["rfilename"] in wanted}
    missing = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"} - files.keys()
    if missing:
        raise ValueError(f"model repository missing required artifacts: {sorted(missing)}")
    for name, entry in files.items():
        target = root / name
        expected = entry.get("lfs", {}).get("sha256")
        if expected and target.is_file() and sha256(target) == expected:
            print(f"Verified existing {name}", flush=True)
            continue
        print(f"Downloading {name}", flush=True)
        url = f"{endpoint}/{MODEL_ID}/resolve/{revision}/{name}?download=true"
        temp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url, timeout=300) as src, temp.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        if expected and sha256(temp) != expected:
            raise ValueError(f"download checksum mismatch: {name}")
        # Copy rather than rename/link: also works on ordinary object-store mounts.
        shutil.copyfile(temp, target)
        temp.unlink()
    if args.wheels:
        requirements = root / "detector-requirements.txt"
        requirements.write_text("\n".join(wheel_requirements()) + "\n")
        wheels = root / "wheels"
        wheels.mkdir(exist_ok=True)
        subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
                        "--dest", str(wheels), "-r", str(requirements)], check=True)
    regular_tree(root)
    hashes = {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob("*")
              if p.is_file() and p.name != "bundle.json"}
    (root / "bundle.json").write_text(json.dumps(
        {"model_id": MODEL_ID, "revision": revision, "packages": VERSIONS, "sha256": hashes}, indent=2) + "\n")
    verify(root)


if __name__ == "__main__":
    main()
