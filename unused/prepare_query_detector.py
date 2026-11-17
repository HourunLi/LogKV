"""Download only the GLiClass model artifacts as ordinary files; never install or change packages.

Online: python unused/prepare_query_detector.py
Offline: python unused/prepare_query_detector.py --verify
The default directory is hf_cache/gliclass-instruct-edge under the ModelArts bucket.
HTTPS certificate verification is disabled by default; --verify-ssl enables it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import ssl
import urllib.parse
import urllib.request
from pathlib import Path

MODEL_ID = "knowledgator/gliclass-instruct-edge-v1.0"
MODEL_REVISION = "727be8a417f6a7718e591b025e07054c146d8139"
DEFAULT_MODEL_DIR = Path("/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/gliclass-instruct-edge")
REQUIRED_FILES = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}


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


def ssl_context(verify_ssl=False):
    if verify_ssl:
        return ssl.create_default_context()
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--verify-ssl", action="store_true", help="enable HTTPS certificate verification")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.out.expanduser().absolute()
    if args.verify:
        verify(root)
        return
    root.mkdir(parents=True, exist_ok=True)
    regular_tree(root)
    context = ssl_context(args.verify_ssl)
    endpoint = args.endpoint.rstrip("/")
    request = urllib.request.Request(
        f"{endpoint}/api/models/{MODEL_ID}/revision/{urllib.parse.quote(args.revision, safe='')}?blobs=true",
        headers={"User-Agent": "LogKV-query-detector-bundle"},
    )
    with urllib.request.urlopen(request, timeout=60, context=context) as response:
        info = json.load(response)
    revision = info["sha"]
    files = {entry["rfilename"]: entry for entry in info["siblings"] if entry["rfilename"] in REQUIRED_FILES}
    missing = REQUIRED_FILES - files.keys()
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
        with urllib.request.urlopen(url, timeout=300, context=context) as src, temp.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        if expected and sha256(temp) != expected:
            raise ValueError(f"download checksum mismatch: {name}")
        # Copy rather than rename/link: also works on ordinary object-store mounts.
        shutil.copyfile(temp, target)
        temp.unlink()
    regular_tree(root)
    hashes = {name: sha256(root / name) for name in files}
    (root / "bundle.json").write_text(json.dumps(
        {"model_id": MODEL_ID, "revision": revision, "sha256": hashes}, indent=2) + "\n")
    verify(root)


if __name__ == "__main__":
    main()
