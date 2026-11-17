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
import time
import urllib.parse
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


def fetch(session, url, *, target=None, expected_hash=None, expected_size=None, verify_ssl=False, retries=5):
    """Use the existing downloaders' requests streaming/SSL policy, refreshing expired redirects."""
    import requests

    label = target.name if target is not None else "model metadata"
    failure = "unknown error"
    for attempt in range(retries + 1):
        address = url
        if attempt:
            # Retry the mirror resolve URL, not its possibly expired signed CDN URL.
            parts = urllib.parse.urlsplit(url)
            query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
            query.append(("_logkv_retry", str(time.time_ns())))
            address = urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))
        wait = min(2 ** attempt, 60)
        try:
            with session.get(address, stream=target is not None, timeout=(60, 300), verify=verify_ssl) as response:
                status = response.status_code
                host = urllib.parse.urlsplit(response.url).hostname
                if status >= 400:
                    failure = f"HTTP {status} from {host}"
                    if status not in {403, 408, 429, 500, 502, 503, 504}:
                        raise RuntimeError(f"Cannot download {label}: {failure}")
                    if status == 429:
                        after = response.headers.get("Retry-After", "")
                        wait = min(int(after), 300) if after.isdigit() else 60
                elif target is None:
                    return response.json()
                else:
                    received, started = 0, time.monotonic()
                    last_report = started
                    # A failed stream restarts its .part file; never append error-page bytes.
                    with target.open("wb") as stream:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                stream.write(chunk)
                                received += len(chunk)
                                now = time.monotonic()
                                if now - last_report >= 5:
                                    print(f"{label}: {received / 1e6:.1f} MB received", flush=True)
                                    last_report = now
                    if expected_size is not None and received != expected_size:
                        raise ValueError(f"received {received} bytes, expected {expected_size}")
                    if expected_hash and sha256(target) != expected_hash:
                        raise ValueError("SHA256 mismatch")
                    return target
        except requests.RequestException as error:
            # Exception URLs can contain signed credentials; log the class, not the URL.
            failure = type(error).__name__
        except ValueError as error:
            failure = str(error)
        if attempt == retries:
            break
        print(f"{label}: {failure}; retry {attempt + 1}/{retries} via mirror in {wait}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"Cannot download {label} after {retries + 1} attempts: {failure}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--verify-ssl", action="store_true", help="enable HTTPS certificate verification")
    parser.add_argument("--retries", type=int, default=5, help="retries for 403/429, transient HTTP/network and integrity errors")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = args.out.expanduser().absolute()
    if args.verify:
        verify(root)
        return
    if args.retries < 0:
        parser.error("--retries must be >= 0")
    # Same existing dependency and streaming transport as unused/long_eval_download.py.
    # No Hugging Face SDK, hf_transfer, hf_xet or package installation is involved.
    import requests
    import urllib3

    if not args.verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    root.mkdir(parents=True, exist_ok=True)
    regular_tree(root)
    endpoint = args.endpoint.rstrip("/")
    with requests.Session() as session:
        session.headers.update({"Cache-Control": "no-cache", "Pragma": "no-cache", "Accept-Encoding": "identity"})
        options = dict(verify_ssl=args.verify_ssl, retries=args.retries)
        info = fetch(session, f"{endpoint}/api/models/{MODEL_ID}/revision/"
                     f"{urllib.parse.quote(args.revision, safe='')}?blobs=true", **options)
        download_files(session, info, root, endpoint, options)


def download_files(session, info, root, endpoint, options):
    revision = info["sha"]
    files = {entry["rfilename"]: entry for entry in info["siblings"] if entry["rfilename"] in REQUIRED_FILES}
    missing = REQUIRED_FILES - files.keys()
    if missing:
        raise ValueError(f"model repository missing required artifacts: {sorted(missing)}")
    for name, entry in files.items():
        target = root / name
        lfs = entry.get("lfs") or {}
        expected = lfs.get("sha256")
        size = entry.get("size") or lfs.get("size")
        if (expected and target.is_file() and (size is None or target.stat().st_size == size)
                and sha256(target) == expected):
            print(f"Verified existing {name}", flush=True)
            continue
        print(f"Downloading {name}", flush=True)
        url = f"{endpoint}/{MODEL_ID}/resolve/{revision}/{name}"
        temp = target.with_suffix(target.suffix + ".part")
        fetch(session, url, target=temp, expected_hash=expected, expected_size=size, **options)
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
