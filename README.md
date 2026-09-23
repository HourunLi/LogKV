python - <<'PY'
from pathlib import Path
from importlib.metadata import version

for package in ("datasets", "huggingface-hub", "lm_eval"):
    print(package, version(package))

root = Path("/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache")
for subdir in ("hf_cache", "hub", "modules"):
    folder = root / subdir
    print(f"\n[{folder}] exists={folder.is_dir()}")
    for path in folder.rglob("*social*"):
        print(path, "exists=", path.exists(), "symlink=", path.is_symlink())
PY
