python - <<'PY'
from pathlib import Path
from importlib.metadata import version

for package in ("datasets", "huggingface-hub", "lm_eval"):
    print(package, version(package))

root = Path("/workspace/data/hf_cache")
for subdir in ("hf_cache", "hub", "modules"):
    folder = root / subdir
    print(f"\n[{folder}] exists={folder.is_dir()}")
    for path in folder.rglob("*social*"):
        print(path, "exists=", path.exists(), "symlink=", path.is_symlink())
PY


datasets 4.8.4
huggingface-hub 0.36.2
lm_eval 0.4.11

[/workspace/data/hf_cache/hf_cache] exists=True

[/workspace/data/hf_cache/hub] exists=True

[/workspace/data/hf_cache/modules] exists=True


HF_HOME=/workspace/data/hf_cache \
HF_DATASETS_CACHE=/workspace/data/hf_cache/hf_cache \
HF_HUB_CACHE=/workspace/data/hf_cache/hub \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python - <<'PY'
from datasets import load_dataset
from datasets.utils.logging import set_verbosity_debug

set_verbosity_debug()
ds = load_dataset("allenai/social_i_qa", name="default")
print(ds)
PY
