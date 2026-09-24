python - <<'PY'
import os

# 必须放在导入 datasets / huggingface_hub 之前
root = "/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache"
os.environ["HF_HOME"] = root
os.environ["HF_DATASETS_CACHE"] = root + "/hf_cache"
os.environ["HF_HUB_CACHE"] = root + "/hub"
os.environ["HUGGINGFACE_HUB_CACHE"] = root + "/hub"
os.environ["HF_MODULES_CACHE"] = root + "/modules"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import huggingface_hub.constants as hub
import datasets.config as config
from datasets import load_dataset
from datasets.utils.logging import set_verbosity_debug

print("Hub offline:", hub.HF_HUB_OFFLINE, flush=True)
print("Datasets offline:", config.HF_DATASETS_OFFLINE, flush=True)
print("Hub cache:", hub.HF_HUB_CACHE, flush=True)
print("Datasets cache:", config.HF_DATASETS_CACHE, flush=True)

assert hub.HF_HUB_OFFLINE, "Hub 离线设置未生效"
assert config.HF_DATASETS_OFFLINE, "Datasets 离线设置未生效"

set_verbosity_debug()
print(load_dataset("allenai/social_i_qa", name="default"))
PY
