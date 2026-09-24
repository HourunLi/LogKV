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
from lm_eval.tasks import TaskManager, get_task_dict

tasks = get_task_dict(["social_iqa"], task_manager=TaskManager())
print("lm-eval 任务加载成功:", list(tasks))

PY

Datasets offline: True
Hub cache: /home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/hub
Datasets cache: /home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/hf_cache
Using the latest cached version of the dataset since allenai/social_i_qa couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'default' at /home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/hf_cache/allenai___social_i_qa/default/0.1.0/674d85e42ac7430d3dcd4de7007feaffcb1527c535121e09bab2803fbcc925f8 (last modified on Wed Sep 23 11:08:29 2026).
Overwrite dataset info from restored data version if exists.
Loading Dataset info from /home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/hf_cache/allenai___social_i_qa/default/0.1.0/674d85e42ac7430d3dcd4de7007feaffcb1527c535121e09bab2803fbcc925f8
Constructing Dataset for split train, validation, from /home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/hf_cache/allenai___social_i_qa/default/0.1.0/674d85e42ac7430d3dcd4de7007feaffcb1527c535121e09bab2803fbcc925f8
DatasetDict({
    train: Dataset({
        features: ['context', 'question', 'answerA', 'answerB', 'answerC', 'label'],
        num_rows: 33410
    })
    validation: Dataset({
        features: ['context', 'question', 'answerA', 'answerB', 'answerC', 'label'],
        num_rows: 1954
    })
})
Using the latest cached version of the dataset since allenai/social_i_qa couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'default' at /home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/hf_cache/allenai___social_i_qa/default/0.1.0/674d85e42ac7430d3dcd4de7007feaffcb1527c535121e09bab2803fbcc925f8 (last modified on Wed Sep 23 11:08:29 2026).
Overwrite dataset info from restored data version if exists.
Loading Dataset info from /home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/hf_cache/allenai___social_i_qa/default/0.1.0/674d85e42ac7430d3dcd4de7007feaffcb1527c535121e09bab2803fbcc925f8
Constructing Dataset for split train, validation, from /home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/hf_cache/allenai___social_i_qa/default/0.1.0/674d85e42ac7430d3dcd4de7007feaffcb1527c535121e09bab2803fbcc925f8
lm-eval 任务加载成功: ['social_iqa']
