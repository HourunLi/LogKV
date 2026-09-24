[hf-cache] {"event": "runtime", "rank": 7, "host": "os-node-created-4wwvk", "pid": 2782, "python": "/home/ma-user/anaconda3/envs/torch218/bin/python", "script": "/home/ma-user/modelarts/user-job-dir/semanticLogKV/eval.py", "cwd": "/home/ma-user/modelarts/user-job-dir/semanticLogKV", "packages": {"datasets": {"version": "4.8.4", "file": "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/datasets/__init__.py"}, "huggingface_hub": {"version": "0.36.2", "file": "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/huggingface_hub/__init__.py"}, "lm_eval": {"version": "0.4.11", "file": "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/lm_eval/__init__.py"}}, "datasets_cache": "/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/hf_cache", "hub_cache": "/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/hub", "datasets_offline": true, "hub_offline": true, "environment": {"HF_HOME": "/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache", "HF_DATASETS_CACHE": "/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/hf_cache", "HF_HUB_CACHE": "/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/hub", "HF_MODULES_CACHE": "/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/modules", "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1", "HF_ENDPOINT": null, "PYTHONPATH": "/usr/local/seccomponent/lib:/home/ma-user/infer/model/1", "PYTHON_EXEC": null}, "cache_trace_supported": true}


[hf-cache] {"event": "local_cache_failure", "rank": 4, "host": "os-node-created-4wwvk", "pid": 2779, "python": "/home/ma-user/anaconda3/envs/torch218/bin/python", "dataset": "allenai/social_i_qa", "cache_dir_argument": null, "effective_cache_dir": "/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/hf_cache", "traceback": "Traceback (most recent call last):\n  File \"/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/datasets/load.py\", line 1113, in dataset_module_factory\n    _raise_if_offline_mode_is_enabled()\n  File \"/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/datasets/utils/file_utils.py\", line 288, in _raise_if_offline_mode_is_enabled\n    raise huggingface_hub.errors.OfflineModeIsEnabled(\nhuggingface_hub.errors.OfflineModeIsEnabled: Offline mode is enabled.\n\nThe above exception was the direct cause of the following exception:\n\nTraceback (most recent call last):\n  File \"/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/datasets/load.py\", line 1149, in dataset_module_factory\n    raise ConnectionError(f\"Couldn't reach '{path}' on the Hub ({e.__class__.__name__})\") from e\nConnectionError: Couldn't reach 'allenai/social_i_qa' on the Hub (OfflineModeIsEnabled)\n\nDuring handling of the above exception, another exception occurred:\n\nTraceback (most recent call last):\n  File \"/home/ma-user/modelarts/user-job-dir/semanticLogKV/eval.py\", line 358, in traced_get_module\n    return original(self, *args, **kwargs)\n  File \"/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/datasets/load.py\", line 831, in get_module\n    raise FileNotFoundError(f\"Dataset {self.name} is not cached in {self.cache_dir}\")\nFileNotFoundError: Dataset allenai/social_i_qa is not cached in None\n"}



[rank2]: During handling of the above exception, another exception occurred:

[rank2]: Traceback (most recent call last):
[rank2]:   File "/home/ma-user/modelarts/user-job-dir/semanticLogKV/eval.py", line 1659, in <module>
[rank2]:     run_cli(main)
[rank2]:   File "/home/ma-user/modelarts/user-job-dir/semanticLogKV/utils.py", line 225, in run_cli
[rank2]:     return func(**kwargs)
[rank2]:   File "/home/ma-user/modelarts/user-job-dir/semanticLogKV/utils.py", line 147, in wrapper
[rank2]:     return func(*expanded_args, **expanded_kwargs)
[rank2]:   File "/home/ma-user/modelarts/user-job-dir/semanticLogKV/eval.py", line 1412, in main
[rank2]:     results = evaluator.simple_evaluate(
[rank2]:   File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/lm_eval/utils.py", line 498, in _wrapper
[rank2]:     return fn(*args, **kwargs)
[rank2]:   File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/lm_eval/evaluator.py", line 292, in simple_evaluate
[rank2]:     task_dict = get_task_dict(
[rank2]:   File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/lm_eval/tasks/__init__.py", line 646, in get_task_dict
[rank2]:     task_name_from_string_dict = task_manager.load_task_or_group(
[rank2]:   File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/lm_eval/tasks/__init__.py", line 428, in load_task_or_group
[rank2]:     collections.ChainMap(
[rank2]:   File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/lm_eval/tasks/__init__.py", line 430, in <lambda>
[rank2]:     lambda task: self._load_individual_task_or_group(task),
[rank2]:   File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/lm_eval/tasks/__init__.py", line 328, in _load_individual_task_or_group
[rank2]:     return _load_task(task_config, task=name_or_config)
[rank2]:   File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/lm_eval/tasks/__init__.py", line 288, in _load_task
[rank2]:     task_object = ConfigurableTask(config=config)
[rank2]:   File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/lm_eval/api/task.py", line 748, in __init__
[rank2]:     self.download(self.config.dataset_kwargs)
[rank2]:   File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/lm_eval/api/task.py", line 864, in download
[rank2]:     self.dataset = datasets.load_dataset(
[rank2]:   File "/home/ma-user/modelarts/user-job-dir/semanticLogKV/eval.py", line 413, in load_with_local_fallback
[rank2]:     result = _read_social_iqa_cache(directory)
[rank2]:   File "/home/ma-user/modelarts/user-job-dir/semanticLogKV/eval.py", line 324, in _read_social_iqa_cache
[rank2]:     with (directory / "dataset_info.json").open(encoding="utf-8") as stream:
[rank2]:   File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/pathlib.py", line 1119, in open
[rank2]:     return self._accessor.open(self, mode, buffering, encoding, errors,
[rank2]: NotADirectoryError: [Errno 20] Not a directory: '/home/ma-user/work/bucket-pangu-green/lihourun/data/hf_cache/hf_cache/allenai___social_i_qa/default/0.1.0/674d85e42ac7430d3dcd4de7007feaffcb1527c535121e09bab2803fbcc925f8/dataset_info.json'
