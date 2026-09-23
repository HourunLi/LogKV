[rank7]: Traceback (most recent call last):
[rank7]:   File "/home/ma-user/modelarts/user-job-dir/semanticLogKV/eval.py", line 1549, in <module>
[rank7]:     run_cli(main)
[rank7]:   File "/home/ma-user/modelarts/user-job-dir/semanticLogKV/utils.py", line 225, in run_cli
[rank7]:     return func(**kwargs)
[rank7]:   File "/home/ma-user/modelarts/user-job-dir/semanticLogKV/utils.py", line 147, in wrapper
[rank7]:     return func(*expanded_args, **expanded_kwargs)
[rank7]:   File "/home/ma-user/modelarts/user-job-dir/semanticLogKV/eval.py", line 1302, in main
[rank7]:     results = evaluator.simple_evaluate(
[rank7]:   File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/lm_eval/utils.py", line 498, in _wrapper
[rank7]:     return fn(*args, **kwargs)
[rank7]:   File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/lm_eval/evaluator.py", line 292, in simple_evaluate
[rank7]:     task_dict = get_task_dict(
[rank7]:   File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/lm_eval/tasks/__init__.py", line 646, in get_task_dict
[rank7]:     task_name_from_string_dict = task_manager.load_task_or_group(
[rank7]:   File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/lm_eval/tasks/__init__.py", line 428, in load_task_or_group
[rank7]:     collections.ChainMap(
[rank7]:   File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/lm_eval/tasks/__init__.py", line 430, in <lambda>
[rank7]:     lambda task: self._load_individual_task_or_group(task),
[rank7]:   File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/lm_eval/tasks/__init__.py", line 328, in _load_individual_task_or_group
[rank7]:     return _load_task(task_config, task=name_or_config)
[rank7]:   File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/lm_eval/tasks/__init__.py", line 288, in _load_task
[rank7]:     task_object = ConfigurableTask(config=config)
[rank7]:   File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/lm_eval/api/task.py", line 748, in __init__
[rank7]:     self.download(self.config.dataset_kwargs)
[rank7]:   File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/lm_eval/api/task.py", line 864, in download
[rank7]:     self.dataset = datasets.load_dataset(
[rank7]:   File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/datasets/load.py", line 1688, in load_dataset
[rank7]:     builder_instance = load_dataset_builder(
[rank7]:   File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/datasets/load.py", line 1315, in load_dataset_builder
[rank7]:     dataset_module = dataset_module_factory(
[rank7]:   File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/datasets/load.py", line 1207, in dataset_module_factory
[rank7]:     raise e1 from None
[rank7]:   File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/datasets/load.py", line 1149, in dataset_module_factory
[rank7]:     raise ConnectionError(f"Couldn't reach '{path}' on the Hub ({e.__class__.__name__})") from e
[rank7]: ConnectionError: Couldn't reach 'allenai/social_i_qa' on the Hub (OfflineModeIsEnabled)
os-node-created-pfwt5:2777:2777 [5] NCCL INFO ENV/Plugin: Closing env plugin ncclEnvDefault
os-node-created-pfwt5:2773:2773 [1] NCCL INFO ENV/Plugin: Closing env plugin ncclEnvDefault
os-node-created-pfwt5:2774:2774 [2] NCCL INFO ENV/Plugin: Closing env plugin ncclEnvDefault
