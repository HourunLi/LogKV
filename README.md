[rank2]:[E919 06:52:14.737942593 ProcessGroupNCCL.cpp:1628] [PG ID 0 PG GUID 0(default_pg) Rank 2] ProcessGroupNCCL preparing to dump debug info. Include stack trace: 1, only active collectives: 0
[rank0]:[F919 07:00:13.418192812 ProcessGroupNCCL.cpp:1653] [PG ID 0 PG GUID 0(default_pg) Rank 0] [PG ID 0 PG GUID 0(default_pg) Rank 0] Terminating the process after attempting to dump debug info, due to ProcessGroupNCCL watchdog hang.
[rank4]:[F919 07:00:13.482187133 ProcessGroupNCCL.cpp:1653] [PG ID 0 PG GUID 0(default_pg) Rank 4] [PG ID 0 PG GUID 0(default_pg) Rank 4] Terminating the process after attempting to dump debug info, due to ProcessGroupNCCL watchdog hang.
[rank3]:[F919 07:00:14.032642261 ProcessGroupNCCL.cpp:1653] [PG ID 0 PG GUID 0(default_pg) Rank 3] [PG ID 0 PG GUID 0(default_pg) Rank 3] Terminating the process after attempting to dump debug info, due to ProcessGroupNCCL watchdog hang.
[rank5]:[F919 07:00:14.661297504 ProcessGroupNCCL.cpp:1653] [PG ID 0 PG GUID 0(default_pg) Rank 5] [PG ID 0 PG GUID 0(default_pg) Rank 5] Terminating the process after attempting to dump debug info, due to ProcessGroupNCCL watchdog hang.
[rank6]:[F919 07:00:14.671069836 ProcessGroupNCCL.cpp:1653] [PG ID 0 PG GUID 0(default_pg) Rank 6] [PG ID 0 PG GUID 0(default_pg) Rank 6] Terminating the process after attempting to dump debug info, due to ProcessGroupNCCL watchdog hang.
[rank1]:[F919 07:00:14.713922286 ProcessGroupNCCL.cpp:1653] [PG ID 0 PG GUID 0(default_pg) Rank 1] [PG ID 0 PG GUID 0(default_pg) Rank 1] Terminating the process after attempting to dump debug info, due to ProcessGroupNCCL watchdog hang.
[rank7]:[F919 07:00:14.721279999 ProcessGroupNCCL.cpp:1653] [PG ID 0 PG GUID 0(default_pg) Rank 7] [PG ID 0 PG GUID 0(default_pg) Rank 7] Terminating the process after attempting to dump debug info, due to ProcessGroupNCCL watchdog hang.
[rank2]:[F919 07:00:14.739308528 ProcessGroupNCCL.cpp:1653] [PG ID 0 PG GUID 0(default_pg) Rank 2] [PG ID 0 PG GUID 0(default_pg) Rank 2] Terminating the process after attempting to dump debug info, due to ProcessGroupNCCL watchdog hang.
W0919 07:00:18.439000 2643 site-packages/torch/distributed/elastic/multiprocessing/api.py:1012] Sending process 2765 closing signal SIGTERM
W0919 07:00:18.441000 2643 site-packages/torch/distributed/elastic/multiprocessing/api.py:1012] Sending process 2766 closing signal SIGTERM
W0919 07:00:18.442000 2643 site-packages/torch/distributed/elastic/multiprocessing/api.py:1012] Sending process 2767 closing signal SIGTERM
W0919 07:00:18.443000 2643 site-packages/torch/distributed/elastic/multiprocessing/api.py:1012] Sending process 2768 closing signal SIGTERM
W0919 07:00:18.443000 2643 site-packages/torch/distributed/elastic/multiprocessing/api.py:1012] Sending process 2769 closing signal SIGTERM
W0919 07:00:18.444000 2643 site-packages/torch/distributed/elastic/multiprocessing/api.py:1012] Sending process 2770 closing signal SIGTERM
W0919 07:00:18.445000 2643 site-packages/torch/distributed/elastic/multiprocessing/api.py:1012] Sending process 2771 closing signal SIGTERM
E0919 07:00:20.968000 2643 site-packages/torch/distributed/elastic/multiprocessing/api.py:986] failed (exitcode: -6) local_rank: 0 (pid: 2764) of binary: /home/ma-user/anaconda3/envs/torch218/bin/python
Traceback (most recent call last):
  File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/torch/distributed/run.py", line 994, in <module>
    main()
  File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 362, in wrapper
    return f(*args, **kwargs)
  File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/torch/distributed/run.py", line 990, in main
    run(args)
  File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/torch/distributed/run.py", line 981, in run
    elastic_launch(
  File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 170, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/ma-user/anaconda3/envs/torch218/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 317, in launch_agent
















  [rank7]: Traceback (most recent call last):
[rank7]:   File "/home/ma-user/modelarts/user-job-dir/semanticLogKV/demo.py", line 1210, in <module>
[rank7]:     run_cli(main)
[rank7]:   File "/home/ma-user/modelarts/user-job-dir/semanticLogKV/utils.py", line 225, in run_cli
[rank7]:     return func(**kwargs)
[rank7]:   File "/home/ma-user/modelarts/user-job-dir/semanticLogKV/utils.py", line 147, in wrapper
[rank7]:     return func(*expanded_args, **expanded_kwargs)
[rank7]:   File "/home/ma-user/modelarts/user-job-dir/semanticLogKV/demo.py", line 759, in main
[rank7]:     raise FileNotFoundError(
[rank7]: FileNotFoundError: Training input checkpoint (ckpt_dir/base) does not exist or is empty: /home/ma-user/work/bucket-wulan-green/zhaoyusheng/checkpoints/Qwen/Qwen3-1.7B-Base/lit_model.pth. Check ckpt_dir/resume_dir. save_path is the output directory; it is only selected as an input when explicitly requested by resume_dir or when auto_resume finds a checkpoint.







    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
