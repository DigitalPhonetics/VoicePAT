# For more information, see parse_arguments() in
# https://github.com/speechbrain/blob/develop/ speechbrain/speechbrain/core.py
run_opts = {
    'debug': False,
    'debug_batches': 2,
    'debug_epochs': 2,
    'debug_persistently': False,
    'device': 'cuda:0',
    'data_parallel_backend': False,
    'distributed_launch': False,
    'distributed_backend': 'nccl',
    'find_unused_parameters': False,
    'tqdm_colored_bar': False
}