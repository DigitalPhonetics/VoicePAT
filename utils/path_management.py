from pathlib import Path
import shutil
import os
import glob


def create_clean_dir(dir_name:Path):
    if dir_name.exists():
        remove_contents_in_dir(dir_name)
    else:
        dir_name.mkdir(exist_ok=True, parents=True)


def remove_contents_in_dir(dir_name:Path):
    # solution from https://stackoverflow.com/a/56151260
    for path in dir_name.glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def transform_path(file_path, parent_dir=None):
    if not file_path:
        return None
    file_path = Path(file_path)
    if parent_dir and not file_path.is_absolute():
        file_path = parent_dir / file_path
    return file_path


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*****')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return Path(sorted(cp_list)[-1])


def get_datasets(config):
    datasets = {}
    data_dir = config.get('data_dir', None).expanduser() # if '~' is given in path then manually expand
    for dataset in config['datasets']:
        if data_dir:
            for subset in dataset['enrolls'] + dataset['trials']:
                dataset_name = f'{dataset["data"]}_{dataset["set"]}_{subset}'
                datasets[dataset_name] = Path(data_dir, dataset_name)
        else:
            dataset_path = Path(dataset['name'])
            for subset in dataset['enrolls'] + dataset['trials']:
                dataset_name = f'{dataset_path.name}_{subset}'
                datasets[dataset_name] = dataset_path
    return datasets

