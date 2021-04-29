from pathlib import Path
from pprint import pprint
from typing import Union, Any, Dict

import yaml

from ..types import typePath


def path2Path(path: typePath) -> Path:
    return path if isinstance(path, Path) else Path(path)


def yaml_load(yaml_path: Union[Path, str], verbose=False) -> Dict[str, Any]:
    with open(str(yaml_path), "r") as stream:
        data_loaded: dict = yaml.safe_load(stream)
    if verbose:
        print(f"Loaded yaml path:{str(yaml_path)}")
        pprint(data_loaded)
    return data_loaded


def yaml_write(
    dictionary: Dict, save_dir: typePath, save_name: str, force_overwrite=True
) -> None:
    save_path = path2Path(save_dir) / save_name
    if save_path.exists() and not force_overwrite:
        save_name = (save_name.split(".")[0] + "_copy" + "." + save_name.split(".")[1])
        save_path = path2Path(save_dir) / save_name
    with open(str(save_path), "w") as outfile:  # type: ignore
        yaml.dump(dictionary, outfile, default_flow_style=False)
