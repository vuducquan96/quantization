import json
import toml
import yaml
from easydict import EasyDict
from pdb import set_trace as bp
from core.torchie.utils import Config


def config_loader(file_path):

    if ".json" in file_path.lower():
        with open(file_path) as json_file:
            data = json.load(json_file)
        data = EasyDict(data)
        return data

    if ".toml" in file_path.lower():
        data = toml.load(file_path)
        data = EasyDict(data)
        return data

    if ".yaml" in file_path.lower():
        with open(file_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        data = EasyDict(data)
        return data

    if ".py" in file_path.lower():
        data = Config.fromfile(file_path)
        return data
