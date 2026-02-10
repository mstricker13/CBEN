import json


def extract_odata_user(meta_cfg):
    """
    Extract odata username from given path to 'meta_info.json'
    """
    with open(meta_cfg, "r") as f:
        data = json.load(f)
        return data["odata"]["username"]


def extract_odata_password(meta_cfg):
    """
    Extract odata password from given path to 'meta_info.json'
    """
    with open(meta_cfg, "r") as f:
        data = json.load(f)
        return data["odata"]["password"]
