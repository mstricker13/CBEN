from pathlib import Path

import globals


def get_meta_info_loc() -> Path:
    """
    Returns Path object pointing to location 'meta_info.json'
    """
    return globals.META_LOCATION
