import pandas as pd

from downloader.odata import Odata

import sys

class Sen2(Odata):

    def __init__(self, user=None, password=None, meta_cfg=None) -> None:
        self.token, self.refresh_token = super().__init__(user, password, meta_cfg)
        self.constellation = "SENTINEL-2"

    def get_products(self, footprint, begin, end):
        """
        footpring is a geojson as created here https://geojson.io/#map=2/0/20 with reference system WGS84
            for now only supports geojson files with exactly one polygon!
            Path to geojson file!
        begin and end need to be formatted as "%Y-%m-%d" i.e. by calling a date's '.strftime("%Y-%m-%d")' method
        """
        products = super().get_products(self.constellation, footprint, begin, end)
        return products
    
    def refresh(self) -> None:
        self.token, self.refresh_token = super().refresh(self.refresh_token)
