import util
import util.geo_utils
import sys

class EarthEnvCloud():
    def __init__(self, loc) -> None:
        self.location = loc

    def get_cloud_cover(self, lat, lon, month):
        #month is a string with 2 digits
        if len(month) == 1:
            month = "0" + month
        sd_file  = self.location / "MODCF_interannualSD.tif"
        mean_file = self.location / f"MODCF_monthlymean_{month}.tif"
        sd = util.geo_utils.get_val_at_lat_lon(sd_file, lat, lon, 1) * 0.01 #normal percentage as defined in documentation
        mean = util.geo_utils.get_val_at_lat_lon(mean_file, lat, lon, 1) * 0.01 
        return mean, sd

