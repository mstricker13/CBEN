"""
redownload Sentinel2 BigEarthNet data at similar timesteps with cloud cover to create a cloud challenge 
dataset
"""

import argparse
import csv
import datetime
import os
from pathlib import Path
import pandas as pd
import rasterio
import sys
import shapely
import shutil
import re
import matplotlib.pyplot as plt
import numpy as np
# https://senbox.atlassian.net/wiki/spaces/SNAP/pages/2499051521/Configure+Python+to+use+the+new+SNAP-Python+esa_snappy+interface+SNAP+version+10
# https://forum.step.esa.int/t/modulenotfounderror-no-module-named-esa-snappy/42747
# https://senbox.atlassian.net/wiki/spaces/SNAP/pages/19300362/How+to+use+the+SNAP+API+from+Python
sys.path.append('/root/.snap/snap-python/')
#sys.path.append('C:/Users/m_str/.snap/snap-python/')
sys.path.append("/work/Projects/ssl/src")
#sys.path.append("C:/Users/m_str/Documents/PhD_omu_laptop/Projects/ssl/src")
import esa_snappy
from esa_snappy import ProductIO, WKTReader


from util import s2_util
from util import s1_util
from util import sentinel_util
from util import geo_utils
from util import snappy_util
from downloader.sen2 import Sen2
from downloader.sen1 import Sen1
from data_processor.earth_env_cloud_cover import EarthEnvCloud
from processor import sar_processor
from esa_snappy import HashMap
from esa_snappy import GPF
from esa_snappy import jpy

DAYS = 30
PRODUCT_TYPE = "S2MSI2A"
EARHTENVCLOUDCOVER_PATH = Path("/work/Projects/ssl/data/EarthEnv_CloudCover")
#EARHTENVCLOUDCOVER_PATH = Path("C:/Users/m_str/Documents/PhD_omu_laptop/Projects/ssl/data/EarthEnv_CloudCover")
CLOUD_COVER = 0.7

def main(start, end):
    meta_path = Path("/work/Phd_data/Projects/SSL/data/BigEarthNet/metadata.parquet")
    s2_path = Path("/work/Phd_data/Projects/SSL/data/BigEarthNet/BigEarthNet-S2")
    s1_path = Path("/work/Phd_data/Projects/SSL/data/BigEarthNet/BigEarthNet-S1")
    s2c_path = Path("/work/Phd_data/Projects/SSL/data/BigEarthNet_CloudsR/BigEarthNet-S2")
    s1c_path = Path("/work/Phd_data/Projects/SSL/data/BigEarthNet_CloudsR/BigEarthNet-S1")

    #meta_path = Path("C:/Users/m_str/Documents/Phd_data/Projects/SSL/data/BigEarthNet/metadata.parquet")
    #s2_path = Path("C:/Users/m_str/Documents/Phd_data/Projects/SSL/data/BigEarthNet/BigEarthNet-S2")
    #s1_path = Path("C:/Users/m_str/Documents/Phd_data/Projects/SSL/data/BigEarthNet/BigEarthNet-S1")
    #s2c_path = Path("C:/Users/m_str/Documents/Phd_data/Projects/SSL/data/BigEarthNet_CloudsR/BigEarthNet-S2")
    #s1c_path = Path("C:/Users/m_str/Documents/Phd_data/Projects/SSL/data/BigEarthNet_CloudsR/BigEarthNet-S1")

    meta = pd.read_parquet(meta_path)
    meta.sort_values(by=['patch_id'])
    earthcloudCover =  EarthEnvCloud(EARHTENVCLOUDCOVER_PATH)
    sen2 = Sen2()

    patch_list = ['S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP', 'S2A_MSIL2A_20170613T101031_N9999_R022_T34VER', 'S2A_MSIL2A_20170617T113321_N9999_R080_T29UPU', 'S2A_MSIL2A_20170701T093031_N9999_R136_T35VPK', 'S2A_MSIL2A_20170704T112111_N9999_R037_T29SND', 'S2A_MSIL2A_20170717T113321_N9999_R080_T29UPA', 'S2A_MSIL2A_20170717T113321_N9999_R080_T29UPV', 'S2A_MSIL2A_20170720T100031_N9999_R122_T34UDG', 'S2A_MSIL2A_20170803T094031_N9999_R036_T34TCR', 'S2A_MSIL2A_20170813T112121_N9999_R037_T29SNC', 'S2A_MSIL2A_20170818T103021_N9999_R108_T32TMT', 'S2A_MSIL2A_20170905T095031_N9999_R079_T35VNL', 'S2A_MSIL2A_20170905T095031_N9999_R079_T35WPN', 'S2A_MSIL2A_20171002T094031_N9999_R036_T34TCR', 'S2A_MSIL2A_20171002T094031_N9999_R036_T34TCS', 'S2A_MSIL2A_20171002T112111_N9999_R037_T29SNB', 'S2A_MSIL2A_20171002T112111_N9999_R037_T29SNC', 'S2A_MSIL2A_20171015T095031_N9999_R079_T33UXP', 'S2A_MSIL2A_20171101T094131_N9999_R036_T35VNJ', 'S2A_MSIL2A_20171101T094131_N9999_R036_T35VNK', 'S2A_MSIL2A_20171104T095201_N9999_R079_T33TXN', 'S2A_MSIL2A_20171121T112351_N9999_R037_T29SND', 'S2A_MSIL2A_20171201T112431_N9999_R037_T29SNB', 'S2A_MSIL2A_20171208T093351_N9999_R136_T34TEN', 'S2A_MSIL2A_20171210T101411_N9999_R022_T33UWP', 'S2A_MSIL2A_20171221T112501_N9999_R037_T29SND', 'S2A_MSIL2A_20180205T100211_N9999_R122_T35VLJ', 'S2A_MSIL2A_20180219T094031_N9999_R036_T35WPP', 'S2A_MSIL2A_20180225T114351_N9999_R123_T29UPU', 'S2A_MSIL2A_20180228T101021_N9999_R022_T34WFS', 'S2A_MSIL2A_20180318T093031_N9999_R136_T35UMB', 'S2A_MSIL2A_20180413T095031_N9999_R079_T34UEG', 'S2A_MSIL2A_20180413T095031_N9999_R079_T35VLG', 'S2A_MSIL2A_20180419T101031_N9999_R022_T34VDN', 'S2A_MSIL2A_20180430T094031_N9999_R036_T34TCR', 'S2A_MSIL2A_20180506T100031_N9999_R122_T33UWP', 'S2A_MSIL2A_20180507T093041_N9999_R136_T35UMA', 'S2A_MSIL2A_20180508T104031_N9999_R008_T31UGR', 'S2A_MSIL2A_20180508T104031_N9999_R008_T31UGS', 'S2A_MSIL2A_20180509T101031_N9999_R022_T34VDR', 'S2A_MSIL2A_20180510T094031_N9999_R036_T35VLC', 'S2A_MSIL2A_20180510T094031_N9999_R036_T35WPN', 'S2A_MSIL2A_20180510T094031_N9999_R036_T35WPP', 'S2A_MSIL2A_20180526T100031_N9999_R122_T34WFU', 'S2A_MSIL2A_20180529T115401_N9999_R023_T29UNB', 'S2B_MSIL2A_20170709T094029_N9999_R036_T35VNL', 'S2B_MSIL2A_20170716T093039_N9999_R136_T36VVR', 'S2B_MSIL2A_20170718T115359_N9999_R023_T29UPB', 'S2B_MSIL2A_20170719T094029_N9999_R036_T34TCS', 'S2B_MSIL2A_20170725T100029_N9999_R122_T35WPR', 'S2B_MSIL2A_20170801T095029_N9999_R079_T33TXN', 'S2B_MSIL2A_20170802T092029_N9999_R093_T34TFN', 'S2B_MSIL2A_20170808T094029_N9999_R036_T35ULA', 'S2B_MSIL2A_20170817T101019_N9999_R022_T34WFS', 'S2B_MSIL2A_20170824T100019_N9999_R122_T33TWM', 'S2B_MSIL2A_20170825T093029_N9999_R136_T34TEQ', 'S2B_MSIL2A_20170829T105019_N9999_R051_T31UER', 'S2B_MSIL2A_20170830T102019_N9999_R065_T34VDR', 'S2B_MSIL2A_20170831T095029_N9999_R079_T33UXP', 'S2B_MSIL2A_20170831T095029_N9999_R079_T33UXQ', 'S2B_MSIL2A_20170906T101019_N9999_R022_T34VDN', 'S2B_MSIL2A_20170906T101019_N9999_R022_T34WFS', 'S2B_MSIL2A_20170911T092019_N9999_R093_T34TFN', 'S2B_MSIL2A_20170914T093029_N9999_R136_T34TEP', 'S2B_MSIL2A_20170914T093029_N9999_R136_T34TEQ', 'S2B_MSIL2A_20170923T100019_N9999_R122_T33TWM', 'S2B_MSIL2A_20170924T093019_N9999_R136_T35UMA', 'S2B_MSIL2A_20170924T093019_N9999_R136_T35UMB', 'S2B_MSIL2A_20170924T093019_N9999_R136_T35VNH', 'S2B_MSIL2A_20170924T093019_N9999_R136_T35VPK', 'S2B_MSIL2A_20170924T093019_N9999_R136_T36VVQ', 'S2B_MSIL2A_20170927T094019_N9999_R036_T35ULB', 'S2B_MSIL2A_20170927T094019_N9999_R036_T35VLC', 'S2B_MSIL2A_20170930T095019_N9999_R079_T34UEG', 'S2B_MSIL2A_20171015T104009_N9999_R008_T31UGR', 'S2B_MSIL2A_20171015T104009_N9999_R008_T31UGS', 'S2B_MSIL2A_20171016T101009_N9999_R022_T34VDM', 'S2B_MSIL2A_20171019T102019_N9999_R065_T34VDR', 'S2B_MSIL2A_20171107T105229_N9999_R051_T31UER', 'S2B_MSIL2A_20171112T114339_N9999_R123_T29UPU', 'S2B_MSIL2A_20171206T094349_N9999_R036_T34TCR', 'S2B_MSIL2A_20171219T095409_N9999_R079_T33TWN', 'S2B_MSIL2A_20171219T095409_N9999_R079_T33TXN', 'S2B_MSIL2A_20171226T094359_N9999_R036_T34TCS', 'S2B_MSIL2A_20180127T102259_N9999_R065_T34VDN', 'S2B_MSIL2A_20180201T093219_N9999_R136_T34TEP', 'S2B_MSIL2A_20180204T094159_N9999_R036_T35VNH', 'S2B_MSIL2A_20180204T094159_N9999_R036_T35VNJ', 'S2B_MSIL2A_20180204T094159_N9999_R036_T35VPK', 'S2B_MSIL2A_20180220T114339_N9999_R123_T29UPV', 'S2B_MSIL2A_20180223T101019_N9999_R022_T34WFT', 'S2B_MSIL2A_20180224T112109_N9999_R037_T29SNC', 'S2B_MSIL2A_20180225T105019_N9999_R051_T31UER', 'S2B_MSIL2A_20180326T112109_N9999_R037_T29SNB', 'S2B_MSIL2A_20180417T102019_N9999_R065_T34WFV', 'S2B_MSIL2A_20180421T100029_N9999_R122_T33TWM', 'S2B_MSIL2A_20180421T100029_N9999_R122_T33TWN', 'S2B_MSIL2A_20180421T100029_N9999_R122_T33UWQ', 'S2B_MSIL2A_20180421T100029_N9999_R122_T33UXQ', 'S2B_MSIL2A_20180421T114349_N9999_R123_T29UPU', 'S2B_MSIL2A_20180422T093029_N9999_R136_T34TEQ', 'S2B_MSIL2A_20180428T095029_N9999_R079_T33TXN', 'S2B_MSIL2A_20180502T093039_N9999_R136_T34TEP', 'S2B_MSIL2A_20180506T105029_N9999_R051_T31UER', 'S2B_MSIL2A_20180509T092029_N9999_R093_T34TFN', 'S2B_MSIL2A_20180511T100029_N9999_R122_T34VDM', 'S2B_MSIL2A_20180511T100029_N9999_R122_T35WPR', 'S2B_MSIL2A_20180515T094029_N9999_R036_T35VNJ', 'S2B_MSIL2A_20180515T112109_N9999_R037_T29SNC', 'S2B_MSIL2A_20180515T112109_N9999_R037_T29SND', 'S2B_MSIL2A_20180521T100029_N9999_R122_T34WFS', 'S2B_MSIL2A_20180522T093029_N9999_R136_T35VPJ', 'S2B_MSIL2A_20180525T094029_N9999_R036_T35VNH', 'S2B_MSIL2A_20180525T094029_N9999_R036_T35VNK', 'S2B_MSIL2A_20180525T094029_N9999_R036_T35VNL']
    #print(len(patch_list)) #115
    i = 0
    for elem in patch_list[start:end]:
        print(f"current index {i}")
        i = i+1
        parallel_patch_id = elem
        s2_downloaded = False
        for index, row in meta.iterrows():
            s2_id = row["patch_id"]  
        
            if parallel_patch_id == "_".join(s2_id.split("_")[:-2]):
                if not s2_downloaded:
                    print(f"Downloading cloudy version of {parallel_patch_id}")
                    s1_id = row["s1_name"]
                    s2_elem  = s2_path / "_".join(s2_id.split("_")[:-2]) / s2_id
                    s1_elem  = s1_path / "_".join(s1_id.split("_")[:-3]) / s1_id
                    s2_date = s2_util.get_date_from_name(s2_id)
                    tile_id = s2_id.split("_")[-3][1:] #last slice removes first T

                    footprint, src_crs = geo_utils.extract_bounds_from_tif(s2_elem / f"{s2_id}_B02.tif")
                    # download call needs to be in WGS84
                    footprint = geo_utils.transform_bounds(footprint, src_crs, 'EPSG:4326')
                    center = shapely.centroid(shapely.Polygon([(footprint[0][0], footprint[1][0]), (footprint[0][1], footprint[1][1]),
                             (footprint[0][2], footprint[1][2]), (footprint[0][3], footprint[1][3]),
                             (footprint[0][4], footprint[1][4])]))
            
                    #aquire cloud information
                    mean, sd = earthcloudCover.get_cloud_cover(center.y, center.x, str(s2_date.month))
                    min_cloud_cover = mean - sd
                    max_cloud_cover = mean + sd
                    #get sentinel2 tile which satisfies the cloud conditions and is the temporal closest to the original
                    tar_dir, extracted_dir, tar_product_meta, s2_downloaded_id = get_closest_tile(s2_date, footprint, tile_id, min_cloud_cover, max_cloud_cover, s2_id, s2c_path, sen2)            
                    s2_downloaded = True

def make_tar_dir(tar_product_meta, s2_id, s2c_path, tar_dir=None):
    if tar_product_meta is not None:
        out = create_out_dir(tar_product_meta, s2_id, s2c_path)
    else:
        out = create_out_dir(None, s2_id, s2c_path, tar_dir=tar_dir)
    out.mkdir(parents=False, exist_ok=True)
    return out

def get_closest_tile(date, footprint, tile_id, min_cloud_cover, max_cloud_cover, s2_id, s2c_path, s2token, internet_id=None, tar_dir=None):
    s2token.refresh()
    found = False
    default_days = DAYS
    while not found:
        start_date = date - datetime.timedelta(days=default_days)
        end_date = date + datetime.timedelta(days=default_days)
        ids = s2token.get_products(footprint, start_date, end_date)
        ids = s2_util.filter_products(ids, tileId=tile_id, productType=PRODUCT_TYPE, 
                                  cloudCover_min=min_cloud_cover, cloudCover_max=max_cloud_cover)
        ids = s2_util.keep_highest_N(ids)
        if len(ids) > 0:
            found = True
        else:
            default_days += 30
    tar_product = s2_util.get_temporal_closest(date, ids)
    if internet_id is None:
        internet_id = tar_product['Id']
    else:
        tar_product = None
    out = create_out_dir(tar_product, s2_id, s2c_path, tar_dir)
    product_path = out.parent / "product.zip"
    product_path_ex = out.parent / "product"
    #check if folder exists
    if not product_path.exists():
        #This cloud cover is for one whole tile and not a specific subsample of a tile!
        s2token.download(internet_id, s2token.token, product_path)  # will skip download if it already exists for a tile!
    #out.mkdir(parents=False, exist_ok=True)
    return out, product_path_ex, tar_product, internet_id
    
def get_file_paths(extracted_dir):
    #TODO adapt these calls to pure pythonlib...

    # 10m resolution: 2 3 4 8
    # 20m resolution: 5 6 7 8a 11 12
    # 60m resolution: 1 9
    subfolder = [ Path(f.path) for f in os.scandir(extracted_dir) if f.is_dir() ][0] # there should only be one folder
    subfolder = subfolder / "GRANULE" 
    subfolder = [ Path(f.path) for f in os.scandir(subfolder) if f.is_dir() ][0] # there should only be one folder
    subfolder = subfolder / "IMG_DATA" 

    ten_meters = [ f.path for f in os.scandir((subfolder / "R10m" )) if f.is_file() ]
    ten_meter_regular_name = re.compile(r'_B0[2348]_')
    ten_meters = [ Path(file_path) for file_path in ten_meters if ten_meter_regular_name.search(file_path) ]
    
    twenty_meters = [ f.path for f in os.scandir((subfolder / "R20m" )) if f.is_file() ]
    twenty_meter_regular_name = re.compile(r'_B[018][56712A]_')
    twenty_meters = [ file_path for file_path in twenty_meters if twenty_meter_regular_name.search(file_path) ]
    twenty_meter_regular_name = re.compile(r'_B[0][2348]_')
    twenty_meters = [ Path(file_path) for file_path in twenty_meters if not twenty_meter_regular_name.search(file_path) ]

    sixty_meters = [ f.path for f in os.scandir((subfolder / "R60m" )) if f.is_file() ]
    sixty_meter_regular_name = re.compile(r'_B0[19]_')
    sixty_meters = [ Path(file_path) for file_path in sixty_meters if sixty_meter_regular_name.search(file_path) ]

    return (ten_meters + twenty_meters + sixty_meters)

def get_file_paths_s1(extracted_dir):

    #TODO adapt these calls to pure pythonlib...

    #subfolder = [ Path(f.path) for f in os.scandir(extracted_dir) if f.is_dir() ][0] # there should only be one folder
    #subfolder = subfolder / "measurement" 

    subfolder = [ Path(f.path) for f in os.scandir(extracted_dir) if f.is_dir() ][0] # there should only be one folder
    dim_file =  [ Path(f.path) for f in os.scandir(subfolder) if f.is_file() ][0] # there should only be one dim file
    
    #vvvhv= [ f.path for f in os.scandir(subfolder) if f.is_file() ]

    return dim_file

def get_safe_file_path_s1(extracted_dir):

    #TODO adapt these calls to pure pythonlib...

    subfolder = [ Path(f.path) for f in os.scandir(extracted_dir) if f.is_dir() ][0] # there should only be one folder
    subfolder = subfolder / "manifest.safe" 

    return subfolder


def create_out_dir(tar_product, s2_id, s2c_path, tar_dir=None):
    if tar_product is not None:
        tile_dir = "_".join(tar_product["Name"].split("_")[:-1])
        sub_tile_dir = f"{tile_dir}_{s2_id.split('_')[-2]}_{s2_id.split('_')[-1]}"
        out = s2c_path / tile_dir / sub_tile_dir
    else:
        out = tar_dir
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--index_start", type=int)
    parser.add_argument("-e", "--index_end", type=int)
    args = parser.parse_args()

    start = args.index_start
    end = args.index_end
    main(start, end)
