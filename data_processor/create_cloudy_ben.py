"""
redownload Sentinel2 BigEarthNet data at similar timesteps with cloud cover to create a cloud challenge 
dataset
"""

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
sys.path.append('C:/Users/m_str/.snap/snap-python/')
import esa_snappy
from esa_snappy import ProductIO, WKTReader


sys.path.append("C:/Users/m_str/Documents/PhD_omu_laptop/Projects/ssl/src")
from util import s2_util
from util import s1_util
from util import sentinel_util
from util import geo_utils
from util import snappy_util
from downloader.sen2 import Sen2
from downloader.sen1 import Sen1
from earth_env_cloud_cover import EarthEnvCloud
from processor import sar_processor
from esa_snappy import HashMap
from esa_snappy import GPF
from esa_snappy import jpy

DAYS = 30
PRODUCT_TYPE = "S2MSI2A"
EARHTENVCLOUDCOVER_PATH = Path("C:/Users/m_str/Documents/PhD_omu_laptop/Projects/ssl/data/EarthEnv_CloudCover")

def main():
    with open(Path("C:/Users/m_str/Documents/Phd_data/Projects/SSL/data/BigEarthNet_Clouds/log.csv"), 'w', newline='') as csvfile:
        fieldnames = ["internet_id", 'tile', 'cloudover', 'Temporal Distance']
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
    meta_path = Path("C:/Users/m_str/Documents/Phd_data/Projects/SSL/data/BigEarthNet/metadata.parquet")
    s2_path = Path("C:/Users/m_str/Documents/Phd_data/Projects/SSL/data/BigEarthNet/BigEarthNet-S2")
    s1_path = Path("C:/Users/m_str/Documents/Phd_data/Projects/SSL/data/BigEarthNet/BigEarthNet-S1")
    s2c_path = Path("C:/Users/m_str/Documents/Phd_data/Projects/SSL/data/BigEarthNet_Clouds/BigEarthNet-S2")
    s1c_path = Path("C:/Users/m_str/Documents/Phd_data/Projects/SSL/data/BigEarthNet_Clouds/BigEarthNet-S1")
    meta = pd.read_parquet(meta_path)
    meta.sort_values(by=['patch_id'])
    earthcloudCover =  EarthEnvCloud(EARHTENVCLOUDCOVER_PATH)
    cur_tile = None
    sen2 = Sen2()
    sen1 = Sen1()
    for index, row in meta.iterrows():
        if index <= 1:
            #row is still a dictionary!

            #get all important meta information
            s2_id = row["patch_id"]
            s1_id = row["s1_name"]
            s2_elem  = s2_path / "_".join(s2_id.split("_")[:-2]) / s2_id
            s1_elem  = s1_path / "_".join(s1_id.split("_")[:-3]) / s1_id
            s2_date = s2_util.get_date_from_name(s2_id)
            tile_id = s2_id.split("_")[-3][1:] #last slice removes first T

            # cleanup old files as soon as new tile is being handled
            if cur_tile is None:
                cur_tile = tile_id
                print(f"Working on tile {tile_id}")
            elif cur_tile != tile_id:
                cur_tile = tile_id
                print(f"Working on tile {tile_id}")
                full_tile_path = tar_dir.parent
                if (full_tile_path / "product.zip").exists():
                    os.remove((full_tile_path / "product.zip"))
                if product_path_s1.exists():
                    os.remove(product_path_s1)
                if (full_tile_path / "product").exists():
                    shutil.rmtree((full_tile_path / "product"))
                if product_path_ex_s1.exists():
                    shutil.rmtree(product_path_ex_s1)

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

            ####################################################################################################
            # S2 HANDLING
            ####################################################################################################

            #get sentinel2 tile which satisfies the cloud conditions and is the temporal closest to the original
            tar_dir, extracted_dir = get_closest_tile(s2_date, footprint, tile_id, min_cloud_cover, max_cloud_cover, s2_id, s2c_path, sen2)
            # get the paths to the all the downloaded bands
            files = get_file_paths(extracted_dir)
            # extract the relevant roi from the downloads
            footprint10m, src_crs = geo_utils.extract_bounds_from_tif(s2_elem / f"{s2_id}_B02.tif")
            footprint20m, src_crs = geo_utils.extract_bounds_from_tif(s2_elem / f"{s2_id}_B05.tif")
            footprint60m, src_crs = geo_utils.extract_bounds_from_tif(s2_elem / f"{s2_id}_B01.tif")
            for band_file in files:
                band = str(band_file).split("_")[-2]
                out_file = tar_dir / f"{s2_id}_{band}.tif"  # not sure if this makes sense, but so a mapping back to the original BEN sample is possible...
                # so folder name is based on the new tiles properties while the actual sample is based on the original
                #maye the resolution seperation wouldn't have been necessary
                if "10m.jp2" in str(band_file):
                    roi_data, out_transform, crs = geo_utils.extract_roi(band_file, footprint10m, src_crs)
                elif "20m.jp2" in str(band_file):
                    roi_data, out_transform, crs = geo_utils.extract_roi(band_file, footprint20m, src_crs)
                elif "60m.jp2" in str(band_file):
                    roi_data, out_transform, crs = geo_utils.extract_roi(band_file, footprint60m, src_crs)
                else:
                    sys.exit("Error, that should not occur")
                # store the cloudy roi
                geo_utils.write_tif(roi_data, out_transform, crs, out_file)

            ####################################################################################################
            # S1 HANDLING
            ####################################################################################################

            cloudys2_date = s2_util.get_date_from_name(str(tar_dir.parent.name))
            footprintvvvh, src_vvvh_crs = geo_utils.extract_bounds_from_tif(s1_elem / f"{s1_id}_VH.tif")
            src_vvvh_crs, src_vvvh_transform, src_vvvh_w, src_vvvh_h = geo_utils.get_meta_from_tif(s1_elem / f"{s1_id}_VH.tif")

            new_first = [elem-100 if elem==min(footprintvvvh[0]) else elem+100 for elem in footprintvvvh[0]]
            new_second = [elem-100 if elem==min(footprintvvvh[1]) else elem+100 for elem in footprintvvvh[1]]
            footprintvvvh_buffed = (new_first, new_second)

            footprintvvvh_wgs84 = geo_utils.transform_bounds(footprintvvvh_buffed, src_vvvh_crs, 'EPSG:4326')
            footprintvvvh_poly = shapely.Polygon([(footprintvvvh_wgs84[0][0], footprintvvvh_wgs84[1][0]), (footprintvvvh_wgs84[0][1], footprintvvvh_wgs84[1][1]),
                             (footprintvvvh_wgs84[0][2], footprintvvvh_wgs84[1][2]), (footprintvvvh_wgs84[0][3], footprintvvvh_wgs84[1][3]),
                             (footprintvvvh_wgs84[0][4], footprintvvvh_wgs84[1][4])])
            footprintvvvh_wkt = footprintvvvh_poly.wkt
            blacklist = []#["11fa3ef1-d4b3-5e62-8112-7d74680309e0", '682caee6-e894-5a0a-ad08-221f5ce2c697', '17d330a9-760b-5759-bcd6-51a2f0481706']
            found = False
            while not found:
                ids = []
                default_days = 10
                while len(ids) == 0:
                    start_date = cloudys2_date - datetime.timedelta(days=default_days)
                    end_date = cloudys2_date + datetime.timedelta(days=default_days)

                    sen1.refresh()
                    ids = sen1.get_products(footprintvvvh_wgs84, start_date, end_date)
                    #ids = s1_util.filter_products(ids, roi=footprint_poly, operationalMode="IW", productClass="S", productType="CARD-BS", 
                    #                              polarisationChannels="VV&VH", blacklist=blacklist)
                    ids = s1_util.filter_products(ids, operationalMode="IW", productClass="S", productType="IW_GRDH_1S", 
                                                  polarisationChannels="VV&VH", blacklist=blacklist)
                    #default_days += 30
                tar_product = s1_util.get_temporal_closest(cloudys2_date, ids)
                outs1 = create_out_dir(tar_product, s1_id, s1c_path)
                #print(f"Sen 1 Target Directior {outs1}")
                product_path_s1 = outs1.parent / "product.zip"
                product_path_ex_s1 = outs1.parent / "product"
                #check if folder exists
                if not product_path_s1.exists():
                    internet_id = tar_product['Id']
                    print(f"SENTINEL 1 DOWNLOAD {tar_product['Id']}")
                    sen1.download(internet_id, sen1.token, product_path_s1)  # will skip download if it already exists for a tile!
                outs1.mkdir(parents=False, exist_ok=True)

                safe_file = get_safe_file_path_s1(product_path_ex_s1)

                sentinel_1 = ProductIO.readProduct(str(safe_file))
                folder = str(safe_file.parent.name)
                modestamp = folder.split("_")[1]
                productstamp = folder.split("_")[2]
                polstamp = folder.split("_")[3]

                polarization = polstamp[2:4]
                if polarization == 'DV':
                    pols = 'VH,VV'
                elif polarization == 'DH':
                    pols = 'HH,HV'
                elif polarization == 'SH' or polarization == 'HH':
                    pols = 'HH'
                elif polarization == 'SV':
                    pols = 'VV'
                else:
                    print("Polarization error!")

                applyorbit = sar_processor.do_apply_orbit_file(sentinel_1)
                # border noise removal unnecarry? https://forum.step.esa.int/t/removing-border-noise/41233/3
                # no my data is older so I have to do it!
                # could have been checked in the safe file xml at "GRD Post Processing" https://forum.step.esa.int/t/removing-border-noise/41233/3
                borderremoved = sar_processor.do_border_noise_removal(applyorbit, pols)
                thermaremoved = sar_processor.do_thermal_noise_removal(borderremoved)
                calibrated = sar_processor.do_calibration(thermaremoved, polarization, pols, True)
                # This step has a bug and Big Earth Net also skipped it.
                #down_filtered = sar_processor.do_speckle_filtering(calibrated)
                down_filtered = calibrated
                terrain_corrected = sar_processor.do_terrain_correction(down_filtered, 0)
                subset = sar_processor.do_subset(terrain_corrected, footprintvvvh_wkt)
                db_scale = sar_processor.LinearToFromdB(subset)
                out_file_sen1 = outs1 / f"{s1_id}.tif"
                #print(f"writing in {out_file_sen1}")
                #result = sar_processor.extract_bands(subset, 'Sigma0_VH,Sigma0_VV')
                result = sar_processor.extract_bands(db_scale, 'Sigma0_VH_db,Sigma0_VV_db')
                ProductIO.writeProduct(result, str(out_file_sen1), 'GeoTIFF')
                with rasterio.open(out_file_sen1) as sar_processed:
                    band1 = sar_processed.read(1)
                if not np.any(band1):
                    blacklist.append(tar_product['Id'])
                    print(f"Added {tar_product['Id']} to da blacklist")
                    shutil.rmtree(product_path_s1.parent)
                else:
                    found = True
                    final_out_vh = str(out_file_sen1)[:-4] + "vh.tif"
                    final_out_vv = str(out_file_sen1)[:-4] + "vv.tif"
                    final_out_complete = str(out_file_sen1)[:-4] + "complete.tif"
                    geo_utils.reproject_raster(out_file_sen1, final_out_complete, src_vvvh_crs, src_vvvh_transform, src_vvvh_w, src_vvvh_h)
                    s1_roi_data, s1_out_transform, s1_crs = geo_utils.extract_roi(final_out_complete, footprintvvvh, src_vvvh_crs)
                    geo_utils.write_tif(s1_roi_data[0].reshape((1,s1_roi_data.shape[1],s1_roi_data.shape[2])), s1_out_transform, s1_crs, final_out_vh)
                    geo_utils.write_tif(s1_roi_data[1].reshape((1,s1_roi_data.shape[1],s1_roi_data.shape[2])), s1_out_transform, s1_crs, final_out_vv)
                del subset
                del applyorbit
                del thermaremoved
                del calibrated
                del borderremoved
                del down_filtered
                del terrain_corrected                  

    full_tile_path = tar_dir.parent
    if (full_tile_path / "product.zip").exists():
        os.remove((full_tile_path / "product.zip"))
    if product_path_s1.exists():
        os.remove(product_path_s1)
    if (full_tile_path / "product").exists():
        shutil.rmtree((full_tile_path / "product"))
    if product_path_ex_s1.exists():
        shutil.rmtree(product_path_ex_s1)

def get_closest_tile(date, footprint, tile_id, min_cloud_cover, max_cloud_cover, s2_id, s2c_path, s2token):
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
    internet_id = tar_product['Id']
    out = create_out_dir(tar_product, s2_id, s2c_path)
    product_path = out.parent / "product.zip"
    product_path_ex = out.parent / "product"
    #check if folder exists
    if not product_path.exists():
        #This cloud cover is for one whole tile and not a specific subsample of a tile!
        with open(Path("C:/Users/m_str/Documents/Phd_data/Projects/SSL/data/BigEarthNet_Clouds/log.csv"), 'a', newline='') as csvfile:
            atts = tar_product["Attributes"]
            cc_idx = sentinel_util.get_attribute_index(atts, "cloudCover")
            csvfile.write(f"{internet_id}, {tile_id}, {atts[cc_idx]['Value']}, {abs(date-s2_util.get_date_from_name(tar_product['Name'])).days}\n")
        s2token.download(internet_id, s2token.token, product_path)  # will skip download if it already exists for a tile!
    out.mkdir(parents=False, exist_ok=True)
    return out, product_path_ex
    #dictionary with keys '@odata.context' and 'value'. value contains products
    #ids["value"] is a list of products, where each product is a dictionary
    #print(len(ids["value"]))
    #this is the dictionary of the first product
    #for dic in ids["value"]:
    #    if tile_id in dic["Name"]:
    #        internet_id = dic['Id']
            #sen2.download(id, sen2.token)
            #if "S2A_MSIL2A_20170613T101031_N0500_R022_T33UUP_20231008T194656.SAFE" in dic["Name"]:
            #    for elem in dic.keys():
            #        print(elem, dic[elem])
    
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


def create_out_dir(tar_product, s2_id, s2c_path):
    tile_dir = "_".join(tar_product["Name"].split("_")[:-1])
    sub_tile_dir = f"{tile_dir}_{s2_id.split('_')[-2]}_{s2_id.split('_')[-1]}"
    out = s2c_path / tile_dir / sub_tile_dir
    return out


if __name__ == "__main__":
    main()
