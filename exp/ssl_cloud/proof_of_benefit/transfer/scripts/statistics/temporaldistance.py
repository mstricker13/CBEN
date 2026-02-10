from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
import os
import shutil
import sys
import pandas as pd
from datetime import datetime

#cd /code/Projects/ssl/src/exp/ssl_cloud/proof_of_benefit/transfer/scripts/statistics
def main():
    data_pathC = Path("/data/Phd_data/Projects/SSL/data/BigEarthNet_CloudsR/BigEarthNet-S2")
    data_path = Path("/data/Phd_data/Projects/SSL/data/BigEarthNet/metadata.parquet")
    meta = pd.read_parquet(data_path)
    #print(meta.columns.tolist())
    #print(meta["patch_id"][0])
    s2tos1 = list()
    s2tos2C = list()
    s2Ctos1 = list()
    format_string = "%Y%m%dT%H%M%S" #20170613T101031
    for i in range(480037):
        if i % 1000 == 0:
            print(i)
        s2 = meta["patch_id"][i]  #S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57
        s2Time = s2.split("_")[2]
        s1 = meta["s1_name"][i] #S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57
        s1Time = s1.split("_")[4]
        #s2 = meta["s2v1_name"][i] #S2A_MSIL2A_20170613T101031_26_57
        s2_id = s2.split("_")[-3]
        #find all dirs with correct tile id
        cloud_dirs = [entry for entry in data_pathC.iterdir() if entry.is_dir()]
        matchingClouds = []
        for cloud_dir in cloud_dirs:
            if s2_id in str(cloud_dir):
                matchingClouds.append(cloud_dir)
        #go into matchin tile ids and then check if dir date is identical, this is the corresponding file
        found = False
        for match in matchingClouds:
            possibles = [entry for entry in match.iterdir() if entry.is_dir()][0]
            possibleTime = possibles.name.split("_")[2]
            if possibleTime == s2Time:
                s2CTime = match.name.split("_")[2]
                found = True
                break
        if not found:
            sys.exit("Not found lol")
        s2date = datetime.strptime(s2Time, format_string)
        s1date = datetime.strptime(s1Time, format_string)
        s2Cdate = datetime.strptime(s2CTime, format_string)
        
        s2Ctos1.append(abs(s2Cdate - s1date))
        s2tos2C.append(abs(s2date - s2Cdate))
        s2tos1.append(abs(s2date - s1date))

    print("s2Ctos1")
    print(pd.Series(s2Ctos1).describe())
    print("s2tos2C")
    print(pd.Series(s2tos2C).describe())
    print("s2tos1")
    print(pd.Series(s2tos1).describe())
    


if __name__ == "__main__":
    main()
