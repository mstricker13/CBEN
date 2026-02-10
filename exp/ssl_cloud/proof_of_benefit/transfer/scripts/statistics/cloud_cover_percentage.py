from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
import os
import shutil
import sys
import pandas as pd

#cd /code/Projects/ssl/src/exp/ssl_cloud/proof_of_benefit/transfer/scripts/statistics
def main():
    data_path = Path("/data/Phd_data/Projects/SSL/data/BigEarthNet_CloudsR/BigEarthNet-S2")
    all_dir = [entry for entry in data_path.iterdir() if entry.is_dir()]
    cloud_cover = list()
    cloud_shadow = list()
    thin_cirrus = list()
    i = 0
    for elem in all_dir:
        print(i)
        i += 1
        zip_path = elem / "product.zip"
        extract_dir = elem / "extracted"
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        meta_file = elem / [entry for entry in extract_dir.iterdir() if entry.is_dir()][0] / "MTD_MSIL2A.xml"
        tree = ET.parse(meta_file)
        root = tree.getroot()
        for child in root:
            if child.tag == "{https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd}Quality_Indicators_Info":
                for quality in child:
                    if quality.tag == "Cloud_Coverage_Assessment":
                        cloud_cover.append(float(quality.text))
                    if quality.tag == "Image_Content_QI":
                        for image_Content_QI in quality:
                            if image_Content_QI.tag == "CLOUD_SHADOW_PERCENTAGE":
                                cloud_shadow.append(float(image_Content_QI.text))
                            if image_Content_QI.tag == "THIN_CIRRUS_PERCENTAGE":
                                thin_cirrus.append(float(image_Content_QI.text))
        shutil.rmtree(extract_dir)
    print("CLOUD COVER")
    print(pd.Series(cloud_cover).describe())
    print("CLOUD SHADOW")
    print(pd.Series(cloud_shadow).describe())
    print("THIN CIRRUS")
    print(pd.Series(thin_cirrus).describe())


if __name__ == "__main__":
    main()
