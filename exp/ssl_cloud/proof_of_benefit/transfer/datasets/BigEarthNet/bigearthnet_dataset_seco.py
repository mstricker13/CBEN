'''
read data from geotiff files.
modified from SeCo, to be final checked.
'''

import json
from pathlib import Path

import numpy as np
import os
import rasterio
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive, download_url
import cv2
import pandas
import rasterio
import sys

ALL_BANDS_S2 = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
ALL_BANDS_S1 = ['VV','VH']
RGB_BANDS = ['B04', 'B03', 'B02']

#Mean,std of VV is -12.643863376506465,5.133493870383747, mean,std of VH is -19.352558587824998, 5.5905057382245955
BAND_STATS = {
    'mean': {
        'B01': 340.76769064,
        'B02': 429.9430203,
        'B03': 614.21682446,
        'B04': 590.23569706,
        'B05': 950.68368468,
        'B06': 1792.46290469,
        'B07': 2075.46795189,
        'B08': 2218.94553375,
        'B8A': 2266.46036911,
        'B09': 2246.0605464,
        'B11': 1594.42694882,
        'B12': 1009.32729131,
        'VV': -12.643863376506465,
        'VH': -19.352558587824998
    },
    'std': {
        'B01': 554.81258967,
        'B02': 572.41639287,
        'B03': 582.87945694,
        'B04': 675.88746967,
        'B05': 729.89827633,
        'B06': 1096.01480586,
        'B07': 1273.45393088,
        'B08': 1365.45589904,
        'B8A': 1356.13789355,
        'B09': 1302.3292881,
        'B11': 1079.19066363,
        'B12': 818.86747235,
        'VV': 5.133493870383747,
        'VH': 5.5905057382245955
    }
}

LABELS = [
    'Agro-forestry areas', 'Airports',
    'Annual crops associated with permanent crops', 'Bare rock',
    'Beaches, dunes, sands', 'Broad-leaved forest', 'Burnt areas',
    'Coastal lagoons', 'Complex cultivation patterns', 'Coniferous forest',
    'Construction sites', 'Continuous urban fabric',
    'Discontinuous urban fabric', 'Dump sites', 'Estuaries',
    'Fruit trees and berry plantations', 'Green urban areas',
    'Industrial or commercial units', 'Inland marshes', 'Intertidal flats',
    'Land principally occupied by agriculture, with significant areas of '
    'natural vegetation', 'Mineral extraction sites', 'Mixed forest',
    'Moors and heathland', 'Natural grassland', 'Non-irrigated arable land',
    'Olive groves', 'Pastures', 'Peatbogs', 'Permanently irrigated land',
    'Port areas', 'Rice fields', 'Road and rail networks and associated land',
    'Salines', 'Salt marshes', 'Sclerophyllous vegetation', 'Sea and ocean',
    'Sparsely vegetated areas', 'Sport and leisure facilities',
    'Transitional woodland/shrub', 'Vineyards', 'Water bodies', 'Water courses'
]

NEW_LABELS = [
    'Urban fabric',
    'Industrial or commercial units',
    'Arable land',
    'Permanent crops',
    'Pastures',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas',
    'Broad-leaved forest',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland and sparsely vegetated areas',
    'Moors, heathland and sclerophyllous vegetation',
    'Transitional woodland, shrub',
    'Beaches, dunes, sands',
    'Inland wetlands',
    'Coastal wetlands',
    'Inland waters',
    'Marine waters'
]

GROUP_LABELS = {
    'Continuous urban fabric': 'Urban fabric',
    'Discontinuous urban fabric': 'Urban fabric',
    'Non-irrigated arable land': 'Arable land',
    'Permanently irrigated land': 'Arable land',
    'Rice fields': 'Arable land',
    'Vineyards': 'Permanent crops',
    'Fruit trees and berry plantations': 'Permanent crops',
    'Olive groves': 'Permanent crops',
    'Annual crops associated with permanent crops': 'Permanent crops',
    'Natural grassland': 'Natural grassland and sparsely vegetated areas',
    'Sparsely vegetated areas': 'Natural grassland and sparsely vegetated areas',
    'Moors and heathland': 'Moors, heathland and sclerophyllous vegetation',
    'Sclerophyllous vegetation': 'Moors, heathland and sclerophyllous vegetation',
    'Inland marshes': 'Inland wetlands',
    'Peatbogs': 'Inland wetlands',
    'Salt marshes': 'Coastal wetlands',
    'Salines': 'Coastal wetlands',
    'Water bodies': 'Inland waters',
    'Water courses': 'Inland waters',
    'Coastal lagoons': 'Marine waters',
    'Estuaries': 'Marine waters',
    'Sea and ocean': 'Marine waters'
}

#cd /code/Projects/ssl/src/exp/ssl_cloud/proof_of_benefit
#python
#from transfer.datasets.BigEarthNet.bigearthnet_dataset_seco import calcs1stats
#calcs1stats()
#Mean,std of VV is -12.643863376506465,5.133493870383747, mean,std of VH is -19.352558587824998, 5.5905057382245955
def calcs1stats():
    print("Running")
    datapath = "/data/Phd_data/Projects/SSL/data/BigEarthNet/"
    metapath = os.path.join(datapath, "metadata.parquet")
    meta = pandas.read_parquet(metapath)
    total = (120*120*237871)
    vvs = np.zeros(total)
    vhs = np.zeros(total)
    idx = 0
    per = 0
    for index, row in meta.iterrows():
        #['patch_id', 'labels', 'split', 'country', 's1_name', 's2v1_name', 'contains_seasonal_snow', 'contains_cloud_or_shadow']
        if not row["contains_seasonal_snow"]: #useless as its always false
            if not row["contains_cloud_or_shadow"]: #useless as its always false
                if row["split"] == "train": #test, validation
                    s1folder = "_".join(row["s1_name"].split("_")[:-3])
                    s1path = os.path.join(datapath,"BigEarthNet-S1", s1folder, row["s1_name"])
                    vv = os.path.join(s1path, row["s1_name"]+"_VV.tif")
                    vh = os.path.join(s1path, row["s1_name"]+"_VH.tif")
                    with rasterio.open(vv) as vvband:
                        vvval = vvband.read(1)
                    with rasterio.open(vh) as vhband:
                        vhval = vhband.read(1)
                    for x in range(vvval.shape[0]):
                        for y in range(vvval.shape[1]):
                            vvs[idx]=vvval[x,y]
                            vhs[idx]=vhval[x,y]
                            idx+=1
                            if idx % 34253424 == 0:
                                per+=1
                                print(f"running {per}%")
    meanvv = np.mean(vvs)
    meanvh = np.mean(vhs)
    stdvv = np.std(vvs)
    stdvh = np.std(vhs)
    print(f"Mean,std of VV is {meanvv},{stdvv}, mean,std of VH is {meanvh}, {stdvh}")

def getproperparent(tileid, datapath, truetar):
    candidates = []
    subfolders = [ f.path for f in os.scandir(datapath) if f.is_dir() ]
    for elem in subfolders:
        if tileid in elem:
            candidates.append(elem)
    for candidate in candidates:
        subfolders = [ f.path for f in os.scandir(candidate) if f.is_dir() ]
        for sub in subfolders:
            if truetar in sub:
                return sub
    print(tileid)
    print(datapath)
    print(truetar)
    sys.exit("BAD IMPLEMENTATION")
    
#TODO check missing files due to system restarts and redownload these specifically!
def calcs2cloudystats():
    datapath = "/data/Phd_data/Projects/SSL/data/BigEarthNet_CloudsR/"
    datapathold = "/data/Phd_data/Projects/SSL/data/BigEarthNet/"
    metapath = os.path.join(datapathold, "metadata.parquet")
    meta = pandas.read_parquet(metapath)
    total = (120*120*237871)
    b01s = np.zeros(total)
    b02s = np.zeros(total)
    b03s = np.zeros(total)
    b04s = np.zeros(total)
    b05s = np.zeros(total)
    b06s = np.zeros(total)
    b07s = np.zeros(total)
    b08s = np.zeros(total)
    b8as = np.zeros(total)
    b09s = np.zeros(total)
    b11s = np.zeros(total)
    b12s = np.zeros(total)
    idx = 0
    per = 0
    print("START")
    for index, row in meta.iterrows():
        #['patch_id', 'labels', 'split', 'country', 's1_name', 's2v1_name', 'contains_seasonal_snow', 'contains_cloud_or_shadow']
        if not row["contains_seasonal_snow"]: #useless as its always false
            if not row["contains_cloud_or_shadow"]: #useless as its always false
                if row["split"] == "train": #test, validation
                    s2path = getproperparent(row["patch_id"].split("_")[5], os.path.join(datapath, "BigEarthNet-S2"), row["patch_id"])
                    b01 = os.path.join(s2path, row["patch_id"]+"_B01.tif")
                    b02 = os.path.join(s2path, row["patch_id"]+"_B02.tif")
                    b03 = os.path.join(s2path, row["patch_id"]+"_B03.tif")
                    b04 = os.path.join(s2path, row["patch_id"]+"_B04.tif")
                    b05 = os.path.join(s2path, row["patch_id"]+"_B05.tif")
                    b06 = os.path.join(s2path, row["patch_id"]+"_B06.tif")
                    b07 = os.path.join(s2path, row["patch_id"]+"_B07.tif")
                    b08 = os.path.join(s2path, row["patch_id"]+"_B08.tif")
                    b8a = os.path.join(s2path, row["patch_id"]+"_B8A.tif")
                    b09 = os.path.join(s2path, row["patch_id"]+"_B09.tif")
                    b11 = os.path.join(s2path, row["patch_id"]+"_B11.tif")
                    b12 = os.path.join(s2path, row["patch_id"]+"_B12.tif")
                    with rasterio.open(b01) as b01band:
                        b01val = b01band.read(1)
                    with rasterio.open(b02) as b02band:
                        b02val = b02band.read(1)
                    with rasterio.open(b03) as b03band:
                        b03val = b03band.read(1)
                    with rasterio.open(b04) as b04band:
                        b04val = b04band.read(1)
                    with rasterio.open(b05) as b05band:
                        b05val = b05band.read(1)
                    with rasterio.open(b06) as b06band:
                        b06val = b06band.read(1)
                    with rasterio.open(b07) as b07band:
                        b07val = b07band.read(1)
                    with rasterio.open(b08) as b08band:
                        b08val = b08band.read(1)
                    with rasterio.open(b8a) as b8aband:
                        b8aval = b8aband.read(1)
                    with rasterio.open(b09) as b09band:
                        b09val = b09band.read(1)
                    with rasterio.open(b11) as b11band:
                        b11val = b11band.read(1)
                    with rasterio.open(b12) as b12band:
                        b12val = b12band.read(1)
                    for x in range(b01val.shape[0]):
                        for y in range(b01val.shape[1]):
                            #TODO the channels have different shapes
                            b01s[idx]=b01val[x,y]
                            b02s[idx]=b02val[x,y]
                            b03s[idx]=b03val[x,y]
                            b04s[idx]=b04val[x,y]
                            b05s[idx]=b05val[x,y]
                            b06s[idx]=b06val[x,y]
                            b07s[idx]=b07val[x,y]
                            b08s[idx]=b08val[x,y]
                            b8as[idx]=b8aval[x,y]
                            b09s[idx]=b09val[x,y]
                            b11s[idx]=b11val[x,y]
                            b12s[idx]=b12val[x,y]
                            idx+=1
                            if idx % 3425342 == 0:
                                per+=0.1
                                print(f"running {per}%")
    meanb01 = np.mean(b01s)
    stdb01 = np.std(b01s)

    meanb02 = np.mean(b02s)
    stdb02 = np.std(b02s)

    meanb03 = np.mean(b03s)
    stdb03 = np.std(b03s)
    
    meanb04 = np.mean(b04s)
    stdb04 = np.std(b04s)

    meanb05 = np.mean(b05s)
    stdb05 = np.std(b05s)

    meanb06 = np.mean(b06s)
    stdb06 = np.std(b06s)

    meanb07 = np.mean(b07s)
    stdb07 = np.std(b07s)

    meanb08 = np.mean(b08s)
    stdb08 = np.std(b08s)

    meanb8a = np.mean(b8as)
    stdb8a = np.std(b8as)

    meanb09 = np.mean(b09s)
    stdb09 = np.std(b09s)

    meanb11 = np.mean(b11s)
    stdb11 = np.std(b11s)

    meanb12 = np.mean(b12s)
    stdb12 = np.std(b12s)
    print(f"Mean,std of b01 is {meanb01},{stdb01}")
    print(f"Mean,std of b02 is {meanb02},{stdb02}")
    print(f"Mean,std of b03 is {meanb03},{stdb03}")
    print(f"Mean,std of b04 is {meanb04},{stdb04}")
    print(f"Mean,std of b05 is {meanb05},{stdb05}")
    print(f"Mean,std of b06 is {meanb06},{stdb06}")
    print(f"Mean,std of b07 is {meanb07},{stdb07}")
    print(f"Mean,std of b08 is {meanb08},{stdb08}")
    print(f"Mean,std of b8a is {meanb8a},{stdb8a}")
    print(f"Mean,std of b09 is {meanb09},{stdb09}")
    print(f"Mean,std of b11 is {meanb11},{stdb11}")
    print(f"Mean,std of b12 is {meanb12},{stdb12}")

                            

def normalize(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


class Bigearthnet(Dataset):
    url = 'http://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz'
    subdir = 'BigEarthNet-v1.0'
    list_file = {
        'train': 'https://storage.googleapis.com/remote_sensing_representations/bigearthnet-train.txt',
        'val': 'https://storage.googleapis.com/remote_sensing_representations/bigearthnet-val.txt',
        'test': 'https://storage.googleapis.com/remote_sensing_representations/bigearthnet-test.txt'
    }
    bad_patches = [
        'http://bigearth.net/static/documents/patches_with_seasonal_snow.csv',
        'http://bigearth.net/static/documents/patches_with_cloud_and_shadow.csv'
    ]

    def __init__(self, root, split, bands=None, transform=None, target_transform=None, download=False, use_new_labels=True, normalize=False):
        self.root = Path(root)
        self.split = split
        self.bands = bands if bands is not None else RGB_BANDS  #will be ['s1','s2a']
        self.transform = transform
        self.target_transform = target_transform
        self.use_new_labels = use_new_labels
        self.normalize = normalize

        if download:
            download_and_extract_archive(self.url, self.root)
            download_url(self.list_file[self.split], self.root, f'{self.split}.txt')
            for url in self.bad_patches:
                download_url(url, self.root)

        self.samples = []
        metapath = os.path.join(root, "metadata.parquet")
        meta = pandas.read_parquet(metapath)
        for index, row in meta.iterrows():
            #['patch_id', 'labels', 'split', 'country', 's1_name', 's2v1_name', 'contains_seasonal_snow', 'contains_cloud_or_shadow']
            if not row["contains_seasonal_snow"]: #useless as its always false
                if not row["contains_cloud_or_shadow"]: #useless as its always false
                    if row["split"] == split: #test, validation
                        s1folder = "_".join(row["s1_name"].split("_")[:-3])
                        s2folder = "_".join(row["patch_id"].split("_")[:-2])
                        s1path = os.path.join(root,"BigEarthNet-S1", s1folder, row["s1_name"])
                        s2path = os.path.join(root,"BigEarthNet-S2", s2folder, row["patch_id"])
                        tar = row["labels"]  # e.g. ['Broad-leaved forest', 'Coniferous forest', 'Inland waters', 'Mixed forest', 'Pastures']
                        self.samples.append((s1path, s2path, tar))

    def __getitem__(self, index):
        s1path  = self.samples[index][0]
        s2path  = self.samples[index][1]
        tar  = self.samples[index][2]
        s2ID  = os.path.basename(os.path.normpath(s2path))
        s1ID  = os.path.basename(os.path.normpath(s1path))
        #path = self.samples[index]
        #patch_id = path.name

        channelss1 = []
        channelss2 = []
        if 's2a' in self.bands:
            for b in ALL_BANDS_S2:
                ch = rasterio.open(os.path.join(s2path, f'{s2ID}_{b}.tif')).read(1)
                if self.normalize:
                    ch = normalize(ch, mean=BAND_STATS['mean'][b], std=BAND_STATS['std'][b])
                ch = cv2.resize(ch, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
                channelss2.append(ch)
        if 's1' in self.bands:
            for b in ALL_BANDS_S1:
                ch = rasterio.open(os.path.join(s1path, f'{s1ID}_{b}.tif')).read(1)
                if self.normalize:
                    ch = normalize(ch, mean=BAND_STATS['mean'][b], std=BAND_STATS['std'][b])
                ch = cv2.resize(ch, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
                channelss1.append(ch)
        #imgs1 = np.dstack(channelss1)
        imgs1 = np.stack(channelss1, axis=0)
        #imgs2 = np.dstack(channelss2)
        imgs2 = np.stack(channelss2, axis=0)
        target = self.get_multihot_new_no_group(tar)

        if self.transform is not None:
            imgs1 = self.transform(imgs1)
            imgs2 = self.transform(imgs2)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return imgs1, imgs2, target

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def get_multihot_new_no_group(labels):
        target = np.zeros((len(NEW_LABELS),), dtype=np.float32)
        for label in labels:
            target[NEW_LABELS.index(label)] = 1
        return target


    @staticmethod
    def get_multihot_old(labels):
        target = np.zeros((len(LABELS),), dtype=np.float32)
        for label in labels:
            target[LABELS.index(label)] = 1
        return target

    @staticmethod
    def get_multihot_new(labels):
        target = np.zeros((len(NEW_LABELS),), dtype=np.float32)
        for label in labels:
            if label in GROUP_LABELS:
                target[NEW_LABELS.index(GROUP_LABELS[label])] = 1
            elif label not in set(NEW_LABELS):
                continue
            else:
                target[NEW_LABELS.index(label)] = 1
        return target


class BigearthnetClouds(Dataset):
    url = 'http://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz'
    subdir = 'BigEarthNet-v1.0'
    list_file = {
        'train': 'https://storage.googleapis.com/remote_sensing_representations/bigearthnet-train.txt',
        'val': 'https://storage.googleapis.com/remote_sensing_representations/bigearthnet-val.txt',
        'test': 'https://storage.googleapis.com/remote_sensing_representations/bigearthnet-test.txt'
    }
    bad_patches = [
        'http://bigearth.net/static/documents/patches_with_seasonal_snow.csv',
        'http://bigearth.net/static/documents/patches_with_cloud_and_shadow.csv'
    ]

    def __init__(self, root, split, bands=None, transform=None, target_transform=None, download=False, use_new_labels=True, normalize=False):
        self.root = Path(root)
        self.split = split
        self.bands = bands if bands is not None else RGB_BANDS  #will be ['s1','s2a']
        self.transform = transform
        self.target_transform = target_transform
        self.use_new_labels = use_new_labels
        self.normalize = normalize

        datapathold = "/data/Phd_data/Projects/SSL/data/BigEarthNet/"
        metapath = os.path.join(datapathold, "metadata.parquet")
        meta = pandas.read_parquet(metapath)

        if download:
            download_and_extract_archive(self.url, self.root)
            download_url(self.list_file[self.split], self.root, f'{self.split}.txt')
            for url in self.bad_patches:
                download_url(url, self.root)

        self.samples = []
        #metapath = os.path.join(root, "metadata.parquet")
        #meta = pandas.read_parquet(metapath)
        for index, row in meta.iterrows():
            #['patch_id', 'labels', 'split', 'country', 's1_name', 's2v1_name', 'contains_seasonal_snow', 'contains_cloud_or_shadow']
            if not row["contains_seasonal_snow"]: #useless as its always false
                if not row["contains_cloud_or_shadow"]: #useless as its always false
                    if row["split"] == split: #test, validation
                        s1folder = "_".join(row["s1_name"].split("_")[:-3])
                        s1path = os.path.join(datapathold,"BigEarthNet-S1", s1folder, row["s1_name"])
                        s2path = getproperparent(row["patch_id"].split("_")[5], os.path.join(root, "BigEarthNet-S2"), row["patch_id"])
                        tar = row["labels"]  # e.g. ['Broad-leaved forest', 'Coniferous forest', 'Inland waters', 'Mixed forest', 'Pastures']
                        self.samples.append((s1path, s2path, tar))

    def __getitem__(self, index):
        s1path  = self.samples[index][0]
        s2path  = self.samples[index][1]
        tar  = self.samples[index][2]
        s2ID  = os.path.basename(os.path.normpath(s2path))
        s1ID  = os.path.basename(os.path.normpath(s1path))
        #path = self.samples[index]
        #patch_id = path.name

        channelss1 = []
        channelss2 = []
        if 's2a' in self.bands:
            for b in ALL_BANDS_S2:
                ch = rasterio.open(os.path.join(s2path, f'{s2ID}_{b}.tif')).read(1)
                if self.normalize:
                    ch = normalize(ch, mean=BAND_STATS['mean'][b], std=BAND_STATS['std'][b])
                ch = cv2.resize(ch, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
                channelss2.append(ch)
        if 's1' in self.bands:
            for b in ALL_BANDS_S1:
                ch = rasterio.open(os.path.join(s1path, f'{s1ID}_{b}.tif')).read(1)
                if self.normalize:
                    ch = normalize(ch, mean=BAND_STATS['mean'][b], std=BAND_STATS['std'][b])
                ch = cv2.resize(ch, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
                channelss1.append(ch)
        #imgs1 = np.dstack(channelss1)
        imgs1 = np.stack(channelss1, axis=0)
        #imgs2 = np.dstack(channelss2)
        imgs2 = np.stack(channelss2, axis=0)
        target = self.get_multihot_new_no_group(tar)

        if self.transform is not None:
            imgs1 = self.transform(imgs1)
            imgs2 = self.transform(imgs2)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return imgs1, imgs2, target

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def get_multihot_new_no_group(labels):
        target = np.zeros((len(NEW_LABELS),), dtype=np.float32)
        for label in labels:
            target[NEW_LABELS.index(label)] = 1
        return target


    @staticmethod
    def get_multihot_old(labels):
        target = np.zeros((len(LABELS),), dtype=np.float32)
        for label in labels:
            target[LABELS.index(label)] = 1
        return target

    @staticmethod
    def get_multihot_new(labels):
        target = np.zeros((len(NEW_LABELS),), dtype=np.float32)
        for label in labels:
            if label in GROUP_LABELS:
                target[NEW_LABELS.index(GROUP_LABELS[label])] = 1
            elif label not in set(NEW_LABELS):
                continue
            else:
                target[NEW_LABELS.index(label)] = 1
        return target

if __name__ == '__main__':
    import os
    import argparse
    from bigearthnet_dataset_seco_lmdb import make_lmdb
    import time
    import torch
    from torchvision import transforms
    ## change02: `pip install opencv-torchvision-transforms-yuzhiyang`
    from cvtorchvision import cvtransforms

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/mnt/d/codes/SSL_examples/datasets/BigEarthNet')
    parser.add_argument('--save_dir', type=str, default='/mnt/d/codes/SSL_examples/datasets/BigEarthNet/dataload_op1_lmdb')
    parser.add_argument('--make_lmdb_dataset', type=bool, default=False)
    parser.add_argument('--download', type=bool, default=False)
    args = parser.parse_args()

    make_lmdb_dataset = args.make_lmdb_dataset
    all_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    RGB_bands = ['B04', 'B03', 'B02']
    test_loading_time = True
    
    if make_lmdb_dataset:
    
        start_time = time.time()
        train_dataset = Bigearthnet(
            root=args.data_dir,
            split='train',
            bands=all_bands
        )
    
        make_lmdb(train_dataset, lmdb_file=os.path.join(args.save_dir, 'train_B12.lmdb'))

        val_dataset = Bigearthnet(
            root=args.data_dir,
            split='val',
            bands=all_bands
        )

        make_lmdb(val_dataset, lmdb_file=os.path.join(args.save_dir, 'val_B12.lmdb'))
        print('LMDB dataset created: %s seconds.' % (time.time()-start_time))

    '''
    if test_loading_time:
        ## change03: use cvtransforms to process non-PIL image
        train_transforms = cvtransforms.Compose([cvtransforms.Resize((128, 128)),
                                               cvtransforms.ToTensor()])
        train_dataset = Bigearthnet(root=args.data_dir,
                                    split='train',
                                    transform = train_transforms
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, num_workers=4)    
        start_time = time.time()

        runs = 5
        for i in range(runs):
            for idx, (img,target) in enumerate(train_loader):
                print(idx)
                if idx > 188:
                    break

        print("Mean Time over 5 runs: ", (time.time() - start_time) / runs)
    '''