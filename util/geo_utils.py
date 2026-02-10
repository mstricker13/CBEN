# util file for processing geographic information

from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.mask import mask
import sys
import shapely

def extract_roi(in_file, roi, src_crs):
    with rasterio.open(in_file) as geo_file:
        roi = transform_bounds(roi, src_crs, geo_file.crs)
        roi_shape = [shapely.Polygon(zip(roi[0], roi[1]))]
        roi_data, out_transform = mask(geo_file, roi_shape, crop = True)
        crs = geo_file.crs
    return roi_data, out_transform, crs


def write_tif(save_data, out_transform, crs, tar_dir):
    with rasterio.open(
        tar_dir,
        'w',
        driver='GTiff',
        height=save_data.shape[1],
        width=save_data.shape[2],
        count=save_data.shape[0],
        dtype=save_data.dtype,
        crs=crs,
        transform=out_transform,
    ) as dst:
        dst.write(save_data[0], 1)

def get_meta_from_tif(tif_path: Path):
    with rasterio.open(tif_path) as src:
        src_crs =src.crs
        src_transform=src.transform
        src_w=src.width
        src_h=src.height
    return src_crs, src_transform, src_w, src_h

def extract_bounds_from_tif(tif_path: Path):
    with rasterio.open(tif_path) as src:
        xs = [src.bounds[0], src.bounds[0], src.bounds[2], src.bounds[2], src.bounds[0]]
        ys = [src.bounds[3], src.bounds[1], src.bounds[1], src.bounds[3], src.bounds[3]]
        xy = (xs, ys)
        crs = src.crs

        #TODO might refactor return type with shapely polygon
        #shapely.Polygon([(xy[0][0], xy[1][0]), (xy[0][1], xy[1][1]),
        #                     (xy[0][2], xy[1][2]), (xy[0][3], xy[1][3]),
        #                     (xy[0][4], xy[1][4])])

    return xy, crs

def transform_bounds(in_bounds, src_crs, dst_crs):
    return rasterio.warp.transform(src_crs, dst_crs, in_bounds[0], in_bounds[1])

def reproject_raster(raster_in, tar_dir, tar_crs, transform=None, height=None, width=None):
    #https://rasterio.readthedocs.io/en/stable/api/rasterio.warp.html#rasterio.warp.reproject
    with rasterio.open(raster_in) as src:
        src_crs = src.crs
        transform_s, width_s, height_s = rasterio.warp.calculate_default_transform(src_crs, tar_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()

        if transform is None:
            transform = transform_s
        if width is None:
            width = width_s
        if height is None:
            height = height_s

        kwargs.update({
            'crs': tar_crs,
            'transform': transform,
            'width': width,
            'height': height})

        with rasterio.open(tar_dir, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=tar_crs,
                    resampling=rasterio.warp.Resampling.cubic)


def get_val_at_lat_lon(file_path, lat, lon, band_idx):
    res = []
    with rasterio.open(file_path) as data:
        field = data.read(1)
        row, col = data.index(lon, lat)
        value = field[int(row), int(col)]
        return value
