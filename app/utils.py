import numpy as np
import shapely
import geopandas as gpd
import xarray as xr


def shift_data(ds):
    """
    Take in an Xarray dataset and shift the latitudes by 180 degrees.
    """
    ds.coords['x'] = (ds.coords['x'] + 180) % 360 - 180
    ds = ds.sortby(ds.x)

    return ds


def process_dataset(dataset):
    """
    Take in an Xarray dataset, rename the latitude and longitude columns, and shift the latitudes by 180 degrees.
    """
    if('longitude' in dataset.coords and 'latitude' in dataset.coords):
        dataset = dataset.rename({ 'longitude':'x', 'latitude':'y'})
    
    if('L' in dataset.coords):
        dataset = dataset.rename({ 'L':'time'})
    
    if('50%' in dataset.data_vars):
        dataset = dataset.rename({ '50%':'perc'})
    
    dataset.rio.write_crs("epsg:4326", inplace=True)
    dataset = shift_data(dataset)

    return dataset


def create_bbox_from_coords(x_min, x_max, y_min, y_max, crs=4326):
    """
    Create a GeoPandas GeoDataFrame from a list of coordinates with the CRS specified on input.
    """
    top_left = (x_min, y_max)
    top_right = (x_max, y_max)
    bottom_left = (x_min, y_min)
    bottom_right = (x_max, y_min)
    
    bbox_geom = shapely.Polygon([top_left, top_right, bottom_right, bottom_left])
    bbox = gpd.GeoDataFrame(geometry=[bbox_geom], crs=crs)
    
    return bbox