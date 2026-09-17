import functools
import os
from datetime import datetime

import gcsfs
import geopandas as gpd
import pandas as pd
import shapely
import xarray as xr

# calculate the initial conditions from today's day, month, and year
# in general, the month (and potentially year) roll back one month
# however, right before the update occurs, there are times when the month looks back two months

today = datetime.today()

# Determine how many months to go back
# months_back = 1 if today.day >= 15 else 2

# Calculate the target month and year
# target_month = today.month - months_back
# target_year = today.year

# Handle year rollover (handles both going back to previous December and beyond)
# while target_month <= 0:
#     target_month += 12
#     target_year -= 1

target_year = 2026
target_month = 9

if target_month == 1:
    target_month = 12
    target_year = target_year - 1
else:
    target_month -= 1

year_ic = str(target_year)
month_ic = str(target_month).zfill(2)

# generate the list of date we expect to find for historical data
historical_dates = [
    date.strftime('%Y-%m-%d')
    for date in pd.date_range(start='1991-01-01', end=f'{year_ic}-{month_ic}-01', freq='MS')
]
forecast_dates = [
    date.strftime('%Y-%m-%d')
    for date in pd.date_range(start=f'{year_ic}-{month_ic}-01', freq='MS', periods=7)
][1:]

BUCKET = os.getenv('BUCKET_NAME')


def create_bbox_from_coords(
    x_min: float, y_min: float, x_max: float, y_max: float, crs: int = 4326
) -> gpd.GeoDataFrame:
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


# open historical and forecast data for both cdd and hdd data
@functools.lru_cache(maxsize=2)
def load_historical(degree_days: str) -> xr.Dataset:
    return xr.open_zarr(
    f'gs://{BUCKET}/zarr/h-{degree_days}-{year_ic}-{month_ic}-01.zarr', 
        zarr_format=3, 
        consolidated=False, 
        decode_coords="all", 
        chunks={}
    ).compute()


@functools.lru_cache(maxsize=2)
def load_forecast(degree_days: str) -> xr.Dataset:
    return xr.open_zarr(
        f'gs://{BUCKET}/zarr/f-{degree_days}-{year_ic}-{month_ic}-01.zarr', 
        zarr_format=3, 
        consolidated=False, 
        decode_coords="all", 
        chunks={}
    ).compute()[['time', 'x', 'y', degree_days, 'cdd', '95%']]


# lazy load country boundary layer
@functools.lru_cache(maxsize=1)
def load_countries() -> gpd.GeoDataFrame:
    return gpd.read_parquet(f'gs://{BUCKET}/vector/countries.parquet')


# lazy load country boundary layer
@functools.lru_cache(maxsize=1)
def load_states() -> gpd.GeoDataFrame:
    return gpd.read_parquet(f'gs://{BUCKET}/vector/states.parquet')