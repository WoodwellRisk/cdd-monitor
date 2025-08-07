[mit license]: https://badgen.net/badge/license/MIT/blue
![MIT License][]

# Woodwell Risk CDD Monitor and Forecast
This site displays  an **estimate** of historical cooling degree days (CDD) along with an experimental 6-month forecast. Note that the CDD metric is normally calculated with daily data and aggregated at the monthly or yearly level, whereas we are attempting to estimate monthly CDD from monthly temperature data.

## Python environment
To recreate the conda environment we use in this repository, please run:
```
conda env create -f environment.yml
```
And to activate the environment:
```
conda activate shiny
```

## Run the app locally
To start a local server and see the app, please run the following command from within the `app/` directory:
```python
shiny run --reload cdd.py
```

## Data sources and processing steps
### Vector data
National and state outlines were downloaded from [Natural Earth](https://www.naturalearthdata.com/). Crop masks were created using a modified version of the [SPAM 2020](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SWPENT) combined rainfed- and irrigated production data for specific crops.

### Raster data
The temperature data used to create the water CDD estimates comes from [ERA5 monthly averaged data](https://cds.climate.copernicus.eu/stac-browser/collections/reanalysis-era5-single-levels-monthly-means?.language=en) and were downloaded using the Copernicus Climate Data Store (CDS) Application Program Interface (API), or [CDS API](https://cds.climate.copernicus.eu/how-to-api).

For back-end data analysis/transformation of `NetCDF` and `TIF` files, we used Python.
