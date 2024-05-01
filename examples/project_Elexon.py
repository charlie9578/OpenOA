"""
This is the import script for Elexon data. Licenced as follows:

Contains BMRS data © Elexon Limited copyright and database right [2024].

https://www.elexon.co.uk/data/balancing-mechanism-reporting-agent/copyright-licence-bmrs-data/

Below is a description of the data imported and an overview of the steps taken to correct the raw
data for use in the OpenOA code.

1. Meter data
   - half-hourly generation data in energy units (kWh)

2. Reanalysis products
   - monthly MERRA2 and ERA5 reanalysis data
"""

from __future__ import annotations

import re
import json
import datetime
from pathlib import Path
from zipfile import ZipFile

import yaml
import numpy as np
import cdsapi
import pandas as pd
import xarray as xr
import requests
from tqdm import tqdm

import openoa.utils.downloader as downloader
from openoa.plant import PlantData
from openoa.logging import logging


logger = logging.getLogger()


def get_Elexon_plant_information():
    """
    Get Elexon plant information
    See: https://github.com/OSUKED/Power-Station-Dictionary
    """
    base_url = "https://github.com/OSUKED/Power-Station-Dictionary/raw/shiro/data"
    power_station_ids = pd.read_csv(f"{base_url}/dictionary/ids.csv")
    plant_locations = pd.read_csv(
        f"{base_url}/attribute_sources/plant-locations/plant-locations.csv"
    )
    fuel_types = pd.read_csv(f"{base_url}/attribute_sources/bmu-fuel-types/fuel_types.csv")

    power_station_ids["ngc_bmu_id"] = power_station_ids["ngc_bmu_id"].str.split(",")

    power_station_dict = power_station_ids.merge(plant_locations, on=["dictionary_id"])

    fuel_type_ids = fuel_types.merge(power_station_ids.explode("ngc_bmu_id"), on="ngc_bmu_id")

    fuel_type_mapping = fuel_type_ids[["dictionary_id", "fuel_type"]].drop_duplicates()

    power_stations = power_station_dict.merge(fuel_type_mapping, on="dictionary_id")

    wind_farms = power_stations[power_stations["fuel_type"] == "WIND"]

    return wind_farms


def get_Elexon_plant_generation(
    bmUnits=["AKGLW-2", "AKGLW-3"],
    timestamp_from="2024-02-01T00:00:00Z",
    timestamp_to="2024-03-01T00:00:00Z",
):
    """
    Build and make a request for Elexon plant generation

    API documentation: https://developer.data.elexon.co.uk/

    Licence: https://www.elexon.co.uk/data/balancing-mechanism-reporting-agent/copyright-licence-bmrs-data/
    """

    if not isinstance(bmUnits, list):
        bmUnits = [bmUnits]

    dfs = []
    for bmUnit in bmUnits:
        url = f"https://data.elexon.co.uk/bmrs/api/v1/datasets/B1610/stream?from={timestamp_from}&to={timestamp_to}&bmUnit={bmUnit}"

        result = requests.get(url)

        df_tmp = pd.DataFrame.from_records(result.json())

        if not df_tmp.empty:
            df_tmp["datetime"] = pd.to_datetime(df_tmp["halfHourEndTime"])

            dfs.append(df_tmp)

    if dfs:
        df = pd.DataFrame(pd.concat(dfs).groupby("datetime")["quantity"].sum())
    else:
        df = None

    return df


def download_era5_monthly(
    save_pathname: str | Path = "data",
    save_filename: str = "era5_monthly",
    start_date: str = "2000-01",
    end_date: str = None,
    cds_dataset: str = None,
    cds_request: dict = None,
) -> pd.DataFrame:
    """
    Get ERA5 data directly from the CDS service. This requires registration on the CDS service.
    See registration details at: https://cds.climate.copernicus.eu/api-how-to

    This function returns monthly ERA5 data from the "ERA5 monthly averaged data on single levels
    from 1959 to present" dataset. See further details regarding the dataset at:
    https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means.
    Only the 10m wind speed is downloaded here.

    As well as returning the data as a dataframe, the data is also saved as monthly NetCDF files and
    a csv file with the concatenated data. These are located in the "save_pathname" directory, with
    "save_filename" prefix. This allows future loading without download from the CDS service.

    Args:
        save_pathname(:obj:`str` | :obj:`Path`): The path where the downloaded reanalysis data will
            be saved.
        save_filename(:obj:`str`): The file name used to save the downloaded reanalysis data.
        start_date(:obj:`str`): The starting year and month that data is downloaded for. This
            should be provided as a string in the format "YYYY-MM". Defaults to "2000-01".
        end_date(:obj:`str`): The final year and month that data is downloaded for. This should be
            provided as a string in the format "YYYY-MM". Defaults to current year and most recent
            month with full data, accounting for the fact that the ERA5 monthly dataset is released
            around the the 6th of the month.

    Returns:
        df(:obj:`dataframe`): A dataframe containing time series of the requested reanalysis
            variables:
            1. windspeed_ms: the wind speed in m/s at 10m height.

    Raises:
        ValueError: If the start_date is greater than the end_date.
        Exception: If unable to connect to the cdsapi client.
    """

    logger.info("Please note access to ERA5 data requires registration")
    logger.info("Please see: https://cds.climate.copernicus.eu/api-how-to")

    # set up cds-api client
    try:
        c = cdsapi.Client()
    except Exception as e:
        logger.error("Failed to make connection to cds")
        logger.error("Please see https://cds.climate.copernicus.eu/api-how-to for help")
        logger.error(e)
        raise

    # create save_pathname if it does not exist
    save_pathname = Path(save_pathname).resolve()
    if not save_pathname.exists():
        save_pathname.mkdir()

    # get the current date minus 37 days to find the most recent full month of data
    now = datetime.datetime.now() - datetime.timedelta(days=37)

    # assign end_year to current year if not provided by the user
    if end_date is None:
        end_date = f"{now.year}-{now.month:02}"

    # convert dates to datetime objects
    start_date = datetime.datetime.strptime(start_date, "%Y-%m")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m")

    # check that the start and end dates are the right way around
    if start_date > end_date:
        logger.error("The start_date should be less than or equal to the end_date")
        logger.error(f"start_date = {start_date.date()}, end_date = {end_date.date()}")
        raise ValueError("The start_date should be less than or equal to the end_date")

    # list all dates that will be downloaded
    dates = pd.date_range(start=start_date, end=end_date, freq="MS", inclusive="both")

    # See: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=form
    # for formulating other requests from cds

    if cds_dataset is None:
        cds_dataset = "reanalysis-era5-single-levels-monthly-means"

    if cds_request is None:
        cds_request = {
            "product_type": "monthly_averaged_reanalysis",
            "format": "netcdf",
            "variable": [
                "10m_wind_speed",
            ],
            "year": None,
            "month": None,
            "time": ["00:00"],
        }

    # download the data
    for date in dates:
        outfile = save_pathname / f"{save_filename}_{date.year}{date.month:02}.nc"

        if not outfile.is_file():
            logger.info(f"Downloading ERA5: {outfile}")

            try:
                cds_request.update({"year": date.year, "month": date.month})
                c.retrieve(cds_dataset, cds_request, outfile)

            except Exception as e:
                logger.error(f"Failed to download ERA5: {outfile}")
                logger.error(e)


def download_era5_hourly(
    save_pathname: str | Path = "data/era5_hourly/gb",
    save_filename: str = "era5_gb_hourly",
    start_date: str = "2000-01",
    end_date: str = None,
    cds_dataset: str = None,
    cds_request: dict = None,
):
    cds_dataset = "reanalysis-era5-single-levels"

    cds_request = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": [
            "100m_u_component_of_wind",
            "100m_v_component_of_wind",
        ],
        "year": "2021",
        "month": "06",
        "day": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
        ],
        "time": [
            "00:00",
            "01:00",
            "02:00",
            "03:00",
            "04:00",
            "05:00",
            "06:00",
            "07:00",
            "08:00",
            "09:00",
            "10:00",
            "11:00",
            "12:00",
            "13:00",
            "14:00",
            "15:00",
            "16:00",
            "17:00",
            "18:00",
            "19:00",
            "20:00",
            "21:00",
            "22:00",
            "23:00",
        ],
        "area": [
            63,
            -12,
            48,
            5,
        ],
    }

    download_era5_monthly(
        save_pathname=save_pathname,
        save_filename=save_filename,
        start_date=start_date,
        end_date=end_date,
        cds_dataset=cds_dataset,
        cds_request=cds_request,
    )


def get_nearest_slice(sel, ds_nc, num_points_to_include=1):
    # Get the longitude and latitude values of the nearest point
    nearest_lon = sel["longitude"].values
    nearest_lat = sel["latitude"].values

    # Find the indices of the nearest coordinates in the original dataset
    nearest_indices = np.where(
        (ds_nc["longitude"] == nearest_lon) & (ds_nc["latitude"] == nearest_lat)
    )

    # nearest_indices is a tuple, so you can extract the indices
    lon_index, lat_index = nearest_indices[0][0], nearest_indices[1][0]

    # Calculate the start and end indices for the slice
    lon_start = max(0, lon_index - num_points_to_include)
    lon_end = min(
        ds_nc.sizes["longitude"], lon_index + num_points_to_include + 1
    )  # Add 1 to include the endpoint

    lat_start = max(0, lat_index - num_points_to_include)
    lat_end = min(
        ds_nc.sizes["latitude"], lat_index + num_points_to_include + 1
    )  # Add 1 to include the endpoint

    # Extract the slice from the original dataset
    nearest_slice = ds_nc.isel(
        longitude=slice(lon_start, lon_end), latitude=slice(lat_start, lat_end)
    )

    return nearest_slice


def get_era5_monthly(
    lat: float,
    lon: float,
    save_filename: str,
    save_pathname: str | Path = "data/assets",
    data_pathname: str | Path = "data",
    data_filename: str = "era5_monthly",
    start_date: str = "2000-01",
    end_date: str = None,
) -> pd.DataFrame:
    """
        Args:
        lat(:obj:`float`): Latitude in WGS 84 spatial reference system (decimal degrees).
        lon(:obj:`float`): Longitude in WGS 84 spatial reference system (decimal degrees).
        save_pathname(:obj:`str` | :obj:`Path`): The path where the downloaded reanalysis data will
            be saved.
        save_filename(:obj:`str`): The file name used to save the downloaded reanalysis data.
        start_date(:obj:`str`): The starting year and month that data is downloaded for. This
            should be provided as a string in the format "YYYY-MM". Defaults to "2000-01".
        end_date(:obj:`str`): The final year and month that data is downloaded for. This should be
            provided as a string in the format "YYYY-MM". Defaults to current year and most recent
            month with full data, accounting for the fact that the ERA5 monthly dataset is released
            around the the 6th of the month.

    Returns:
        df(:obj:`dataframe`): A dataframe containing time series of the requested reanalysis
            variables:
            1. windspeed_ms: the wind speed in m/s at 10m height.

    Raises:
        ValueError: If the start_date is greater than the end_date.
        Exception: If unable to connect to the cdsapi client.
    """

    # download the monthly era5 data
    download_era5_monthly(
        save_pathname=data_pathname,
        save_filename=data_filename,
        start_date=start_date,
        end_date=end_date,
    )

    # get the saved data
    ds_nc = xr.open_mfdataset(f"{data_pathname}/{data_filename}*.nc")

    # rename variables to conform with OpenOA
    ds_nc = ds_nc.rename_vars({"si10": "windspeed_ms"})

    # select the central node only for now
    if "expver" in ds_nc.dims:
        sel = ds_nc.sel(expver=1, latitude=lat, longitude=lon, method="nearest")
    else:
        sel = ds_nc.sel(latitude=lat, longitude=lon, method="nearest")

    # now take the surrounding nearest nodes as well
    nearest_slice = get_nearest_slice(sel, ds_nc, num_points_to_include=1)

    # convert to a pandas dataframe
    df = nearest_slice.to_dataframe().unstack(["latitude", "longitude"])["windspeed_ms"]

    # rename columns based on their coordinates
    df.columns = [str(item) for item in df.columns.values]

    # rename the index to match other datasets
    df.index.name = "datetime"

    # drop any empty rows
    df = df.dropna()

    # crop time series to only the selected time period
    df = df.loc[start_date:end_date]

    # save to csv for easy loading as required
    # create save_pathname if it does not exist
    save_pathname = Path(save_pathname).resolve()
    if not save_pathname.exists():
        save_pathname.mkdir()

    df.to_csv(save_pathname / f"{save_filename}.csv", index=True)

    return df


def get_era5_hourly(
    lat: float,
    lon: float,
    save_filename: str,
    save_pathname: str | Path = "data/assets",
    data_pathname: str | Path = "data/era5_hourly/gb",
    data_filename: str = "era5_gb_hourly",
    start_date: str = "2000-01",
    end_date: str = None,
) -> pd.DataFrame:
    """
        Args:
        lat(:obj:`float`): Latitude in WGS 84 spatial reference system (decimal degrees).
        lon(:obj:`float`): Longitude in WGS 84 spatial reference system (decimal degrees).
        save_pathname(:obj:`str` | :obj:`Path`): The path where the downloaded reanalysis data will
            be saved.
        save_filename(:obj:`str`): The file name used to save the downloaded reanalysis data.
        start_date(:obj:`str`): The starting year and month that data is downloaded for. This
            should be provided as a string in the format "YYYY-MM". Defaults to "2000-01".
        end_date(:obj:`str`): The final year and month that data is downloaded for. This should be
            provided as a string in the format "YYYY-MM". Defaults to current year and most recent
            month with full data, accounting for the fact that the ERA5 monthly dataset is released
            around the the 6th of the month.

    Returns:
        df(:obj:`dataframe`): A dataframe containing time series of the requested reanalysis
            variables:
            1. windspeed_ms: the wind speed in m/s at 10m height.

    Raises:
        ValueError: If the start_date is greater than the end_date.
        Exception: If unable to connect to the cdsapi client.
    """

    # download the hourly era5 data
    download_era5_hourly(
        save_pathname=data_pathname,
        save_filename=data_filename,
        start_date=start_date,
        end_date=end_date,
    )

    # get the saved data file names
    # files = list(data_pathname.glob(f"{data_filename}*.nc"))

    # get the saved data
    ds_nc = xr.open_mfdataset(f"{data_pathname}/{data_filename}*.nc")

    # rename variables to conform with OpenOA
    ds_nc["windspeed_ms"] = (ds_nc["u100"] ** 2 + ds_nc["v100"] ** 2) ** 0.5

    # select the central node only for now
    if "expver" in ds_nc.dims:
        sel = ds_nc.sel(expver=1, latitude=lat, longitude=lon, method="nearest")
    else:
        sel = ds_nc.sel(latitude=lat, longitude=lon, method="nearest")

    # now take the surrounding nearest nodes as well
    nearest_slice = get_nearest_slice(sel, ds_nc, num_points_to_include=1)

    # convert to a pandas dataframe
    df = nearest_slice.to_dataframe().unstack(["latitude", "longitude"])["windspeed_ms"]

    # rename columns based on their coordinates
    df.columns = [str(item) for item in df.columns.values]

    # rename the index to match other datasets
    df.index.name = "datetime"

    # drop any empty rows
    df = df.dropna()

    # crop time series to only the selected time period
    df = df.loc[start_date:end_date]

    # save to csv for easy loading as required
    # create save_pathname if it does not exist
    save_pathname = Path(save_pathname).resolve()
    if not save_pathname.exists():
        save_pathname.mkdir()

    df.to_csv(save_pathname / f"{save_filename}.csv", index=True)

    return df


def download_file(url: str, outfile: str | Path) -> None:
    """
    Download a file from the web, based on its url, and save to the outfile.

    Args:
        url(:obj:`str`): Url of data to download.
        outfile(:obj:`str` | :obj:`Path`): File path to which the download is saved.

    Raises:
        HTTPError: If unable to access url.
        Exception: If the request failed for another reason.
    """

    outfile = Path(outfile).resolve()
    result = requests.get(url, stream=True)

    try:
        result.raise_for_status()
        try:
            with outfile.open("wb") as f:
                for chunk in tqdm(
                    result.iter_content(chunk_size=1024 * 1024), desc="MB downloaded"
                ):
                    if chunk:
                        f.write(chunk)

            logger.info(f"Contents of {url} written to {outfile}")

        except Exception as e:
            logger.error(e)
            raise

    except requests.exceptions.HTTPError as eh:
        logger.error(eh)
        raise

    except Exception as e:
        logger.error(e)
        raise


def download_merra2_monthly(
    save_pathname: str | Path = "data",
    save_filename: str = "era5_monthly",
    start_date: str = "2000-01",
    end_date: str = None,
) -> pd.DataFrame:
    """
    Get MERRA2 data directly from the NASA GES DISC service, which requires registration on the
    GES DISC service. See: https://disc.gsfc.nasa.gov/data-access#python-requests.

    This function returns monthly MERRA2 data from the "M2IMNXLFO" dataset. See further details
    regarding the dataset at: https://disc.gsfc.nasa.gov/datasets/M2IMNXLFO_5.12.4/summary.
    Only surface wind speed, temperature and surface pressure are downloaded here.

    As well as returning the data as a dataframe, the data is also saved as monthly NetCDF files
    and a csv file with the concatenated data. These are located in the "save_pathname" directory,
    with "save_filename" prefix. This allows future loading without download from the CDS service.

    Args:
        save_pathname(:obj:`str` | :obj:`Path`): The path where the downloaded reanalysis data will
            be saved.
        save_filename(:obj:`str`): The file name used to save the downloaded reanalysis data.
        start_date(:obj:`str`): The starting year and month that data is downloaded for. This
            should be provided as a string in the format "YYYY-MM". Defaults to "2000-01".
        end_date(:obj:`str`): The final year and month that data is downloaded for. This should be
            provided as a string in the format "YYYY-MM". Defaults to current year and most recent
            month.

    Returns:
        df(:obj:`dataframe`): A dataframe containing time series of the requested reanalysis
            variables:
            1. windspeed_ms: the surface wind speed in m/s.

    Raises:
        ValueError: If the start_year is greater than the end_year.
    """

    # base url containing the monthly data set M2IMNXLFO
    base_url = r"https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2_MONTHLY/M2IMNXLFO.5.12.4/"

    # create save_pathname if it does not exist
    save_pathname = Path(save_pathname).resolve()
    if not save_pathname.exists():
        save_pathname.mkdir()

    # get the current date minus 37 days to find the most recent full month of data
    now = datetime.datetime.now() - datetime.timedelta(days=37)

    # assign end_year to current year if not provided by the user
    if end_date is None:
        end_date = f"{now.year}-{now.month:02}"

    # convert dates to datetime objects
    start_date = datetime.datetime.strptime(start_date, "%Y-%m")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m")

    # check that the start and end dates are the right way around
    if start_date > end_date:
        logger.error("The start_date should be less than or equal to the end_date")
        logger.error(f"start_date = {start_date.date()}, end_date = {end_date.date()}")
        raise ValueError("The start_date should be less than or equal to the end_date")

    # list all dates that will be downloaded
    dates = pd.date_range(start=start_date, end=end_date, freq="MS", inclusive="both")

    # check what years need downloading
    years = []
    for date in dates:
        outfile = save_pathname / f"{save_filename}_{date.year}{date.month:02}.nc"
        if not outfile.is_file():
            years.append(date.year)

    # list all years that will be downloaded
    years = list(set(years))

    # download the data
    for year in years:
        # get the file names from the GES DISC site for the year
        result = requests.get(f"{base_url}{year}")

        files = re.findall(r"(>MERRA2_\S+.nc4)", result.text)
        files = list(dict.fromkeys(files))
        files = [x[1:] for x in files]

        # download each of the files and save them
        for f in files:
            outfile = save_pathname / f"{save_filename}_{f.split('.')[-2]}.nc"

            if not outfile.is_file():
                # download each file
                url = f"{base_url}{year}/{f}" + r".nc4?SPEEDLML,time,lat,lon"
                download_file(url, outfile)


def get_merra2_monthly(
    lat: float,
    lon: float,
    save_filename: str,
    save_pathname: str | Path = "data/assets",
    data_pathname: str | Path = "data",
    data_filename: str = "merra2_monthly",
    start_date: str = "2000-01",
    end_date: str = None,
) -> pd.DataFrame:
    """
    Get MERRA2 data directly from the NASA GES DISC service, which requires registration on the
    GES DISC service. See: https://disc.gsfc.nasa.gov/data-access#python-requests.

    This function returns monthly MERRA2 data from the "M2IMNXLFO" dataset. See further details
    regarding the dataset at: https://disc.gsfc.nasa.gov/datasets/M2IMNXLFO_5.12.4/summary.
    Only surface wind speed, temperature and surface pressure are downloaded here.

    As well as returning the data as a dataframe, the data is also saved as monthly NetCDF files
    and a csv file with the concatenated data. These are located in the "save_pathname" directory,
    with "save_filename" prefix. This allows future loading without download from the CDS service.

    Args:
        lat(:obj:`float`): Latitude in WGS 84 spatial reference system (decimal degrees).
        lon(:obj:`float`): Longitude in WGS 84 spatial reference system (decimal degrees).
        save_pathname(:obj:`str` | :obj:`Path`): The path where the downloaded reanalysis data will
            be saved.
        save_filename(:obj:`str`): The file name used to save the downloaded reanalysis data.
        start_date(:obj:`str`): The starting year and month that data is downloaded for. This
            should be provided as a string in the format "YYYY-MM". Defaults to "2000-01".
        end_date(:obj:`str`): The final year and month that data is downloaded for. This should be
            provided as a string in the format "YYYY-MM". Defaults to current year and most recent
            month.

    Returns:
        df(:obj:`dataframe`): A dataframe containing time series of the requested reanalysis
            variables:
            1. windspeed_ms: the surface wind speed in m/s.

    Raises:
        ValueError: If the start_year is greater than the end_year.
    """

    logger.info("Please note access to MERRA2 data requires registration")
    logger.info("Please see: https://disc.gsfc.nasa.gov/data-access#python-requests")

    download_merra2_monthly(
        save_pathname=data_pathname,
        save_filename=data_filename,
        start_date=start_date,
        end_date=end_date,
    )

    # get the saved data
    ds_nc = xr.open_mfdataset(f"{data_pathname}/{data_filename}*.nc")

    # rename variables to conform with OpenOA
    ds_nc = ds_nc.rename_vars({"SPEEDLML": "windspeed_ms"})

    # rename coords to match across code
    ds_nc = ds_nc.rename({"lon": "longitude", "lat": "latitude"})

    # wrap -180..179 to 0..359
    ds_nc.coords["longitude"] = np.mod(ds_nc["longitude"], 360)

    # sort the data
    ds_nc = ds_nc.reindex({"longitude": np.sort(ds_nc["longitude"])})

    # select the central node only for now
    sel = ds_nc.sel(latitude=lat, longitude=lon, method="nearest")

    # now take the surrounding nearest nodes as well
    nearest_slice = get_nearest_slice(sel, ds_nc, num_points_to_include=1)

    # convert to a pandas dataframe
    df = nearest_slice.to_dataframe().unstack(["latitude", "longitude"])["windspeed_ms"]

    # rename columns based on their coordinates
    df.columns = [str(item) for item in df.columns.values]

    # rename the index to match other datasets
    df.index.name = "datetime"

    # drop any empty rows
    df = df.dropna()

    # crop time series to only the selected time period
    df = df.loc[start_date:end_date]

    # save to csv for easy loading as required
    # create save_pathname if it does not exist
    save_pathname = Path(save_pathname).resolve()
    if not save_pathname.exists():
        save_pathname.mkdir()

    df.to_csv(save_pathname / f"{save_filename}.csv", index=True)

    return df


def prepare(
    dictionary_id: float = 10252, return_value: str = "plantdata"
) -> PlantData | pd.DataFrame:
    """
    Do all loading and preparation of the data for this plant.

    Args:
        asset(:obj:`str`): Asset name, currently either "kelmarsh" or "penmanshiel". Defaults
            to "kelmarsh".
        return_value(:obj:`str`):  One of "plantdata" or "dataframes" with the below behavior.
            Defaults to "plantdata".

            - "plantdata" will return a fully constructed PlantData object.
            - "dataframes" will return a list of dataframes instead.

    Returns:
        Either PlantData object or Dataframes dependent upon return_value.
    """

    #################
    # ELEXON PLANTS #
    #################

    # Get Elexon plant information for wind assets
    plant_info_file = Path("data/plant_list_Elexon.csv").resolve()
    if not plant_info_file.is_file():
        df_plant = get_Elexon_plant_information()
        df_plant.to_csv(plant_info_file)
    else:
        df_plant = pd.read_csv(plant_info_file)

    df_plant["country"] = "UK"
    df_plant["dataSource"] = "Elexon"

    df_plant = df_plant.rename(
        columns={
            "name": "plantName",
            "fuel_type": "technology",
            "latitude": "Latitude",
            "longitude": "Longitude",
        }
    )

    df_plant = df_plant.sort_values("plantName").set_index("dictionary_id")

    ##############
    # ASSET DATA #
    ##############

    logger.info("Reading in the asset data")

    asset_df = df_plant.loc[dictionary_id].copy()

    asset_df["plantName"] = asset_df["plantName"].replace("/", "")

    # Remove any empty lines
    asset_df = asset_df.dropna(how="all")

    asset_df["asset_id"] = dictionary_id

    asset_df["ngc_bmu_id"] = (
        asset_df["ngc_bmu_id"].strip("[]").replace(" ", "").replace("'", "").split(",")
    )

    # Assign type to turbine for all assets
    asset_df["type"] = "turbine"

    # Set the path to store and access all the data
    path = f"data/elexon/{dictionary_id}"

    # Create the folder if it doesn't exist
    Path(path).mkdir(parents=True, exist_ok=True)

    # get generation data
    df_generation = (
        get_Elexon_plant_generation(
            bmUnits=asset_df["ngc_bmu_id"],
            timestamp_from="2000-01-01T00:00:00Z",
            timestamp_to="2024-04-30T23:59:59Z",
        )
        * 1000
    )

    ##############
    # SCADA DATA #
    ##############

    logger.info("Reading in the SCADA data")
    scada_df = df_generation.copy()
    scada_df = scada_df.reset_index()
    scada_df["asset_id"] = dictionary_id

    ##############
    # METER DATA #
    ##############

    logger.info("Reading in the meter data")

    meter_df = df_generation.copy()
    meter_df = meter_df.reset_index()

    #####################################
    # Availability and Curtailment Data #
    #####################################

    logger.info("Reading in the curtailment and availability losses data")
    curtail_df = df_generation.copy()
    curtail_df = curtail_df.reset_index()
    curtail_df["IAVL_ExtPwrDnWh"] = 0
    curtail_df["IAVL_DnWh"] = 0

    ###################
    # REANALYSIS DATA #
    ###################

    logger.info("Reading in the reanalysis data")

    # reanalysis datasets are held in a dictionary
    reanalysis_dict = dict()

    # ERA5 monthly 10m from CDS
    # get era5 data for the selected plant
    reanalysis_era5_monthly_df = get_era5_monthly(
        lat=asset_df["Latitude"],
        lon=asset_df["Longitude"] % 360,
        save_pathname=path,
        save_filename=f"era5_monthly_plant_{asset_df['plantName']}",
        data_pathname="data/era5_monthly",
        data_filename="era5_monthly",
        start_date="2000-01",
        end_date="2024-03",
    )

    reanalysis_era5_monthly_df = reanalysis_era5_monthly_df.reset_index()
    reanalysis_era5_monthly_df["WMETR_AirDen"] = 1
    reanalysis_dict.update(dict(era5_monthly=reanalysis_era5_monthly_df))

    # MERRA2 monthly 10m from GES DISC
    reanalysis_merra2_monthly_df = get_merra2_monthly(
        lat=asset_df["Latitude"],
        lon=asset_df["Longitude"] % 360,
        save_pathname=path,
        save_filename=f"merra2_monthly_plant_{asset_df['plantName']}",
        data_pathname="data/merra2_monthly",
        data_filename="merra2_monthly",
        start_date="2000-01",
        end_date="2024-03",
    )

    reanalysis_merra2_monthly_df = reanalysis_merra2_monthly_df.reset_index()
    reanalysis_merra2_monthly_df["WMETR_AirDen"] = 1
    reanalysis_dict.update(dict(merra2_monthly=reanalysis_merra2_monthly_df))

    ###################
    # PLANT DATA #
    ###################

    # Create plant_meta.json
    asset_json = {
        "asset": {
            "elevation": "Elevation (m)",
            "hub_height": "Hub Height (m)",
            "asset_id": "asset_id",
            "latitude": "Latitude",
            "longitude": "Longitude",
            "rated_power": "Rated power (kW)",
            "rotor_diameter": "Rotor Diameter (m)",
        },
        "curtail": {
            "IAVL_DnWh": "IAVL_DnWh",
            "IAVL_ExtPwrDnWh": "IAVL_ExtPwrDnWh",
            "frequency": "10min",
            "time": "datetime",
        },
        "latitude": str(asset_df["Latitude"]),
        "longitude": str(asset_df["Longitude"]),
        # "capacity": str(asset_df["Rated power (kW)"].sum() / 1000),
        "meter": {"MMTR_SupWh": "quantity", "time": "datetime"},
        "reanalysis": {
            "era5_monthly": {
                "WMETR_EnvPres": "surf_pres_Pa",
                "WMETR_EnvTmp": "temperature_K",
                "WMETR_HorWdSpd": reanalysis_era5_monthly_df.columns[5],
                "frequency": "1MS",
                "time": "datetime",
            },
            "merra2_monthly": {
                "WMETR_EnvPres": "surf_pres_Pa",
                "WMETR_EnvTmp": "temperature_K",
                "WMETR_HorWdSpd": reanalysis_merra2_monthly_df.columns[5],
                "frequency": "1MS",
                "time": "datetime",
            },
        },
        "scada": {
            "WMET_EnvTmp": "Nacelle ambient temperature (°C)",
            "WMET_HorWdDir": "Wind direction (°)",
            "WNAC_Dir": "Nacelle position (°)",
            "WMET_HorWdDirRel": "Yaw error",  # wind direction relative to nacelle orientation, degrees
            "WMET_HorWdSpd": "Wind speed (m/s)",
            "WROT_BlPthAngVal": "Blade angle (pitch position) A (°)",
            "WROT_BlPthAngVal_MIN": r"Blade angle (pitch position) A, Min (°)",
            "WROT_BlPthAngVal_MAX": r"Blade angle (pitch position) A, Max (°)",
            "asset_id": "asset_id",
            "WTUR_W": "quantity",
            "WTUR_W_MIN": "Power, Minimum (kW)",
            "WTUR_W_MAX": "Power, Maximum (kW)",
            "frequency": "10min",
            "time": "datetime",
        },
    }

    with open(f"{path}/plant_meta.json", "w") as outfile:
        json.dump(asset_json, outfile, indent=2)

    with open(f"{path}/plant_meta.yml", "w") as outfile:
        yaml.dump(asset_json, outfile, default_flow_style=False)

    mappings = yaml.safe_load(Path(f"{path}/plant_meta.yml").read_text())

    asset_df = asset_df.to_frame().T

    scada_df = scada_df.rename(columns={v: k for k, v in mappings["scada"].items()})
    meter_df = meter_df.rename(columns={v: k for k, v in mappings["meter"].items()})
    curtail_df = curtail_df.rename(columns={v: k for k, v in mappings["curtail"].items()})
    asset_df = asset_df.rename(columns={v: k for k, v in mappings["asset"].items()})
    mappings_remaining = {key: mappings[key] for key in ["reanalysis", "latitude", "longitude"]}

    # print(asset_df)

    # Return the appropriate data format
    if return_value == "dataframes":
        return (
            scada_df,
            meter_df,
            curtail_df,
            asset_df,
            reanalysis_dict,
        )
    elif return_value == "plantdata":
        # Build and return PlantData
        plantdata = PlantData(
            analysis_type=None,  # Choosing a random type that doesn't fail validation
            metadata=mappings_remaining,
            scada=scada_df,
            meter=meter_df,
            curtail=curtail_df,
            asset=asset_df,
            reanalysis=reanalysis_dict,
        )

        return plantdata


if __name__ == "__main__":
    prepare()
