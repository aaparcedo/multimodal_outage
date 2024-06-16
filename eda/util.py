from matplotlib.backends.backend_agg import FigureCanvasAgg
from gadm import GADMDownloader
from blackmarble.raster import bm_raster
from datetime import datetime, timedelta
import pickle
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import contextily as cx
import os
import io
from PIL import Image
import random
import time
import concurrent.futures
import xarray as xr

bearer = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImFhcGFyY2VkbyIsImV4cCI6MTcxOTc3MzAxNywiaWF0IjoxNzE0NTg5MDE3LCJpc3MiOiJFYXJ0aGRhdGEgTG9naW4ifQ.gok0oSUdK3Ak4p9QSnuD8b3wCRizrjG-LCJMvmglB122IqK6BHPhEbgu9fohRYi15935n69_tC1gYO0nI_oNZauRzgvI1b1bf0fFAlrnnL9rKI7Jtlh9ECkAKRchidDYzb-ilSeMWLVuSBrEPbf9a4-XanbsoYlkSzBqmsZauuaaqnKyH1YNh5yFwd1MYkfP9ampmmiy2UTwW0sRbFSW2MWEe3go0ZLB2_qFKhnIXvSbIpP90JgPFa__eOu0wtOrLyKA286iRTU5tS562dFIffiZHK4nStLzTS45dY4ba1exYdGV4QLlPeMkON3rO-I9M-vq5Wd-XuQhCvxy5t5Fjw"

ia_id_m_S_dates = ['2018_09_10.jpg', '2018_09_11.jpg', '2018_09_12.jpg', '2018_09_13.jpg', '2018_09_14.jpg', '2018_09_15.jpg', '2018_09_16.jpg', '2018_09_17.jpg', '2018_09_18.jpg',
         '2018_09_19.jpg', '2018_09_20.jpg', '2018_09_21.jpg', '2018_09_22.jpg', '2018_09_23.jpg', '2018_09_24.jpg', '2018_09_25.jpg', '2018_09_26.jpg', '2018_09_27.jpg',
         '2018_09_28.jpg', '2018_09_29.jpg', '2018_09_30.jpg', '2018_10_01.jpg', '2018_10_02.jpg', '2018_10_03.jpg', '2018_10_04.jpg', '2018_10_05.jpg', '2018_10_06.jpg', '2018_10_07.jpg', '2018_10_08.jpg', '2018_10_09.jpg', '2018_10_10.jpg', '2018_10_11.jpg', '2018_10_12.jpg', '2018_10_13.jpg', '2018_10_14.jpg', '2018_10_15.jpg', '2018_10_16.jpg', '2018_10_17.jpg', '2018_10_18.jpg', '2018_10_19.jpg', '2018_10_20.jpg', '2018_10_21.jpg', '2018_10_22.jpg', '2018_10_23.jpg', '2018_10_24.jpg', '2018_10_25.jpg', '2018_10_26.jpg', '2018_10_27.jpg', '2018_10_28.jpg', '2018_10_29.jpg', '2018_10_30.jpg', '2018_10_31.jpg', '2018_11_01.jpg', '2018_11_02.jpg', '2018_11_03.jpg', '2018_11_04.jpg', '2018_11_05.jpg', '2018_11_06.jpg', '2018_11_07.jpg', '2018_11_08.jpg', '2022_08_27.jpg', '2022_08_28.jpg', '2022_08_29.jpg', '2022_08_30.jpg', '2022_08_31.jpg', '2022_09_01.jpg', '2022_09_02.jpg', '2022_09_03.jpg', '2022_09_04.jpg', '2022_09_05.jpg', '2022_09_06.jpg', '2022_09_07.jpg', '2022_09_08.jpg', '2022_09_09.jpg', '2022_09_10.jpg', '2022_09_11.jpg', '2022_09_12.jpg', '2022_09_13.jpg', '2022_09_14.jpg', '2022_09_15.jpg', '2022_09_16.jpg', '2022_09_17.jpg', '2022_09_18.jpg', '2022_09_19.jpg', '2022_09_20.jpg', '2022_09_21.jpg', '2022_09_22.jpg', '2022_09_23.jpg', '2022_09_24.jpg', '2022_09_25.jpg', '2022_09_26.jpg', '2022_09_27.jpg', '2022_09_28.jpg', '2022_09_29.jpg', '2022_09_30.jpg', '2022_10_01.jpg', '2022_10_02.jpg', '2022_10_03.jpg', '2022_10_04.jpg', '2022_10_05.jpg', '2022_10_06.jpg', '2022_10_07.jpg', '2022_10_08.jpg', '2022_10_09.jpg', '2022_10_10.jpg', '2022_10_11.jpg', '2022_10_12.jpg', '2022_10_13.jpg', '2022_10_14.jpg', '2022_10_15.jpg', '2022_10_16.jpg', '2022_10_17.jpg', '2022_10_18.jpg', '2022_10_19.jpg', '2022_10_20.jpg', '2022_10_21.jpg', '2022_10_22.jpg', '2022_10_23.jpg', '2022_10_24.jpg', '2022_10_25.jpg', '2023_07_31.jpg', '2023_08_01.jpg', '2023_08_02.jpg', '2023_08_03.jpg', '2023_08_04.jpg', '2023_08_05.jpg', '2023_08_06.jpg', '2023_08_07.jpg', '2023_08_08.jpg', '2023_08_09.jpg', '2023_08_10.jpg', '2023_08_11.jpg', '2023_08_12.jpg', '2023_08_13.jpg', '2023_08_14.jpg', '2023_08_15.jpg', '2023_08_16.jpg', '2023_08_17.jpg', '2023_08_18.jpg', '2023_08_19.jpg', '2023_08_20.jpg', '2023_08_21.jpg', '2023_08_22.jpg', '2023_08_23.jpg', '2023_08_24.jpg', '2023_08_25.jpg', '2023_08_26.jpg', '2023_08_27.jpg', '2023_08_28.jpg', '2023_08_29.jpg', '2023_08_30.jpg', '2023_08_31.jpg', '2023_09_01.jpg', '2023_09_02.jpg', '2023_09_03.jpg', '2023_09_04.jpg', '2023_09_05.jpg', '2023_09_06.jpg', '2023_09_07.jpg', '2023_09_08.jpg', '2023_09_09.jpg', '2023_09_10.jpg', '2023_09_11.jpg', '2023_09_12.jpg', '2023_09_13.jpg', '2023_09_14.jpg', '2023_09_15.jpg', '2023_09_16.jpg', '2023_09_17.jpg', '2023_09_18.jpg', '2023_09_19.jpg', '2023_09_20.jpg', '2023_09_21.jpg', '2023_09_22.jpg', '2023_09_23.jpg', '2023_09_24.jpg', '2023_09_25.jpg', '2023_09_26.jpg', '2023_09_27.jpg', '2023_09_28.jpg']

ia_id_m_S_dates = [date.split('.')[0].replace('_', '-') for date in ia_id_m_S_dates]

def find_available_dates(base_dir=None, county_dir=None):
    """
    Itererate through all the the county subfolders and find shared dates.

    Parameters:
    - base_dir (str): directory of interest

    Returns:
    common_dates (list): pd.Timestamps of shared dates
    """

    # Initialize a set to hold the common dates/filenames across all county_dirs
    common_dates = None

    if base_dir == None:
        already_preprocessed_days = os.listdir(county_dir)
        day_list = [pd.Timestamp(day.split('.')[0].replace('_', '-'))
                    for day in already_preprocessed_days]
        return day_list

    else:
        print("base_dir: ", base_dir)
        county_dirs = os.listdir(base_dir)
        for county_idx, county_dir in enumerate(county_dirs):
            
            county_dir_path = os.path.join(base_dir, county_dir)
            days_avail = os.listdir(county_dir_path)
            # Set comprehension to extract dates
            dates_set = {day.split('.')[0] for day in days_avail}
            if common_dates is None:
                common_dates = dates_set
            else:
                common_dates = common_dates.intersection(dates_set)
        common_dates_list = list(common_dates)
        common_dates = [pd.Timestamp(date.replace('_', '-'))
                        for date in common_dates_list]
        return common_dates


def calculate_average_month_ntl(daily_image, month_composites):
    """
    Calculates the average monthly composite of the last three months from a given date.

    Parameters:
    - daily_image (xarray.Dataset): 
    - month_composites (xarray.Dataset): object containing necessary monthly composites

    Returns:
    average_month_ntl (np.ndarray): represents the last 3 month average ntl
    """

    day = daily_image.time.values
    np.set_printoptions(threshold=np.inf)

    # get dates of previous 3 months
    filtered_dates = pd.date_range(
        start=day - pd.DateOffset(months=4), end=day - pd.DateOffset(months=1), freq='MS')
    print(f"dates of previous three months: {filtered_dates}")
    monthly_ntl = []
    for month in filtered_dates:
        month_ntl = np.array(
            month_composites["NearNadir_Composite_Snow_Free"].sel(time=month))
        monthly_ntl.append(month_ntl)

    # calculate pixel-wise mean of last three months, ignoring NaN values
    average_month_ntl = np.nanmean(
        monthly_ntl, axis=0, where=~np.isnan(monthly_ntl))

    return average_month_ntl
    return daily_ntl, average_month_ntl

def load_month_composites(county_name):
    """
    Loads all the available monthly composites into memory.

    Parameters:
    - county_name (str): name of county, e.g., 'orange'

    Returns:
    - month_composites (xarray.Dataset): dataset of monthly composites
    """

    base_dir = "/groups/mli/multimodal_outage/data/black_marble/hq/monthly"
    county_dir = os.path.join(base_dir, county_name)
    file_path = os.path.join(county_dir, f"{county_name}.pickle")

    with open(file_path, 'rb') as file:
        month_composites = pickle.load(file)

    return month_composites


def download_monthly_composites(quality_flag):
    """
    Download all month files.
    Passes quality flag to Black Marble download function. 

    Parameters:
    - quality_flag (list): desired quality, e.g., [1, 255]

    Returns:
    - N/A
    """

    county_names = get_county_names_from_state_gdf()

    start_date = pd.Timestamp('2012-01-19')
    end_date = pd.Timestamp('2024-04-17')

    base_dataset_path = '/groups/mli/multimodal_outage/data/black_marble'
    # hq because quality_flag_rm=[2, 255]
    flag_dataset_path = os.path.join(base_dataset_path, "hq/monthly")
    os.makedirs(flag_dataset_path, exist_ok=True)

    for county in county_names:
        flag_county_dataset_path = os.path.join(flag_dataset_path, county)
        os.makedirs(flag_county_dataset_path, exist_ok=True)

        try:
            monthly_dataset = download_county_raster(
                county, quality_flag, 'MS', start_date, end_date)
        except Exception as e:
            print(f"Error: {e}", flush=True)
            return

        pickle_path = os.path.join(
            flag_county_dataset_path, f"{county}.pickle")
        with open(pickle_path, 'wb') as pickled_monthly_dataset:
            pickle.dump(monthly_dataset, pickled_monthly_dataset)


def get_gdf(county_name):
    gdf = GADMDownloader(version="4.0").get_shape_data_by_country_name(
        country_name="USA", ad_level=2)
    florida_gdf = gdf[gdf['NAME_1'] == 'Florida']
    county_gdf = florida_gdf[florida_gdf['NAME_2'].str.lower().str.replace(
        ' ', '_').str.replace('-', '_') == county_name]
    return county_gdf


def download_county_raster(county, quality_flag, freq, start_date, end_date=None):
    """
    Downloads the satellite image (xarray) for a specified county, day, and quality_flag.

    Parameters:
    - county (str): name of the county
    - start_date (Timestamp) :
    - end_date (Timestamp) :
    - quality_flag (list): desired quality flag, e.g., [2, 255]
    - freq (str): frequency of image (options 'D' or 'MS')

    Returns:
    raster (xarray): object of satellite image
    """

    county_gdf = get_gdf(county)
    dates = pd.date_range(
        start_date, end_date if end_date else start_date, freq=freq)

    if freq == 'D':
        product_id = "VNP46A2"
        #variable = "DNB_BRDF-Corrected_NTL"
        variable = "Gap-Filled-DNB_BRDF-Corrected_NTL"
    elif freq == 'MS':
        product_id = "VNP46A3"
        variable = "NearNadir_Composite_Snow_Free"
    else:
        print("Pick a valid time frequency, i.e., 'D' or 'M'")

    raster = bm_raster(county_gdf, product_id=product_id, date_range=dates,
                       bearer=bearer, variable=variable, quality_flag_rm=quality_flag)

    return raster


def download_missing_dates(dataset_county_path):
    """
    Find dates where Black Marble data is missing.
    Parameters:
    - dataset_county_path (str): the path to the dataset directory for the county

    Returns:
    - inverse_dates
    """
    # want data in this time frame
    required_dates = pd.date_range('2012-01-19', '2024-04-17', freq='D')

    available_data_file_names = os.listdir(dataset_county_path)
    available_data_file_names_formatted = pd.to_datetime([file_name.replace(
        '_', '-').replace('.pickle', '') for file_name in available_data_file_names])
    inverse_dates = required_dates[~required_dates.isin(
        available_data_file_names_formatted)]

    return inverse_dates

def big_download_fl_county_raster():
    """
    Parameters:



    Returns:
    - N/A
    """
    base_dataset_path = '/groups/mli/multimodal_outage/data/black_marble'
    flag_dataset_path = os.path.join(base_dataset_path, "hq/original_latest_hq_retrieval")
    county_names = ['lee', 'holmes', 'pinellas', 'seminole', 'jefferson', 'gadsden', 'desoto', 'hendry', 'suwannee', 'levy', 'franklin', 'hernando', 'flagler', 'orange', 'dixie']
    #county_names = ['liberty', 'putnam', 'taylor', 'okeechobee', 'jackson', 'charlotte', 'indian_river', 'pasco', 'walton', 'sumter', 'osceola', 'gilchrist']
    #county_names = ['wakulla', 'leon', 'monroe', 'saint_lucie', 'madison', 'alachua', 'clay', 'citrus', 'nassau', 'martin', 'broward', 'escambia', 'baker']
    #county_names = ['hillsborough', 'manatee', 'lafayette', 'gulf', 'glades', 'highlands', 'calhoun', 'duval', 'marion', 'santa_rosa', 'volusia', 'sarasota', 'washington']
    #county_names = ['hardee', 'bay', 'bradford', 'columbia', 'palm_beach', 'union', 'collier', 'hamilton', 'miami_dade', 'polk', 'okaloosa', 'saint_johns', 'lake', 'brevard']


    for county in county_names:
        flag_county_dataset_path = os.path.join(flag_dataset_path, county)
        os.makedirs(flag_county_dataset_path, exist_ok=True)

        gdf = get_gdf(county)
        dates = ia_id_m_S_dates
        try:
            #daily_dataset = bm_raster(gdf, product_id="VNP46A2", date_range=dates,
            #                          bearer=bearer, variable="Gap_Filled_DNB_BRDF-Corrected_NTL", quality_flag_rm=[255])
            daily_dataset = bm_raster(gdf, product_id="VNP46A2", date_range=dates, bearer=bearer, variable="Latest_High_Quality_Retrieval")
        except Exception as e:
            print(f"Error for {county}: {e}", flush=True)
            continue

        num_days_retrieved = daily_dataset.sizes['time']
        for day_idx in range(num_days_retrieved):
            raster = daily_dataset.isel(time=day_idx)
            day = str(raster.coords['time'])[-10:].replace('-', '_')
            pickle_path = os.path.join(
                flag_county_dataset_path, f"{day}.pickle")
            with open(pickle_path, 'wb') as pickled_raster:
                pickle.dump(raster, pickled_raster)
        print(f"Successfully downloaded all available dates for {county}")

    print(f'finished downloading all data')


def download_annual_composite(gdf, date):
    annual_composite = bm_raster(
        gdf, product_id="VNP46A4", date_range=date, bearer=bearer)
    return annual_composite


def get_county_names_from_state_gdf(state_gdf=None):
    """
    Parameters:
    state_gdf (geodataframe): gdf of state we want to find county names for

    Returns:
    normalized_county_names (list): array of county names for Florida (lowercase, no spaces or hyphens)
    """

    if state_gdf == None:
        gdf = GADMDownloader(version="4.0").get_shape_data_by_country_name(
            country_name="USA", ad_level=2)
        state_gdf = gdf[gdf['NAME_1'] == 'Florida']

    county_names = list(state_gdf['NAME_2'].values)
    normalized_county_names = [name.lower().replace(
        ' ', '_').replace('-', '_') for name in county_names]

    return normalized_county_names


def get_county_dims_from_county_raster(raster):
    """
    Parameters:
    raster (xarray): daily raster for a particular county

    Returns:
    xy_dims (tuple): size of raster as (x,y)
    """

    dims = []

    for dim, size in raster.dims.items():
        dims.append(size)

    xy_dims = tuple(dims[:2])

    return xy_dims


def get_dims_for_all_counties():
    """
    Parameters:
    N/A

    Returns:
    counties_dims (dict): name of county as key(s) and xy dimensions as value(s)
    """

    gdf = GADMDownloader(version="4.0").get_shape_data_by_country_name(
        country_name="USA", ad_level=2)
    florida_gdf = gdf[gdf['NAME_1'] == 'Florida']

    county_names = get_county_names_from_state_gdf(florida_gdf)

    counties_dims = {}

    for county_name in county_names:
        county_gdf = florida_gdf[florida_gdf['NAME_2'] == county_name]

        county_raster = bm_raster(county_gdf, product_id="VNP46A2",
                                  date_range="2024-01-01", bearer=bearer, quality_flag_rm=[255])

        dims = get_county_dims_from_county_raster(county_raster)

        counties_dims[county_name] = dims

    return counties_dims

