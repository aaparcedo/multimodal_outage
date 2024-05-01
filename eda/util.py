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


bearer="eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImFhcGFyY2VkbyIsImV4cCI6MTcxOTc3MzAxNywiaWF0IjoxNzE0NTg5MDE3LCJpc3MiOiJFYXJ0aGRhdGEgTG9naW4ifQ.gok0oSUdK3Ak4p9QSnuD8b3wCRizrjG-LCJMvmglB122IqK6BHPhEbgu9fohRYi15935n69_tC1gYO0nI_oNZauRzgvI1b1bf0fFAlrnnL9rKI7Jtlh9ECkAKRchidDYzb-ilSeMWLVuSBrEPbf9a4-XanbsoYlkSzBqmsZauuaaqnKyH1YNh5yFwd1MYkfP9ampmmiy2UTwW0sRbFSW2MWEe3go0ZLB2_qFKhnIXvSbIpP90JgPFa__eOu0wtOrLyKA286iRTU5tS562dFIffiZHK4nStLzTS45dY4ba1exYdGV4QLlPeMkON3rO-I9M-vq5Wd-XuQhCvxy5t5Fjw"

# TODO: modify this function to pass in a ratio rater to save_satellite_image_square
# or (check other function comment)
def preprocess_raster_images():

  base_dir = '/groups/mli/multimodal_outage/data/black_marble/hq'
  ntl_dir = os.path.join(base_dir, 'ntl')
  percent_normal_dir = os.path.join(base_dir, 'percent_normal') 
 
  dates = pd.date_range('2012-01-19', '2024-04-17', freq='D')

  for county in county_names:
    county_dir = os.path.join(base_dir, county)

    # these two are for the save function
    save_file_path_ntl = os.path.join(ntl_dir, county)
    save_file_path_percent_normal = os.path.join(percent_normal_dir, county)
    os.makedirs(save_file_path_ntl)
    os.makedirs(save_file_path_percent_normal)

    for day in dates:

      # day must be in str format, e.g., '2012_01_19'
      file_path = os.path.join(county_dir, f'{day.strftime("%Y_%m_%d")}.pickle')

      with open(file_path, 'rb') as file:
        daily_image = pickle.load(file)

      daily_ntl = np.array(daily_image["DNB_BRDF-Corrected_NTL"])
 
      # TODO:  find the proper year by indexing with the day

      # e.g.: if day is '2017-01-01' then the composite should be for 2016
      composite_image = composites[day]
      composite_ntl = np.array(composite_image["NearNadir_Composite_Snow_Free"])
 
      # make a copy of the daily image to modify
      percent_normal_image = daily_image.copy()

      # TODO: add a condition that makes sure that dims of daily_ntl and composite_ntl match 
      percent_normal_np = 100 * (daily_ntl / (composite_ntl + 1e-10))
      
      # this value doesnt actually represent "DNB_BRDF-Corrected_NTL", it represents the percent of normal ratio
      percent_normal_image['DNB_BRDF-Corrected_NTL'] = (['y', 'x'], percent_normal_np[0])

      # send to be resized to a specified tbd size and saved to special folder
      save_satellite_image_square(daily_image, save_file_path_ntl) 
      save_satellite_image_square(percent_normal_image, save_file_path_percent_normal)


def save_satellite_image_square(raster_image, save_path):

  day = str(raster_image.time)[-10:].replace('-', '_')
  save_path = os.path.join(save_path, f'{day}.jpg')

  dpi = 300
  fig_width = (raster_image.sizes['x'] * 20) / dpi
  fig_height = (raster_image.sizes['y'] * 20)  / dpi

  fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

  # TODO: verify this shading parameter is the same as built-in plot method in BlackMarblePy
  if 'ntl' in save_path:
    cmap = "cividis"
  else:
    cmap = "RdYlGn"
    
  c = ax.pcolormesh(raster_image.x, raster_image.y, raster_image["DNB_BRDF-Corrected_NTL"], shading='auto', cmap=cmap)

  cx.add_basemap(ax, crs="EPSG:4326")
  plt.gca().set(xticks=[], yticks=[])
  plt.tight_layout(pad=0)
  plt.close(fig)
  
  buf = io.BytesIO()
  fig.savefig(buf, format='jpg')
  buf.seek(0)
  image = Image.open(buf)
  resized_image = image.resize((128, 128), Image.LANCZOS)
  resized_image.save(save_path)
  buf.close()


def get_gdf(county_name):
  gdf = GADMDownloader(version="4.0").get_shape_data_by_country_name(country_name="USA", ad_level=2)
  florida_gdf = gdf[gdf['NAME_1'] == 'Florida'] 
  county_gdf  = florida_gdf[florida_gdf['NAME_2'].str.lower().str.replace(' ', '_').str.replace('-', '_') == county_name]   

  return county_gdf


def download_county_raster(county, quality_flag, start_date, end_date=None):
  """
  Downloads the satellite image (xarray) for a specified county, day, and quality_flag.

  Parameters:
  - county (str): name of the county
  - start_date (Timestamp) :
  - end_date (Timestamp) :
  - quality_flag (list): desired quality flag, e.g., [2, 255]

  Returns:
  raster (xarray): object of satellite image
  """

  county_gdf = get_gdf(county)
  county_gdf = get_gdf(county)

  county_gdf = get_gdf(county)
  dates = pd.date_range(start_date, end_date if end_date else start_date, freq='D')
  raster = bm_raster(county_gdf, product_id="VNP46A2", date_range=dates, bearer=bearer, variable="DNB_BRDF-Corrected_NTL", quality_flag_rm=quality_flag)
  return raster 

# TODO: remove, deprecated check download_county_raster_by_year 
def download_county_raster_in_folder(start_date, end_date, quality_flag):
  """
  Downloads the raster image for all Florida counties within the specified range.  

  Parameters:
  - start_date (str): Start date in 'YYYY-MM-DD' format.
  - end_date (str): End date in 'YYYY-MM-DD' format.
  - quality_flag (int): Quality flag to select specific raster data.

  Returns:
  - None
  """

  desired_dates = pd.date_range(start_date, end_date, freq='D')
  county_names = get_county_names_from_state_gdf()

  base_dataset_path = '/groups/mli/multimodal_outage/data/black_marble'
  flag_dataset_path = os.path.join(base_dataset_path, str(quality_flag))
  log_folder_path = os.path.join(base_dataset_path, 'download_logs')
  csv_file_path = os.path.join(log_folder_path, f'quality_flag_{quality_flag}.csv')

  os.makedirs(flag_dataset_path, exist_ok=True)
  os.makedirs(log_folder_path, exist_ok=True)

  if os.path.exists(csv_file_path):
    df = pd.read_csv(csv_file_path, index_col=0, parse_dates=True, infer_datetime_format=True)
    df = df.reindex(df.index.union(desired_dates))
  else:
    df = pd.DataFrame(index=pd.to_datetime(desired_dates).normalize(), columns=county_names)
    df.to_csv(csv_file_path)
  
  df.index = df.index.normalize()

  for day in desired_dates:
    start_time = datetime.now() 
    for county in county_names:
   
      flag_county_dataset_path = os.path.join(flag_dataset_path, county)
      os.makedirs(flag_county_dataset_path, exist_ok=True)
      pickle_path = os.path.join(flag_county_dataset_path, f"{day.strftime('%Y_%m_%d')}.pickle")

      # Check if the value is already filled in before any processing
      if pd.isna(df.loc[day, county]):
        try:
          raster = download_county_raster(county, day, quality_flag)
          df.loc[day, county] = 1
          
          with open(pickle_path, 'wb') as pickled_raster:
            pickle.dump(raster, pickled_raster)
#          print(f"Filled in value for {county} on {day}", flush=True)
        except Exception as e:
          print(f"No data found for {county} on {day}, error: {e}", flush=True)
          df.loc[day, county] = 0
      else:
        print(f"Value for {county} on {day} already filled, skipping.", flush=True)
    
    df.to_csv(csv_file_path)
    end_time = datetime.now()
    time_taken = end_time - start_time
    print(f'Saved satellite images of all counties for {day}. Processing took {time_taken}.')



def download_county_raster_by_year(quality_flag, first_year, last_year):
  """
  Download satellite image rasters for all FL counties one year at a time.

  Parameters:
  - quality_flag (list): desired quality, e.g., [2, 255]
  - first_year (str): year we want the download to start from
  - last_year (str): year we want the download to end

  Returns:
  - N/A
  """

  year_start_date = pd.date_range(first_year, last_year, freq='YS')
  year_end_date = pd.date_range(first_year, last_year, freq='YE')

  special_year_start_date = list(year_start_date) 
  special_year_end_date = list(year_end_date)

  if first_year == '2012':
    print('special start date')
    special_year_start_date[0] = pd.Timestamp('2012-01-19') 
    year_start_date = pd.DatetimeIndex(special_year_start_date)
 
  if last_year == '2025':
    print(f'special end date')
    special_year_end_date[-1] = pd.Timestamp('2024-04-17')  
    year_end_date = pd.DatetimeIndex(special_year_end_date)

  county_names = get_county_names_from_state_gdf()

  base_dataset_path = '/groups/mli/multimodal_outage/data/black_marble'  
  flag_dataset_path = os.path.join(base_dataset_path, "hq/original") # hq because quality_flag_rm=[2, 255]

  os.makedirs(flag_dataset_path, exist_ok=True)

  for year in range(len(year_start_date)):
    for county in county_names:
      flag_county_dataset_path = os.path.join(flag_dataset_path, county)
      os.makedirs(flag_county_dataset_path, exist_ok=True)
      print(f'Trying {county} for {year_start_date[year]}.') 
      try:
        year_raster = download_county_raster(county, quality_flag, year_start_date[year], year_end_date[year])
        num_days_retrieved = year_raster.sizes['time']
        
        for day_idx in range(num_days_retrieved):
          raster = year_raster.isel(time=day_idx)
          day = str(raster.coords['time'])[-10:].replace('-', '_')

          pickle_path = os.path.join(flag_county_dataset_path, f"{day}.pickle")
          with open(pickle_path, 'wb') as pickled_raster:
            pickle.dump(raster, pickled_raster)
      except Exception as e:
        print(f"Error downloading data for {county} on {year}. Try again. Error: {e}", flush=True)
  print('Successfully saved raster images from 2012 to 2023.')


def download_annual_composite(gdf, date):
  annual_composite = bm_raster(gdf, product_id="VNP46A4", date_range=date, bearer=bearer)
  return annual_composite


def get_county_names_from_state_gdf(state_gdf=None):
  """
  Parameters:
  state_gdf (geodataframe): gdf of state we want to find county names for

  Returns:
  normalized_county_names (list): array of county names for Florida (lowercase, no spaces or hyphens)
  """
 

  if state_gdf == None:
    gdf = GADMDownloader(version="4.0").get_shape_data_by_country_name(country_name="USA", ad_level=2)
    state_gdf = gdf[gdf['NAME_1'] == 'Florida']

  county_names = list(state_gdf['NAME_2'].values)
  normalized_county_names = [name.lower().replace(' ', '_').replace('-', '_') for name in county_names]

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

  gdf = GADMDownloader(version="4.0").get_shape_data_by_country_name(country_name="USA", ad_level=2)
  florida_gdf = gdf[gdf['NAME_1'] == 'Florida']

  county_names = get_county_names_from_state_gdf(florida_gdf)

  counties_dims = {}

  for county_name in county_names:
    county_gdf = florida_gdf[florida_gdf['NAME_2'] == county_name]

    county_raster = bm_raster(county_gdf, product_id="VNP46A2", date_range="2024-01-01", bearer=bearer, quality_flag_rm=[255])

    dims = get_county_dims_from_county_raster(county_raster)

    counties_dims[county_name] = dims

  return counties_dims


# deprecated check blackmarblepy documentation
def count_light_pixels(ntl_array):
  non_zero_non_nan_count = np.sum(np.logical_and(ntl_array != 0, ~np.isnan(ntl_array)))
  return non_zero_non_nan_count


# Calculate the mean radiance per day using the daily raster images
def calculate_daily_percent_normal_mean_radiance(raster_dataset, annual_non_zero_non_nan_count, annual_composite, filter_cloudy_days=True):
    """
    Loop through each day of the year and calculate the mean daily radiance.
    
    Parameters:
    raster_dataset: Raster satellite image dataset with time and radiance data.
    filter_cloudy_days: Boolean to decide whether to filter days based on cloud cover.
    
    Returns:
    percent_normal_dict (dictionary): "percent of normal" radiance for each day of the year
    """
    
    percent_normal_dict = {}

    # Loop through each day in the dataset
    for i in range(len(raster_dataset['time'])):
        time_index = np.datetime_as_string(raster_dataset['time'][i].values.astype('datetime64[D]'))
        daily_ntl = raster_dataset["Gap_Filled_DNB_BRDF-Corrected_NTL"][i].values
        on_pixel_count = count_light_pixels(daily_ntl)
        daily_ntl = daily_ntl[:-1] # exclude the last entry from the first dim since the annual ntl array is (153, 147) while daily is (154, 147)
        
        if filter_cloudy_days:
            # Assuming a condition to filter out days based on pixel count
            if on_pixel_count < (annual_non_zero_non_nan_count * 0.95):
                continue  # Skip the calculation for this day

       # if i == 0:
       #   save_daily_raster_image(raster_dataset["Gap_Filled_DNB_BRDF-Corrected_NTL"][i], "NTL in Bay on 2018-01-01", "bay_ntl_2018_01_01.png")

        non_zero_daily_ntl = daily_ntl[daily_ntl != 0]
        nonzero_mean_radiance = np.nanmean(non_zero_daily_ntl)
        nonnan_indices_of_nonzero_daily_ntl = np.where(~np.isnan(non_zero_daily_ntl))[0]
      
        relevant_mean_annual_radiance = calculate_mean_annual_radiance_with_relevant_pixels(annual_composite, nonnan_indices_of_nonzero_daily_ntl)

        daily_percent_normal_radiance = nonzero_mean_radiance / relevant_mean_annual_radiance
        percent_normal_dict[time_index] = daily_percent_normal_radiance
    
    return percent_normal_dict


def calculate_mean_annual_radiance_with_relevant_pixels(annual_composite, relevant_pixels_indices):
  """
  Find the mean radiance of an annual composite using only relevant pixels. Relevant
  refers to non-nan or nonzero in a daily raster.

  Parameters:
  annual_composite (xarray): raster image representing the annual nighttime lights
  relevant_pixels_indices (list): indices of the non-nan or nonzero pixels in daily raster

  Returns:
  relevant_mean_annual_radiance (float): scalar value representing mean nighttime light of the annual composite
  """

  # get ntl array from xarray
  annual_ntl = annual_composite["NearNadir_Composite_Snow_Free"].values[0]

  # filter by relevant pixel index, non-nan, nonzero
  nonnan_annual_ntl = annual_ntl.flatten()[relevant_pixels_indices]
  nonzero_nonnan_annual_ntl = nonnan_annual_ntl[nonnan_annual_ntl != 0]
  
  # get mean
  relevant_mean_annual_radiance = np.nanmean(nonzero_nonnan_annual_ntl)

  return relevant_mean_annual_radiance




#These functions are outdated. 
#TODO: Check BlackMarblePy documention and see how they find nonzero, zero, or nan pixels

# Calculate the number of light pixels per day using the daily raster images
def calculate_daily_num_light_pixels(raster_dataset):
  
  daily_num_light_pixels_dict = {}

  # loop thru each day
  for i in range(len(raster_dataset['time'])):
    time_index = np.datetime_as_string(raster_dataset['time'][i].values.astype('datetime64[D]'))
    daily_ntl = raster_dataset["Gap_Filled_DNB_BRDF-Corrected_NTL"][i].values
    num_light_pixels = count_light_pixels(daily_ntl)

    daily_num_light_pixels_dict[time_index] = num_light_pixels
  
  return daily_num_light_pixels_dict


# Calculate the number of zero pixels per day using the daily raster images  
def calculate_daily_num_zero_pixels(raster_dataset):
  
  daily_num_zero_pixels_dict = {}

  # loop thru each day
  for i in range(len(raster_dataset['time'])):
    time_index = np.datetime_as_string(raster_dataset['time'][i].values.astype('datetime64[D]'))
    daily_ntl = raster_dataset["Gap_Filled_DNB_BRDF-Corrected_NTL"][i].values 

    zero_pixel_count = np.sum(daily_ntl == 0)
  
    daily_num_zero_pixels_dict[time_index] = zero_pixel_count

  return daily_num_zero_pixels_dict


# Calculate the number of nan pixels per day using the daily raster images 
def calculate_daily_num_nan_pixels(raster_dataset):
  
  daily_num_nan_pixels_dict = {}
    
  for i in range(len(raster_dataset['time'])):
    time_index = np.datetime_as_string(raster_dataset['time'][i].values.astype('datetime64[D]'))
    daily_ntl = raster_dataset["Gap_Filled_DNB_BRDF-Corrected_NTL"][i].values
    nan_pixel_count = np.sum(np.isnan(daily_ntl))
    
    daily_num_nan_pixels_dict[time_index] = nan_pixel_count

  return daily_num_nan_pixels_dict

# above 3 functions are outdates


def plot_daily_radiance(radiance_array, labels):
  # Plot the mean radiance and save the plot
  plt.bar(labels, radiance_array)
  plt.title(f'{county_name}, FL: nonzero Daily mean NTL in {year}')  
  plt.xlabel('day')
  plt.ylabel('mean radiance')


  # save
  #plt.savefig(f'{county_name}_nonzero_daily_radiance_{year}.png', dpi=300)

def save_daily_raster_figure(raster_image, title, save_path):
  """
  Plots the raster image onto a map. Includes axes, colorbar, and background map.
  """
  plt.rcParams["figure.figsize"] = (18, 10)

  fig, ax = plt.subplots()
  raster_image.plot.pcolormesh(
      ax=ax,
      cmap="cividis",
      robust=True,
  )
  cx.add_basemap(ax, crs="EPSG:4326")

  ax.text(
      0,
      -0.1,
      f"Source: NASA Black Marble",
      ha="left",
      va="center",
      transform=ax.transAxes,
      fontsize=10,
      color="black",
      weight="normal",
  )

  ax.set_title(title, fontsize=20);
  plt.savefig(save_path, dpi=300)

def save_daily_raster_image(raster_image, save_path):
  """
  Saves image of daily raster to folder.
  """
  plt.rcParams["figure.figsize"] = (18, 10)

  fig, ax = plt.subplots()
  raster_image['Gap_Filled_DNB_BRDF-Corrected_NTL'].isel(time=0).plot.pcolormesh(
      ax=ax,
      cmap="cividis",
      robust=True,
      add_colorbar=False
  )

  ax.axis('off')

  plt.savefig(save_path, dpi=300)
  plt.close(fig)

def save_annual_raster_image(gdf, annual_raster_image, date, title, save_path):
  plt.rcParams["figure.figsize"] = (18, 10)
  fig, ax = plt.subplots()
  annual_raster_image["NearNadir_Composite_Snow_Free"].sel(time=date).plot.pcolormesh(
      ax=ax,
      cmap="cividis",
      robust=True,
  )
  cx.add_basemap(ax, crs=gdf.crs.to_string())
  ax.text(
      0,
      -0.1,
      f"Source: NASA Black Marble",
      ha="left",
      va="center",
      transform=ax.transAxes,
      fontsize=10,
      color="black",
      weight="normal",
  )

  ax.set_title(title, fontsize=20);
  plt.savefig(save_path, dpi=300)

def get_zips_from_county(county):
  """
  Parameters:
  county (str) : county name (e.g. 'orange')
  
  Returns:
  zip_codes (list): zip codes for desired county
  """

  # open dataframe with county (col1) and zipcode list (col2)
  with open('county_zips_df.pickle', 'rb') as file:
    data = pickle.load(file)

  # set the 'County' column to index
  # TODO-low-priority: resave the pickle file with set_index already done
  data.set_index('County', inplace=True)
  zip_codes = data.loc[county]['Zip Codes']

  return zip_codes

def get_total_customers_in_county(county):
  """
  Find the total number of customers for a specified county.

  Parameters:
  -  county (str) = name of county

  Returns:
  - total_customers_in_county (int): sum of the total number of customers from all available zip codes
  """


  # total customers per county is consistent for every year
  # doesnt matter which year/month we open
  zip_level_data = pd.read_csv('/lustre/fs1/groups/mli/multimodal_outage/data/FL_Zip_Outages_201411_202403/monthly_zip_outage_data_2024_01.csv')

  # get list of zip codes for desired county
  zips_in_county = get_zips_from_county(county)

  # filter the data by zip code
  county_zip_data = zip_level_data[zip_level_data['Zip Code'].isin(zips_in_county)]

  # get the total customers in our desired county
  total_customers_in_county = county_zip_data.groupby('Zip Code')['Total Customers'].max().sum()

  return total_customers_in_county


def plot_sat_images_diff_qualities(var_quality_images, dates):
  """

  Plots a sample_count x 4 grid of satellite images, each row represents a different randomly picked day. Each column represents the varying qualities.

  Parameters:
  - var_quality_images (dict): Dictionary of xarray.DataArray where keys represent quality types and values are the corresponding images.
  - dates (list): List of dates corresponding to each row.

  Returns:
  N/A
  """
  
  sample_count = var_quality_images[0].dims['time']

  plt.rcParams["figure.figsize"] = (60, 50)
  fig, axes = plt.subplots(sample_count, 4) # 4 types of qualities

  quality_types = list(var_quality_images.keys())

  for idx, ax in enumerate(axes[0]):  # Only need to set titles on the first row of axes
    ax.set_title(quality_types[idx])

  for time in range(sample_count):
    for quality_idx, quality_type in enumerate(quality_types):
      image = var_quality_images[quality_type]['Gap_Filled_DNB_BRDF-Corrected_NTL'].isel(time=time)
      ax = axes[time, quality_idx]

      image.plot.pcolormesh(ax=ax, cmap="cividis", robust=True, add_colorbar=False)
  
      ax.axis('off')
      
    axes[time][0].set_ylabel(dates[time], size='large', rotation=0, labelpad=0, verticalalignment='center')

  plt.tight_layout() 
  plt.savefig('effect_of_diff_qualities.png', dpi=300)


def preview_sat_image_qualities(county_name):
  """
  Preview the differences in satellite image quality of a county.

  Parameters:
  - county_name (str): name of county name to preview (e.g., 'bay')

  Returns:
  - dict_of_xarray_dataset (dict): quality flag as keys, xarray.DataArray as values
  """


  gdf = get_gdf(county_name)

  dates = ["2018-09-02", "2018-06-08", "2018-02-22", "2018-04-17"]
  
  quality_list = [0, 1, 2, 255]
  dict_of_xarray_dataset = {}

  for quality in quality_list:
    plot_sat_images_diff_qualities(dict_of_xarray_dataset, dates)

  return dict_of_daily_raster_arrays

