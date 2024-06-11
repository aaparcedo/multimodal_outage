import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def plot_ntl_data(ntl, date, exp_factor=1, save_path='ntl_plot.jpg'):
    ntl_data = ntl["Gap_Filled_DNB_BRDF-Corrected_NTL"].values
    ntl_data_masked = np.ma.masked_where(ntl_data == 0, ntl_data)
    ntl_data_exp = np.where(ntl_data_masked > 0, ntl_data_masked ** exp_factor, ntl_data_masked)
    
    if np.all(np.isnan(ntl_data_exp)):
        print("Warning: All data values are NaN.")
        norm = Normalize(vmin=0, vmax=1)  # Set a default normalization
    else:
        norm = Normalize(vmin=0, vmax=np.nanpercentile(ntl_data_exp, 95))
    
    cmap = plt.cm.gray
    cmap.set_bad(color='black')
    cmap.set_under(color='black')
    
    fig, ax = plt.subplots()
    ax.imshow(ntl_data_exp, cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()



# Define the directory containing the county directories and the output directory
base_dir = "/groups/mli/multimodal_outage/data/black_marble/hq/original_gap_filled/"
output_dir = "/groups/mli/multimodal_outage/data/black_marble/hq/ntl_gray_gap_filled/"
os.makedirs(output_dir, exist_ok=True)

# Define the list of dates

dates = ['2018_09_10.jpg', '2018_09_11.jpg', '2018_09_12.jpg', '2018_09_13.jpg', '2018_09_14.jpg', '2018_09_15.jpg', '2018_09_16.jpg', '2018_09_17.jpg', '2018_09_18.jpg',
         '2018_09_19.jpg', '2018_09_20.jpg', '2018_09_21.jpg', '2018_09_22.jpg', '2018_09_23.jpg', '2018_09_24.jpg', '2018_09_25.jpg', '2018_09_26.jpg', '2018_09_27.jpg',
         '2018_09_28.jpg', '2018_09_29.jpg', '2018_09_30.jpg', '2018_10_01.jpg', '2018_10_02.jpg', '2018_10_03.jpg', '2018_10_04.jpg', '2018_10_05.jpg', '2018_10_06.jpg', '2018_10_07.jpg', '2018_10_08.jpg', '2018_10_09.jpg', '2018_10_10.jpg', '2018_10_11.jpg', '2018_10_12.jpg', '2018_10_13.jpg', '2018_10_14.jpg', '2018_10_15.jpg', '2018_10_16.jpg', '2018_10_17.jpg', '2018_10_18.jpg', '2018_10_19.jpg', '2018_10_20.jpg', '2018_10_21.jpg', '2018_10_22.jpg', '2018_10_23.jpg', '2018_10_24.jpg', '2018_10_25.jpg', '2018_10_26.jpg', '2018_10_27.jpg', '2018_10_28.jpg', '2018_10_29.jpg', '2018_10_30.jpg', '2018_10_31.jpg', '2018_11_01.jpg', '2018_11_02.jpg', '2018_11_03.jpg', '2018_11_04.jpg', '2018_11_05.jpg', '2018_11_06.jpg', '2018_11_07.jpg', '2018_11_08.jpg', '2022_08_27.jpg', '2022_08_28.jpg', '2022_08_29.jpg', '2022_08_30.jpg', '2022_08_31.jpg', '2022_09_01.jpg', '2022_09_02.jpg', '2022_09_03.jpg', '2022_09_04.jpg', '2022_09_05.jpg', '2022_09_06.jpg', '2022_09_07.jpg', '2022_09_08.jpg', '2022_09_09.jpg', '2022_09_10.jpg', '2022_09_11.jpg', '2022_09_12.jpg', '2022_09_13.jpg', '2022_09_14.jpg', '2022_09_15.jpg', '2022_09_16.jpg', '2022_09_17.jpg', '2022_09_18.jpg', '2022_09_19.jpg', '2022_09_20.jpg', '2022_09_21.jpg', '2022_09_22.jpg', '2022_09_23.jpg', '2022_09_24.jpg', '2022_09_25.jpg', '2022_09_26.jpg', '2022_09_27.jpg', '2022_09_28.jpg', '2022_09_29.jpg', '2022_09_30.jpg', '2022_10_01.jpg', '2022_10_02.jpg', '2022_10_03.jpg', '2022_10_04.jpg', '2022_10_05.jpg', '2022_10_06.jpg', '2022_10_07.jpg', '2022_10_08.jpg', '2022_10_09.jpg', '2022_10_10.jpg', '2022_10_11.jpg', '2022_10_12.jpg', '2022_10_13.jpg', '2022_10_14.jpg', '2022_10_15.jpg', '2022_10_16.jpg', '2022_10_17.jpg', '2022_10_18.jpg', '2022_10_19.jpg', '2022_10_20.jpg', '2022_10_21.jpg', '2022_10_22.jpg', '2022_10_23.jpg', '2022_10_24.jpg', '2022_10_25.jpg', '2023_07_31.jpg', '2023_08_01.jpg', '2023_08_02.jpg', '2023_08_03.jpg', '2023_08_04.jpg', '2023_08_05.jpg', '2023_08_06.jpg', '2023_08_07.jpg', '2023_08_08.jpg', '2023_08_09.jpg', '2023_08_10.jpg', '2023_08_11.jpg', '2023_08_12.jpg', '2023_08_13.jpg', '2023_08_14.jpg', '2023_08_15.jpg', '2023_08_16.jpg', '2023_08_17.jpg', '2023_08_18.jpg', '2023_08_19.jpg', '2023_08_20.jpg', '2023_08_21.jpg', '2023_08_22.jpg', '2023_08_23.jpg', '2023_08_24.jpg', '2023_08_25.jpg', '2023_08_26.jpg', '2023_08_27.jpg', '2023_08_28.jpg', '2023_08_29.jpg', '2023_08_30.jpg', '2023_08_31.jpg', '2023_09_01.jpg', '2023_09_02.jpg', '2023_09_03.jpg', '2023_09_04.jpg', '2023_09_05.jpg', '2023_09_06.jpg', '2023_09_07.jpg', '2023_09_08.jpg', '2023_09_09.jpg', '2023_09_10.jpg', '2023_09_11.jpg', '2023_09_12.jpg', '2023_09_13.jpg', '2023_09_14.jpg', '2023_09_15.jpg', '2023_09_16.jpg', '2023_09_17.jpg', '2023_09_18.jpg', '2023_09_19.jpg', '2023_09_20.jpg', '2023_09_21.jpg', '2023_09_22.jpg', '2023_09_23.jpg', '2023_09_24.jpg', '2023_09_25.jpg', '2023_09_26.jpg', '2023_09_27.jpg', '2023_09_28.jpg']

dates = [date.split('.')[0] for date in dates]

#overall_min = np.inf
#overall_max = 0

# Loop through all counties
for county in os.listdir(base_dir):
    county_dir = os.path.join(base_dir, county)
    if os.path.isdir(county_dir):
        # Create output directory for each county
        county_output_dir = os.path.join(output_dir, county)
        os.makedirs(county_output_dir, exist_ok=True)
        
        for date in dates:
            pickle_filename = f'{date}.pickle'
            pickle_filepath = os.path.join(county_dir, pickle_filename)
            if os.path.exists(pickle_filepath):
                # Load the pickle file
                with open(pickle_filepath, 'rb') as f:
                    ntl = pickle.load(f)
                
                # Call the plot_ntl_data function
                output_filename = f'{date}.jpg'
                output_filepath = os.path.join(county_output_dir, output_filename)
                plot_ntl_data(ntl, date, exp_factor=1, save_path=output_filepath)
