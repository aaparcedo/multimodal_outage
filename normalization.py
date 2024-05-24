import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd


def load_images_for_dates(county_path, dates):
    image_list = []
    for date in dates:
        image_path = os.path.join(county_path, f'{date}.jpg')
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                image_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return image_list


def calculate_pixel_distributions(images):
    # Stack images to create a 4D array (num_images, height, width, channels)
    stacked_images = np.stack(images, axis=0)
    return stacked_images


def get_distribution_statistics(distributions):
    # Calculate mean and std for each pixel across all images
    mean = np.mean(distributions, axis=0)
    std = np.std(distributions, axis=0)
    return mean, std


def calculate_z_scores(distributions, mean, std):
    # Calculate z-scores for each pixel value
    z_scores = (distributions - mean) / std
    return z_scores


def plot_pixel_distribution(distributions, x, y, channel, bins=256):
    # Extract the pixel values for the specific location and channel
    pixel_values = distributions[:, y, x, channel]

    # Plot the histogram
    plt.hist(pixel_values, bins=bins, range=(0, 256), color='blue', alpha=0.7)
    plt.title(f'Pixel Value Distribution at ({x}, {y}), Channel {channel}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()


def replace_with_z_scores(image, mean, std):
    # Ensure the image has the same shape as mean and std
    assert image.shape == mean.shape, "Target image dimensions must match the distribution dimensions"

    # Calculate z-scores for the target image
    z_scores = (image - mean) / std

    # Normalize z-scores to 0-255 range for visualization (optional)
    z_scores_normalized = 255 * \
        (z_scores - z_scores.min()) / (z_scores.max() - z_scores.min())
    z_scores_normalized = z_scores_normalized.astype(np.uint8)

    return z_scores_normalized


def normalize_and_transform_images(main_directory, root_directory, dates):
    for county in os.listdir(main_directory):
        county_path = os.path.join(main_directory, county)
        image_list = load_images_for_dates(county_path, dates)
        distributions = calculate_pixel_distributions(image_list)
        mean, std = get_distribution_statistics(distributions)
        # Load the target image
        target_image_path = 'path_to_target_image.jpg'
        target_image = cv2.imread(target_image_path)
        target_image = cv2.cvtColor(
            target_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Replace pixel values with z-scores
        z_score_image = replace_with_z_scores(target_image, mean, std)

        # Save and display the z-score image
        # Convert back to BGR for saving with OpenCV
        z_score_image_bgr = cv2.cvtColor(z_score_image, cv2.COLOR_RGB2BGR)
        z_score_image_path = 'z_score_image.png'
        cv2.imwrite(z_score_image_path, z_score_image_bgr)
        print(f"Z-score image saved as {z_score_image_path}")
        z_scores = calculate_z_scores(distributions, mean, std)
        x, y, channel = 64, 64, 0
        print(
            f"Z-score for pixel at ({x}, {y}), Channel {channel}: {z_scores[:, y, x, channel]}")
        plot_pixel_distribution(distributions, x=64, y=64, channel=0)


def find_case_study_dates(size, image_paths, case_study):

    if size == 'S':
        horizon = 20  # or 90
    elif size == 'M':
        horizon = 60
    elif size == 'L':
        horizon = 90
    else:
        print('Invalid size. Please select a valid size, i.e., "S", "M", or "L"')

    timestamp_to_image = {pd.Timestamp(image_path.split('.')[0].replace(
        '_', '-')): image_path for image_path in image_paths}
    dates = [pd.Timestamp(image_path.split('.')[0].replace('_', '-'))
             for image_path in image_paths]

    case_study_indices = [dates.index(date) for date in case_study.values()]

    filtered_dates = set()
    main_directory = '/groups/mli/multimodal_outage/data/black_marble/hq/percent_normal'
    root_directory = '/groups/mli/multimodal_outage/data/black_marble/hq/'

    for case_study_index in case_study_indices:
        start_index = case_study_index - horizon
        end_index = case_study_index + horizon

        case_study_dates = dates[start_index:end_index]
        case_study_subset = dates[start_index:case_study_index]
        normalize_and_transform_images(
            main_directory=main_directory, root_directory=root_directory, dates=case_study_subset)

        filtered_dates.update(case_study_dates)

    filtered_image_paths = [timestamp_to_image[date]
                            for date in sorted(filtered_dates)]
    return filtered_image_paths


main_directory = '/groups/mli/multimodal_outage/data/black_marble/hq/percent_normal'
dates = []  # You need to specify the dates here
case_study = {'h_idalia': pd.Timestamp('2023-08-30')}

for county in os.listdir(main_directory):
    county_path = os.path.join(main_directory, county)
    image_paths = os.listdir(county_path)
    find_case_study_dates('S', image_paths, case_study)
