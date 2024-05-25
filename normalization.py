import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from datetime import datetime, timedelta


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


def replace_with_z_scores(image, mean, std):
    # Ensure the image has the same shape as mean and std
    assert image.shape == mean.shape, "Target image dimensions must match the distribution dimensions"
    print("mean: ", mean)
    print("std: ", std)
    invalid_indices = np.logical_or(np.isnan(std), std == 0)
    std[invalid_indices] = 1e-10

    # Calculate z-scores for the target image
    z_scores = (image - mean) / std

    # Normalize z-scores to 0-255 range for visualization (optional)
    z_scores_normalized = 255 * \
        (z_scores - z_scores.min()) / (z_scores.max() - z_scores.min())
    z_scores_normalized = z_scores_normalized.astype(np.uint8)

    return z_scores_normalized

# if you want to see the plotted image of the the z score image


def plot_z_score_image(z_score_image):
    z_min = np.min(z_score_image)
    z_max = np.max(z_score_image)
    plt.imshow(z_score_image, cmap='gray', vmin=z_min, vmax=z_max)
    plt.title('Z-Score Image')
    plt.axis('off')
    plt.show()

# if you want to see z score for each pixel


def plot_pixel_distributions(z_scores):
    height, width, channels = z_scores.shape[1:]

    # Loop through each pixel
    for y in range(height):
        for x in range(width):
            for channel in range(channels):
                # Extract the z-scores for the specific pixel and channel
                pixel_z_scores = z_scores[:, y, x, channel]

                # Plot the histogram
                plt.hist(pixel_z_scores, bins=50, color='blue', alpha=0.7)
                plt.title(
                    f'Z-Score Distribution at ({x}, {y}), Channel {channel}')
                plt.xlabel('Z-Score')
                plt.ylabel('Frequency')
                plt.show()


def normalize_and_transform_images(main_directory, root_directory, dates, current_day):
    for county in os.listdir(main_directory):
        county_path = os.path.join(main_directory, county)
        image_list = load_images_for_dates(county_path, dates)
        distributions = calculate_pixel_distributions(image_list)
        mean, std = get_distribution_statistics(distributions)

        # Load the target image
        target_image_path = f'/home/akotta2025/github-stuff/counties/original/alachua/{
            current_day}.jpg'
        target_image = cv2.imread(target_image_path)
        if target_image is None:
            print(f"Error: Could not load image at {target_image_path}")
            exit()
        target_image = cv2.cvtColor(
            target_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Replace pixel values with z-scores
        z_score_image = replace_with_z_scores(target_image, mean, std)

        # Save and display the z-score image

        # Convert back to BGR for saving with OpenCV
        # z_score_image_bgr = cv2.cvtColor(z_score_image, cv2.COLOR_RGB2BGR)

        new_folder_path = os.path.join(root_directory, 'z_score_images')
        os.makedirs(new_folder_path, exist_ok=True)
        z_score_image_filename = f"{current_day}.jpg"
        z_score_image_path = os.path.join(
            new_folder_path, z_score_image_filename)
        cv2.imwrite(z_score_image_path, z_score_image)
        print(f"Z-score image saved as {z_score_image_path}")


def get_prev_ten(hurricane_day):
    '''
    this function is here to mimic the horizon
    '''
    current_date = datetime.strptime(hurricane_day, '%Y_%m_%d')
    previous_10_days = [(current_date - timedelta(days=i)
                         ).strftime('%Y_%m_%d') for i in range(1, 11)]
    return previous_10_days


main_directory = '/home/akotta2025/github-stuff/counties/original'
root_directory = '/home/akotta2025/github-stuff/counties'
hurricane_days = ['2023_01_31']

for day in hurricane_days:
    prev_10 = get_prev_ten(day)
    normalize_and_transform_images(
        main_directory, root_directory, prev_10, day)
