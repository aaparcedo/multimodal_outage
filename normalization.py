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


def flatten_and_concatenate_images(images):
    flattened_data = []
    for img in images:
        flattened_data.append(img.flatten())
    return np.concatenate(flattened_data)


def reshape_data_to_images(flattened_data, original_shapes):
    reshaped_images = []
    index = 0
    for shape in original_shapes:
        size = np.prod(shape)
        reshaped_image = flattened_data[index:index + size].reshape(shape)
        reshaped_images.append(reshaped_image)
        index += size
    return reshaped_images


def normalize_and_transform_images(main_directory, root_directory, dates):
    percent_normal_dir = os.path.join(root_directory, 'percent_normalized')
    os.makedirs(percent_normal_dir, exist_ok=True)

    for county_dir in os.listdir(main_directory):
        county_path = os.path.join(main_directory, county_dir)
        if os.path.isdir(county_path):
            images = load_images_for_dates(county_path, dates)
            print("images loaded")
            if not images:
                continue

            original_shapes = [img.shape for img in images]
            flattened_data = flatten_and_concatenate_images(images)
            mean = np.mean(flattened_data)
            std = np.std(flattened_data)
            standardized_data = (flattened_data - mean) / std
            normal_dist_data = norm.ppf(norm.cdf(standardized_data))
            min_val = np.min(normal_dist_data)
            max_val = np.max(normal_dist_data)
            scaled_data = 255 * (normal_dist_data -
                                 min_val) / (max_val - min_val)
            scaled_data = scaled_data.astype(np.uint8)
            transformed_images = reshape_data_to_images(
                scaled_data, original_shapes)

            county_normalized_directory = os.path.join(
                percent_normal_dir, county_dir)
            print("making county dir")
            os.makedirs(county_normalized_directory, exist_ok=True)

            for i, img in enumerate(transformed_images):
                transformed_image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                transformed_image_path = os.path.join(
                    county_normalized_directory, f'{dates[i]}.png')
                cv2.imwrite(transformed_image_path, transformed_image_bgr)
                print(f"Transformed image saved as {transformed_image_path}")

            colors = ('r', 'g', 'b')
            channel_ids = (0, 1, 2)

            plt.figure(figsize=(20, 10))
            for channel_id, color in zip(channel_ids, colors):
                combined_channel_data = np.concatenate(
                    [img[:, :, channel_id].flatten() for img in transformed_images])
                histogram, bin_edges = np.histogram(
                    combined_channel_data, bins=256, range=(0, 256))
                plt.plot(bin_edges[0:-1], histogram, color=color)

            plt.title(
                f'Color Channel Histogram of Transformed Images for County {county_dir}')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.legend(['Red Channel', 'Green Channel', 'Blue Channel'])

            transformed_histogram_output_path = os.path.join(
                county_normalized_directory, 'transformed_color_histogram_combined.png')
            plt.savefig(transformed_histogram_output_path,
                        bbox_inches='tight', pad_inches=0)
            plt.close()

            print(f"Histogram plot of transformed images for county {
                  county_dir} saved as {transformed_histogram_output_path}")


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
