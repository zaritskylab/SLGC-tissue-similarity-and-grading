"""Liver data analysis
The script should be ran and will read imzML files, process data and create
relevant plot such as spatial distribution of feature counts.

"""

import os
import random
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import sem, ttest_ind
from pyimzml.ImzMLParser import ImzMLParser
from processing import process
from correlation import correlation_analysis
from utils import read_msi


def spectra_num_features(spectra: np.ndarray, percentage=0.3) -> int:
  """
  Calculates the number of spectra features based on a given intensity
  percentage threshold.

  Args:
    spectra (np.ndarray): The array of spectra intensities.
    percentage (float, optional): The threshold percentage for considering a
        peak significant. Defaults to 0.3.

  Returns:
    int: The number of significant spectra features.
    
  """
  # Find all local maxima in the spectra
  indexes = find_peaks(spectra)[0]
  if len(indexes) == 0:
    # Return 0 if no peaks are found
    return 0
  # Find the highest peak value
  top_peak = spectra[indexes].max()
  # Count and return peaks that are at least a certain percentage of the top
  # peak value
  return (spectra[indexes] >= (top_peak * percentage)).sum()


def msi_sum_spectra_num_features(
    p: ImzMLParser, mask: np.ndarray, percentage=0.3
) -> int:
  """
  Calculates the number of significant spectra features from the sum of 
  spectra in a masked area.

  Args:
    p (ImzMLParser): The ImzML parser object.
    mask (np.ndarray): A binary mask to select specific regions in the MSI data.
    percentage (float, optional): The threshold percentage for considering a
        peak significant. Defaults to 0.3.

  Returns:
    int: The number of significant spectra features.
  """
  # Read MSI data and get the intensity matrix
  msi = read_msi(p)[1]
  # Apply the mask to the MSI data
  spectras = msi[mask]
  # Sum the spectra within the masked region
  sum_spectra = np.sum(spectras, axis=0)
  # Calculate and return the number of significant spectra features
  return spectra_num_features(sum_spectra, percentage=percentage)


def msi_mean_spectra_num_features(
    p: ImzMLParser, mask: np.ndarray, percentage=0.3
) -> int:
  """
  Calculates the number of significant spectra features from the mean of 
  spectra in a masked area.

  Args:
    p (ImzMLParser): The ImzML parser object.
    mask (np.ndarray): A binary mask to select specific regions in the MSI data.
    percentage (float, optional): The threshold percentage for considering a
        peak significant. Defaults to 0.3.

  Returns:
    int: The number of significant spectra features.
  """
  # Read MSI data and get the intensity matrix
  msi = read_msi(p)[1]
  # Apply the mask to the MSI data
  spectras = msi[mask]
  # Calculate the mean of the spectra within the masked region
  mean_spectra = np.mean(spectras, axis=0)
  # Calculate and return the number of significant spectra features
  return spectra_num_features(mean_spectra, percentage=percentage)


def msi_mean_num_features(
    p: ImzMLParser, mask: np.ndarray, percentage=0.3
) -> int:
  """
  Calculates the mean number of spectra features across all spectra in a 
  masked area.

  Args:
    p (ImzMLParser): The ImzML parser object.
    mask (np.ndarray): A binary mask to select specific regions in the MSI data.
    percentage (float, optional): The threshold percentage for considering a
        peak significant. Defaults to 0.3.


  Returns:
    int: The mean number of significant spectra features.
  """
  # Read MSI data and get the intensity matrix
  msi = read_msi(p)[1]
  # Apply the mask to the MSI data
  spectras = msi[mask]
  # List to store the number of features for each spectrum
  msi_num_features = []
  # Iterate over each spectrum and calculate its number of features
  for spectra in spectras:
    msi_num_features.append(
        spectra_num_features(spectra, percentage=percentage)
    )
  # Calculate and return the mean number of features
  return np.mean(msi_num_features)


def msi_spatial_num_features(
    p: ImzMLParser, mask: np.ndarray, percentage=0.3
) -> np.ndarray:
  """
  Calculates the spatial distribution of spectra features in a masked area of
  MSI data.

  Args:
    p (ImzMLParser): The ImzML parser object.
    mask (np.ndarray): A binary mask to select specific regions in the MSI data.
    percentage (float, optional): The threshold percentage for considering a
        peak significant. Defaults to 0.3.

  Returns:
    np.ndarray: A 2D array representing the number of spectra features at each 
        position.
  """
  # Read MSI data and get the intensity matrix
  msi = read_msi(p)[1]
  # Initialize an array to store the spatial number of features
  msi_spatial_num_features = np.zeros(mask.shape)
  # Loop over each pixel in the mask
  for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
      if mask[i, j]:
        # Calculate the number of features for the current pixel
        msi_spatial_num_features[
            i, j] = spectra_num_features(msi[i, j, :], percentage=percentage)
      else:
        # Set the value to NaN if the pixel is not in the mask
        msi_spatial_num_features[i, j] = np.nan
  return msi_spatial_num_features


def concatenate_images_array(
    img_1: np.ndarray, img_2: np.ndarray
) -> np.ndarray:
  """
  Concatenates two images horizontally with a separator and applies a 
  vertical shift to the second image.

  Args:
      img_1 (: np.ndarray): The first image to be concatenated. 
          It should be a 2D numpy array.
      img_2 (: np.ndarray): The second image to be concatenated. 
          It should also be a 2D numpy array.

  Returns:
      np.ndarray: A new image created by horizontally concatenating `img_1` 
          and `img_2`, with a vertical shift applied to `img_2`.

  """
  # Set the width of the separator between images
  separator = 1
  # Calculate the new shape for the concatenated image
  # It's the height of the first image and the sum of widths of both
  # images plus the separator
  new_shape = (img_1.shape[0], img_1.shape[1] + img_2.shape[1] + separator)
  # Initialize a new image with zeros, of the calculated shape
  new_image = np.zeros(new_shape)
  # Place the first image in the beginning part of the new image
  new_image[:, :img_1.shape[1]] = img_1
  # Set the separator area to NaN values
  new_image[:, img_1.shape[1]:img_1.shape[1] + separator] = np.nan
  # Set the amount of vertical shift for the second image
  shift_y = 13
  # Place the second image in the remaining part of the new image,
  # applying the vertical shift to it
  new_image[:, img_1.shape[1] + separator:] = np.roll(img_2, -shift_y, axis=0)
  # Return the concatenated image
  return new_image


def plot_spatial_num_features(
    parsers: Dict[str, ImzMLParser], masks: Dict[str, np.ndarray],
    metadata_df: pd.DataFrame, save_path: Path
) -> None:
  """
  Plots and saves an image that represents spatially the number of features 
  in each msi.

  Args:
    parsers (Dict[str, ImzMLParser]): A dictionary containing ImzMLParser 
        objects with keys as sample names.
    masks (Dict[str, np.ndarray]): A dictionary containing masks for each
        sample.
    metadata_df (pd.DataFrame): DataFrame containing metadata for the
        samples.
    save_path (Path): Path object where the output image will be saved.
  """
  # Dictionary to store spatial number of features for each sample
  spatial_num_features = {}
  # Loop through each parser and calculate spatial number of features
  for name, p in parsers.items():
    spatial_num_features[name] = msi_spatial_num_features(p, masks[name])
  # List to store images for each group
  images = []
  # Group the metadata by file name and process each group
  for group_name, group in metadata_df.groupby("file_name"):
    # Initialize an empty image array
    img = np.zeros((group["y_max"].max(), group["x_max"].max() + 1))
    # Loop through each row in the group and update the image array
    for index, row in group.iterrows():
      img[:, row["x_min"]:row["x_max"] +
          1] = spatial_num_features[row.sample_file_name]
    # Append the processed image to the images list
    images.append(img)
  # Create a combined image from the first two images in the list
  combined_img = concatenate_images_array(
      np.rot90(images[1], 1), np.rot90(images[0], 1)
  )[:-2, 4:-6]
  # Create a figure for plotting
  fig, ax = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True)
  # Show the combined image
  shw = ax.imshow(combined_img, cmap='jet')
  # Add a color bar to the plot
  bar = fig.colorbar(shw, orientation='vertical')
  bar.set_label(
      'Number of features', rotation=270, labelpad=15, fontweight='bold',
      fontsize=14, color='0.2'
  )
  bar.outline.set_edgecolor('0.2')
  bar.ax.tick_params(labelsize=14, width=2.5, color='0.2')
  for l in bar.ax.get_yticklabels():
    l.set_fontweight('bold')
    l.set_color('0.2')
  # Set the ticks for the color bar
  ticks = np.linspace(
      combined_img[~np.isnan(combined_img)].max() - 22,
      combined_img[~np.isnan(combined_img)].min(), 6
  )
  bar.set_ticks(ticks)
  bar.set_ticklabels(['{:.0f}'.format(t) for t in ticks])
  # Remove the axis for a cleaner look
  plt.axis("off")
  plt.tight_layout()
  # Save the plot
  plt.savefig(
      save_path.joinpath("maps_of_the_number_of_features.png"),
      bbox_inches='tight', dpi=1200, transparent=True
  )
  # Display the plot
  plt.show()


def plot_num_features(
    parsers: Dict[str, ImzMLParser], masks: Dict[str, np.ndarray],
    metadata_df: pd.DataFrame, save_path: Path
):
  """
  Plots and saves a bar chart comparing the number of features between standard 
  and optimized conditions

  Args:
    parser (Dict[str, ImzMLParser]): A dictionary containing ImzMLParser 
        objects with keys as sample names.
    masks (Dict[str, np.ndarray]): A dictionary containing masks for each
        sample.
    metadata_df (pd.DataFrame): DataFrame containing metadata for the
        samples.
    save_path (Path): Path object where the output image will be saved.
  """
  # Create dict to store num features
  num_features = {}
  # Group the metadata by file name and process each group
  for group_name, group in metadata_df.groupby("file_name"):
    # Loop through each row in the group
    for index, row in group.iterrows():
      # Get MSO num features
      msi_features = msi_mean_num_features(
          parsers[row.sample_file_name], masks[row.sample_file_name]
      )
      # Update group number of features list
      num_features[group_name] = num_features.get(group_name,
                                                  []) + [msi_features]
  # Get values for bar plot
  means = {key: np.mean(value) for key, value in num_features.items()}
  print(means)
  sems = {key: sem(value) for key, value in num_features.items()}
  mean_values = list(means.values())[::-1]
  sem_values = list(sems.values())[::-1]
  # Extract the points for plotting on the bar chart
  std_points = num_features['220224-optimization-liver-standard-1 Analyte 1_1']
  opt_points = num_features['220224-optimization-liver-optimised-1 Analyte 1_1']
  # Get ttest p value
  stat, pvalue = ttest_ind(std_points, opt_points)
  if pvalue < 0.0001:
    p_text = "* * * *"
  elif 0.0001 <= pvalue < 0.001:
    p_text = "* * *"
  elif 0.001 <= pvalue < 0.01:
    p_text = "* *"
  elif 0.01 <= pvalue < 0.05:
    p_text = "*"
  else:
    p_text = "ns"
  # Define categories for the x-axis
  categories = ['Std', 'Opt']
  # Create the bar chart
  fig, ax = plt.subplots(1, 1, figsize=(3, 6), tight_layout=True)
  bars = ax.bar(
      categories, mean_values, yerr=sem_values, color=['tab:red', 'tab:blue'],
      capsize=10, error_kw=dict(ecolor='0.2', lw=2.5, capsize=10, capthick=2.5)
  )
  # Add individual points to the bars
  ax.plot(
      np.repeat(0, len(std_points)), std_points, 's', markersize=7, color='0.2'
  )
  ax.plot(
      np.repeat(1, len(opt_points)), opt_points, 's', markersize=7, color='0.2'
  )
  # Make ticks bold and font size 14
  plt.xticks(fontsize=14, fontweight='bold', color='0.2')
  plt.yticks(fontsize=14, fontweight='bold', color='0.2')
  # Make the tick lines thicker
  ax.tick_params(axis='both', which='major', width=2.5, color='0.2')
  ax.tick_params(axis='both', which='minor', width=2.5, color='0.2')
  # Set labels font size to 14
  ax.set_ylabel('Number of Features', fontsize=14, weight='bold', color='0.2')
  # Limit the y-axis to the max point
  max_point = max(max(std_points), max(opt_points))
  # Add a line to show the p-value
  y, h, col = max(mean_values) + max(sem_values) + 60, 5, 'k'
  ax.plot([0, 0, 1, 1], [y, y + h, y + h, y], lw=2, c='0.2')
  ax.text(
      0.5, y + h + 5, p_text, ha='center', va='bottom', color='0.2',
      fontweight="bold", fontsize=14
  )
  # Customize the spines
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2.5)
    ax.spines[axis].set_color('0.2')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  # Add space between subplots and show the plot
  plt.tight_layout()
  # Save the plot
  plt.savefig(
      save_path.joinpath("number_of_features_bar_graph.png"),
      bbox_inches='tight', dpi=1200, transparent=True
  )
  # Display the plot
  plt.show()


def main():
  """Function containing main code"""
  # Define current folder using this file
  CWD = Path(os.path.dirname(os.path.abspath(__file__)))
  # Define folder that contains the revision chip type dataset
  REV_PATH = Path("D:/Thesis/chapter_one/data/LIVER/")
  # Define folder that contains raw data
  REV_RAW_DATA = REV_PATH.joinpath("raw")
  # Define folder to save processed data
  REV_PROCESSED_DATA = REV_PATH.joinpath("processed")
  # Define file that contains dhg metadata
  METADATA_PATH = REV_PATH.joinpath("metadata.csv")
  # Define mass range start value
  MZ_START = 50
  # Define mass range end value
  MZ_END = 1200
  # Define mass resolution of the data
  MASS_RESOLUTION = 0.025
  # Define representative peaks
  REPRESENTATIVE_PEAKS = [682.58, 844.64, 860.63, 888.62, 600.49, 834.53]
  # Define random seed
  SEED = 42
  random.seed(SEED)
  np.random.seed(SEED)
  # Read metadata csv
  metadata_df = pd.read_csv(METADATA_PATH)
  # Loop over each ROI in data frame
  for index, roi in metadata_df.iterrows():
    # Define path to msi imzML file
    msi_path = os.path.join(REV_RAW_DATA, f"{roi.file_name}.imzML")
    # Define path to new msi imzML file after processing
    output_path = os.path.join(REV_PROCESSED_DATA, f"{roi.sample_file_name}")
    # Create output folder if doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # Process msi
    process(
        msi_path, output_path, roi.x_min, roi.x_max, roi.y_min, roi.y_max,
        MZ_START, MZ_END, MASS_RESOLUTION, REPRESENTATIVE_PEAKS
    )
  # Define path to save figures
  PLOT_PATH = Path(CWD / "liver/")
  # Create dict of msi parsers and masks
  parsers = {}
  masks = {}
  for folder in REV_PROCESSED_DATA.iterdir():
    name = folder.stem
    parsers[name] = ImzMLParser(folder.joinpath("meaningful_signal.imzML"))
    masks[name] = np.load(folder.joinpath("segmentation.npy"), mmap_mode='r')
  # Plot figures
  plot_spatial_num_features(parsers, masks, metadata_df, PLOT_PATH)
  plot_num_features(parsers, masks, metadata_df, PLOT_PATH)


if __name__ == '__main__':
  main()
