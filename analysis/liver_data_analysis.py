"""Liver data analysis
The script should be ran and will read imzML files, process data and create
relevant plot such as spatial distribution of feature counts.

"""

import os
import random
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.stats import sem, ttest_ind
from pyimzml.ImzMLParser import ImzMLParser
from processing import process, aligned_representation
from analysis.chip_types_data_analysis import (
    num_features_df, msi_spatial_num_features, msi_mean_num_features,
    plot_num_features_thresholds
)


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
    metadata_df: pd.DataFrame, save_path: Path,
    mz_range: Tuple[float, float] = (600, 900)
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
    mz_range (Tuple[float,float]): Start and end m/z value to consider. 
        Defaults to (600, 900).

  """
  # Dictionary to store spatial number of features for each sample
  spatial_num_features = {}
  # Loop through each parser and calculate spatial number of features
  for name, p in parsers.items():
    spatial_num_features[name] = msi_spatial_num_features(
        p, masks[name], mz_range=mz_range
    )
  # List to store images for each group
  images = []
  # Group the metadata by file name and process each group
  for _, group in metadata_df.groupby("file_name"):
    # Initialize an empty image array
    img = np.zeros((group["y_max"].max(), group["x_max"].max() + 1))
    # Loop through each row in the group and update the image array
    for _, row in group.iterrows():
      img[:, row["x_min"]:row["x_max"] +
          1] = spatial_num_features[row.sample_file_name]
    # Append the processed image to the images list
    images.append(img)
  # Create a combined image from the first two images in the list
  combined_img = concatenate_images_array(
      np.rot90(images[1], 1), np.rot90(images[0], 1)
  )[:-14, 14:-8]
  # Create a figure for plotting
  fig, ax = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True)
  # Show the combined image
  shw = ax.imshow(
      combined_img, cmap='magma',
      vmax=np.percentile(combined_img[~np.isnan(combined_img)], 99)
  )
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
      np.percentile(combined_img[~np.isnan(combined_img)], 99),
      combined_img[~np.isnan(combined_img)].min(), 6
  )
  bar.set_ticks(ticks)
  bar.set_ticklabels(['{:.0f}'.format(t) for t in ticks])
  # Remove the axis for a cleaner look
  plt.axis("off")
  plt.tight_layout()
  # Save the plot
  plt.savefig(
      save_path / "maps_of_the_number_of_features.png", bbox_inches='tight',
      dpi=1200, transparent=True
  )
  # Display the plot
  plt.show()


def plot_num_features(
    parsers: Dict[str, ImzMLParser], masks: Dict[str, np.ndarray],
    metadata_df: pd.DataFrame, save_path: Path,
    mz_range: Tuple[float, float] = (600, 900)
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
    mz_range (Tuple[float,float]): Start and end m/z value to consider. 
        Defaults to (600, 900).

  """
  # Create dict to store num features
  num_features = {}
  # Group the metadata by file name and process each group
  for group_name, group in metadata_df.groupby("file_name"):
    # Loop through each row in the group
    for index, row in group.iterrows():
      # Get MSO num features
      msi_features = msi_mean_num_features(
          parsers[row.sample_file_name], masks[row.sample_file_name],
          mz_range=mz_range
      )
      # Update group number of features list
      num_features[group_name] = num_features.get(group_name,
                                                  []) + [msi_features]
  # Get values for bar plot
  means = {key: np.mean(value) for key, value in num_features.items()}
  sems = {key: sem(value) for key, value in num_features.items()}
  mean_values = list(means.values())[::-1]
  sem_values = list(sems.values())[::-1]
  # Extract the points for plotting on the bar chart
  std_points = num_features['220224-optimization-liver-standard-1 Analyte 1_1']
  opt_points = num_features['220224-optimization-liver-optimised-1 Analyte 1_1']
  # Get ttest p value
  _, pvalue = ttest_ind(std_points, opt_points)
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
  _, ax = plt.subplots(1, 1, figsize=(3, 6), tight_layout=True)
  ax.bar(
      categories, mean_values, yerr=sem_values, color=['tab:red', 'tab:blue'],
      capsize=10, error_kw=dict(ecolor='0.2', lw=2.5, capsize=10, capthick=2.5)
  )
  # Add individual points to the bars
  ax.plot(
      np.repeat(0, len(std_points)), std_points, 'o', markersize=7, color='0.2'
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
  # Add a line to show the p-value
  y, h = max(mean_values) + max(sem_values) + 60, 5
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
      save_path / "number_of_features_bar_graph.png", bbox_inches='tight',
      dpi=1200, transparent=True
  )
  # Display the plot
  plt.show()

  # Save number of features csv
  pd.DataFrame(num_features
              ).to_csv(save_path / "number_of_features.csv", index=False)


def main():
  """Function containing main code"""
  # Define current folder using this file
  CWD = Path(os.path.dirname(os.path.abspath(__file__)))
  # Define folder that contains dataset
  LIVER_PATH = CWD / ".." / ".." / "data" / "LIVER"
  # Define folder that contains raw data
  RAW_DATA = LIVER_PATH / "raw"
  # Define folder to save aligned data
  ALIGNED_DATA = LIVER_PATH / "aligned"
  # Define folder to save processed data
  PROCESSED_DATA = LIVER_PATH / "processed"
  # Define file that contains metadata
  METADATA_PATH = LIVER_PATH / "metadata.csv"
  # Define mass range start value
  MZ_START = 50
  # Define mass range end value
  MZ_END = 1200
  # Define mass resolution of the data
  MASS_RESOLUTION = 0.025
  # Define lock mass reference peak
  LOCK_MASS_PEAK = 885.5498
  # Define lock mass tol
  LOCK_MASK_TOL = 0.3
  # Define mz where lipid start
  MZ_LIPID_START = 600
  # Define mz where lipid end
  MZ_LIPID_END = 900
  # Define representative peaks
  representative_peaks_map = {
      'std': [634.40, 794.54, 886.55], 'opt': [794.5, 834.5, 886.6]
  }
  # Define random seed
  SEED = 42
  random.seed(SEED)
  np.random.seed(SEED)
  # Read metadata csv
  metadata_df = pd.read_csv(METADATA_PATH)

  # Loop over each unique msi imzML file
  for file_name in metadata_df.file_name.unique():
    # Define path to msi imzML file
    msi_path = os.path.join(RAW_DATA, f"{file_name}.imzML")
    # Define path to new msi imzML file after alignment
    output_path = os.path.join(ALIGNED_DATA, f"{file_name}.imzML")
    # Align MSI
    aligned_representation(msi_path, output_path, LOCK_MASS_PEAK, LOCK_MASK_TOL)

  # Loop over each ROI in data frame
  for _, roi in metadata_df.iterrows():
    #
    representative_peaks = next(
        (
            value for key, value in representative_peaks_map.items()
            if key in roi.sample_file_name
        ), None
    )
    # Define path to msi imzML file
    msi_path = os.path.join(ALIGNED_DATA, f"{roi.file_name}.imzML")
    # Define path to new msi imzML file after processing
    output_path = os.path.join(PROCESSED_DATA, f"{roi.sample_file_name}")
    # Create output folder if doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # Process msi
    process(
        msi_path, output_path, roi.x_min, roi.x_max, roi.y_min, roi.y_max,
        MZ_START, MZ_END, MASS_RESOLUTION, representative_peaks
    )

  # Define path to save figures
  PLOT_PATH = Path(CWD / "liver/")
  # Create dirs
  PLOT_PATH.mkdir(parents=True, exist_ok=True)
  # Create dict of msi parsers and masks
  parsers = {}
  masks = {}
  for folder in PROCESSED_DATA.iterdir():
    name = folder.stem
    parsers[name] = ImzMLParser(folder / "meaningful_signal.imzML")
    masks[name] = np.load(folder / "segmentation.npy", mmap_mode='r')
  # Plot figures
  plot_spatial_num_features(
      parsers, masks, metadata_df, PLOT_PATH, (MZ_LIPID_START, MZ_LIPID_END)
  )
  plot_num_features(
      parsers, masks, metadata_df, PLOT_PATH, (MZ_LIPID_START, MZ_LIPID_END)
  )
  num_features = num_features_df(parsers, masks, np.arange(1.0, 3.1, 0.2))
  num_features.to_csv(PLOT_PATH / "num_features.csv", index=False)
  plot_num_features_thresholds(num_features, PLOT_PATH)


if __name__ == '__main__':
  main()
