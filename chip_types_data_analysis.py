"""Chip types data analysis
The script should be ran and will read imzML files, process data and create
relevant plot such as feature counts per Chip types.

"""
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from pyimzml.ImzMLParser import ImzMLParser
from processing import process
from correlation import correlation_analysis
from utils import read_msi
from liver_data_analysis import msi_mean_num_features


def plot_msi_segmentation(
    metadata_df: pd.DataFrame, masks, save_path: Path
) -> None:
  """Plots and saves an image that represents the segmentation mask of each msi.

  Args:
    metadata (pd.DataFrame): Data frame of metadata.
    masks (Dict[str, np.ndarray]): A dictionary containing masks for each
        sample.
    save_path (Path): Path where the correlation plot will be saved.
  
  """
  # Determine the layout of the subplot grid based on the data
  grouped = metadata_df.groupby("sample_number")
  # Determine the layout of the subplot grid based on the data
  max_rows = grouped.size().max()
  num_groups = len(grouped)
  # Creating a grid of subplots
  _, axs = plt.subplots(
      max_rows, num_groups, figsize=(num_groups * 2, max_rows * 2)
  )
  # Ensuring axs is always a 2D array for consistency in indexing
  if num_groups == 1:
    axs = np.expand_dims(axs, axis=-1)
  if max_rows == 1:
    axs = np.expand_dims(axs, axis=0)
  # Plotting each mask image in the appropriate subplot
  for group_index, (_, group) in enumerate(grouped):
    for image_index, (_, row) in enumerate(group.iterrows()):
      # Determine the subplot index
      subplot_index = image_index if group_index == 0 else image_index + 1
      # Access the subplot axis
      ax = axs[subplot_index, group_index]
      try:
        # Get the mask image
        img = masks[row.sample_file_name]
        # Display the image
        ax.imshow(img, cmap="gray_r")
        # Customizing the ticks for better readability
        ax.set_xticks([0, img.shape[1] - 1])
        ax.set_yticks([0, img.shape[0] - 1])
        ax.tick_params(axis='both', length=0)
        # Customizing tick labels
        for yticklabel in ax.get_yticklabels():
          yticklabel.set_fontweight('bold')
          yticklabel.set_fontsize(14)
          yticklabel.set_color('0.2')
        for xticklabel in ax.get_xticklabels():
          xticklabel.set_fontweight('bold')
          xticklabel.set_fontsize(14)
          xticklabel.set_color('0.2')
        # Customizing the spines for aesthetics
        for axis in ['bottom', 'left']:
          ax.spines[axis].set_linewidth(2.5)
          ax.spines[axis].set_color('0.2')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
      except FileNotFoundError:
        # Hide the axis if the file is not found and print a warning
        ax.axis('off')
        print(f"File not found: {row['ImagePath']}")
    # Turning off unused subplots to maintain a clean grid
    for ax in axs[len(group) + (0 if group_index == 0 else 1):, group_index]:
      ax.axis('off')
    # Specifically turning off the skipped subplot for non-first groups
    if group_index != 0:
      axs[0, group_index].axis('off')
  # Add space between subplots
  plt.tight_layout()
  # Save the plot
  plt.savefig(
      save_path.joinpath("segmentations.png"), bbox_inches='tight', dpi=1200,
      transparent=True
  )
  # Show the plot
  plt.show()


def plot_spectras_corr(processed_path: str, output_path: str) -> None:
  """Plots a correlation matrix for the chip types and saves the figure.

  Args:
      processed_path (str): Path to processed continuos imzML files. 
      output_path (str): Path to save correlation matrices.
  """
  # Apply correlation analysis
  correlation_analysis(processed_path, output_path)
  # Load the correlation matrix from the file
  corr_matrix = pd.read_csv(
      output_path / 'meaningful-signal_tissue_tissue.csv', index_col=0
  )
  # Filter for 'replica' (-r) and 'sections' (-s) spectras
  replica_columns = [col for col in corr_matrix.columns if col.endswith('-r')]
  section_rows = corr_matrix.index[corr_matrix.index.str.endswith('-s')]
  # Create a new correlation matrix with the specified conditions
  filtered_corr_matrix = corr_matrix.loc[section_rows, replica_columns]
  # Generate the heatmap
  _, ax = plt.subplots(1, 1, figsize=(15, 5), tight_layout=True)
  # Plot correlation matrix
  ax = sns.heatmap(
      filtered_corr_matrix, annot=True, cbar=False, cmap="YlGn", fmt=".2f",
      vmin=-1, vmax=1, linewidth=2, linecolor='w', square=True, ax=ax,
      annot_kws={
          "fontsize": 14,
          "fontweight": 'bold',
          'color': '0.2'
      }
  )
  # Rename x and y ticks labels
  x_ticks_labels = [
      label.replace('_', '\n').replace("-r", "") for label in replica_columns
  ]
  y_ticks_labels = []
  for label in section_rows:
    tick_label = label[0]
    try:
      tick_label += "_" + str(int(label.replace("-s", "")[-1]))
    except:
      pass
    y_ticks_labels.append(tick_label)
  # Set x and y ticks labels
  ax.set_xticklabels(x_ticks_labels, rotation=0, ha='center')
  ax.set_yticklabels(y_ticks_labels, va='center')
  # Customize the ticks
  for yticklabel in ax.get_yticklabels():
    yticklabel
    yticklabel.set_fontweight('bold')
    yticklabel.set_fontsize(14)
    yticklabel.set_color('0.2')
  for xticklabel in ax.get_xticklabels():
    xticklabel.set_fontweight('bold')
    xticklabel.set_fontsize(14)
    xticklabel.set_color('0.2')
  # Remove ticks line
  ax.tick_params(axis='both', length=0)
  # Customize the axis labels
  ax.set_ylabel("Tissue", fontsize=14, weight='bold', color='0.2')
  ax.set_xlabel("Chips", fontsize=14, weight='bold', color='0.2')
  # Add space between subplots
  plt.tight_layout()
  # Save the plot
  plt.savefig(
      output_path.joinpath("correlations.png"), bbox_inches='tight', dpi=1200,
      transparent=True
  )
  # Show plot
  plt.show()


def get_sample_type(sample_file_name) -> str:
  """Retrieves the sample type given its sample_file_name

  Args:
      sample_file_name (str): sample file name from metadata.

  Returns:
      str: sample type.
  
  """
  vals = sample_file_name.split("_")[1:]
  vals[-1] = vals[-1].split("-")[0]
  try:
    int(vals[-1])
    vals.pop()
  except:
    pass
  return "_".join(vals)


def num_features_df(
    parsers: Dict[str, ImzMLParser], masks: Dict[str, np.ndarray],
    percentages: np.ndarray
) -> pd.DataFrame:
  """Creates a dataframe containing the number of features in each percentage
  threshold for each parser.

  Args:
    parser (Dict[str, ImzMLParser]): A dictionary containing ImzMLParser 
        objects with keys as sample names.
    masks (Dict[str, np.ndarray]): A dictionary containing masks for each
        sample.
    save_path (np.ndarray): Array of percentages to calculate features.
  
  """
  # Dictionary to store number of features for each sample
  num_features = {}
  # Loop through each parser and calculate number of features
  for name, p in parsers.items():
    name_num_features = []
    for percentage in percentages:
      name_num_features.append(
          msi_mean_num_features(p, masks[name], percentage)
      )
    num_features[name] = name_num_features
  # Creating a DataFrame from the dictionary
  df = pd.DataFrame.from_dict(num_features)
  # Transposing the DataFrame and setting the columns
  df_transposed = df.T
  df_transposed.columns = percentages
  # Resetting the index to make the original column names a regular column
  df_reset = df_transposed.reset_index()
  # Applying the function to extract the type
  types = [get_sample_type(name) for name in df.columns]
  df_reset['Type'] = types
  # Melting the DataFrame for a suitable format
  df_melted = df_reset.melt(
      id_vars='Type', var_name='Peak Percentage',
      value_name='Number of Features'
  )
  # Removing non numeric values
  df_melted['Peak Percentage'] = pd.to_numeric(
      df_melted['Peak Percentage'], errors='coerce'
  )
  df_melted = df_melted.dropna(subset=['Peak Percentage'])
  return df_melted


def plot_num_features_thresholds(num_features: pd.DataFrame, save_path: Path):
  """Plots and saves a line plot comparing the number of features between chip
  types in each percentage threshold.

  Args:
    num_features_df (pd.DataFrame): dataframe containing the number of features
        in each percentage threshold for each parser.
    save_path (Path): Path object where the output image will be saved.
  
  """
  # Creating the line plot with the corrected x-ticks
  plt.figure(figsize=(7, 7))
  ax = sns.lineplot(
      data=num_features, x='Peak Percentage', y='Number of Features',
      hue='Type', linewidth=2
  )
  plt.xticks(sorted(num_features['Peak Percentage'].unique()))

  # Customize the spines
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2.5)
    ax.spines[axis].set_color('0.2')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  # Customize the ticks
  for yticklabel in ax.get_yticklabels():
    yticklabel
    yticklabel.set_fontweight('bold')
    yticklabel.set_fontsize(14)
    yticklabel.set_color('0.2')
  for xticklabel in ax.get_xticklabels():
    xticklabel.set_fontweight('bold')
    xticklabel.set_fontsize(14)
    xticklabel.set_color('0.2')
    xticklabel.set_rotation(90)
  # Remove ticks line
  ax.tick_params(axis='both', length=0)
  # Customize the axis labels
  ax.set_ylabel("Number of Features", fontsize=14, weight='bold', color='0.2')
  ax.set_xlabel("Peak Percentage", fontsize=14, weight='bold', color='0.2')

  # Creating legend and Customizing it
  leg = plt.legend(loc='upper right', prop={"size": 14})
  leg.set_frame_on(False)
  for line in leg.get_lines():
    line.set_linewidth(3.5)
  for text in leg.get_texts():
    text.set_text(" ".join(text.get_text().split("_")).capitalize())
    text.set_weight('bold')
    text.set_color('0.2')
  # Add space between subplots
  plt.tight_layout()
  # Save the plot
  plt.savefig(
      save_path.joinpath("num_features_by_threshold.png"), bbox_inches='tight',
      dpi=1200, transparent=True
  )
  # Show plot
  plt.show()


def plot_num_features(
    num_features: pd.DataFrame, percentage: float, save_path: Path
):
  """Plots and saves a bar chart comparing the number of features between chip
  types in a certain percentage.

  Args:
    num_features_df (pd.DataFrame): dataframe containing the number of features
        in each percentage threshold for each parser.
    percentage (float): percentage to get number of features. (should be in the
        already calculated percentages)
    save_path (Path): Path object where the output image will be saved.
  """
  #
  num_features = num_features[num_features["Peak Percentage"] == percentage]

  # Now we plot the bars for the mean and add error bars for the SEM.
  _, ax = plt.subplots(1, 1, figsize=(10, 12))

  # Create a bar plot using seaborn with custom error bars
  ax = sns.barplot(
      data=num_features, x="Type", y="Number of Features", errorbar=lambda x:
      (x.mean() - x.sem(), x.mean() + x.sem()), capsize=0.15
  )
  sns.scatterplot(
      data=num_features, x="Type", y="Number of Features", legend=False,
      zorder=10, color='0.2', edgecolor='0.2', marker='s', s=70
  )

  # Customize the spines
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2.5)
    ax.spines[axis].set_color('0.2')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  # Customize ticks
  plt.xticks(fontsize=14, fontweight='bold', color='0.2')
  plt.xticks(rotation=45)
  plt.yticks(fontsize=14, fontweight='bold', color='0.2')
  ax.set_xticklabels([])

  # Add legend to the plot with a title
  legend_patches = [
      Patch(color=color, label=label) for label, color in
      zip(num_features['Type'].unique(), sns.color_palette())
  ]
  leg = ax.legend(
      loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fancybox=True,
      shadow=True, handles=legend_patches, prop={"size": 14}
  )
  leg.set_frame_on(False)
  for line in leg.get_lines():
    line.set_linewidth(3.5)
  for text in leg.get_texts():
    text.set_text(" ".join(text.get_text().split("_")).capitalize())
    text.set_weight('bold')
    text.set_color('0.2')

  # Remove ticks line
  ax.tick_params(axis='both', length=0)

  # Set labels font size to 14
  ax.set_ylabel('Number of Features', fontsize=14, weight='bold', color='0.2')

  # Improve Aesthetics
  plt.xlabel('')
  plt.ylabel('Number of Features')

  # Add space between subplots and show the plot
  plt.tight_layout()
  # Save the plot
  plt.savefig(
      save_path.joinpath("number_of_features_bar_graph.png"),
      bbox_inches='tight', dpi=1200, transparent=True
  )
  # Display the plot
  plt.show()


def plot_area_ratio(masks: Dict[str, np.ndarray], save_path: Path):
  """Plots and saves a bar chart comparing the ratio area of chip
  types and tissue sections.

  Args:
    masks (Dict[str, np.ndarray]): A dictionary containing masks for each
    sample.
    save_path (Path): Path object where the output image will be saved.
  """
  #
  areas = []
  for mask_name_1, mask_1 in masks.items():
    for mask_name_2, mask_2 in masks.items():
      mask_1_type = get_sample_type(mask_name_1)
      mask_2_type = get_sample_type(mask_name_2)
      if (mask_name_1[0]
          == mask_name_2[0]) and (mask_1_type != "tissue_section"
                                 ) and (mask_2_type == "tissue_section"):
        areas.append([mask_1_type, mask_1.sum() / mask_2.sum()])
  areas = pd.DataFrame(areas, columns=["Type", "Area Ratio"])

  # Now we plot the bars for the mean and add error bars for the SEM.
  _, ax = plt.subplots(1, 1, figsize=(10, 10))

  # Create a bar plot using seaborn with custom error bars
  ax = sns.barplot(
      data=areas, x="Type", y="Area Ratio", errorbar=lambda x:
      (x.mean() - x.sem(), x.mean() + x.sem()), capsize=0.15
  )
  sns.scatterplot(
      data=areas, x="Type", y="Area Ratio", legend=False, zorder=10,
      color='0.2', edgecolor='0.2', marker='s', s=70
  )

  # Customize the spines
  for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2.5)
    ax.spines[axis].set_color('0.2')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  # Customize ticks
  plt.xticks(fontsize=14, fontweight='bold', color='0.2')
  plt.xticks(rotation=45)
  plt.yticks(fontsize=14, fontweight='bold', color='0.2')
  ax.set_xticklabels([])

  # Add legend to the plot with a title
  legend_patches = [
      Patch(color=color, label=label)
      for label, color in zip(areas['Type'].unique(), sns.color_palette())
  ]
  leg = ax.legend(
      loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fancybox=True,
      shadow=True, handles=legend_patches, prop={"size": 14}
  )
  leg.set_frame_on(False)
  for line in leg.get_lines():
    line.set_linewidth(3.5)
  for text in leg.get_texts():
    text.set_text(" ".join(text.get_text().split("_")).capitalize())
    text.set_weight('bold')
    text.set_color('0.2')

  # Remove ticks line
  ax.tick_params(axis='both', length=0)

  # Set labels font size to 14
  ax.set_ylabel('Area Ratio', fontsize=14, weight='bold', color='0.2')

  # Improve Aesthetics
  plt.xlabel('')
  plt.ylabel('Replica Area / Tissue Area')

  # Add space between subplots and show the plot
  plt.tight_layout()
  # Save the plot
  plt.savefig(
      save_path.joinpath("area_ratio_bar_graph.png"), bbox_inches='tight',
      dpi=1200, transparent=True
  )
  # Display the plot
  plt.show()


def main():
  """Function containing main code"""
  # Define current folder using this file
  CWD = Path(os.path.dirname(os.path.abspath(__file__)))
  # Define folder that contains the revision chip type dataset
  CHIP_TYPES_PATH = Path(os.path.join(CWD, "..", "data", "CHIP_TYPES"))
  # Define folder that contains raw data
  RAW_DATA = CHIP_TYPES_PATH.joinpath("raw")
  # Define folder to save processed data
  PROCESSED_DATA = CHIP_TYPES_PATH.joinpath("processed")
  # Define file that contains dhg metadata
  METADATA_PATH = CHIP_TYPES_PATH.joinpath("metadata.csv")
  # Define mass range start value
  MZ_START = 50
  # Define mass range end value
  MZ_END = 1200
  # Define mass resolution of the data
  MASS_RESOLUTION = 0.025
  # Define representative peaks
  REPRESENTATIVE_PEAKS = [
      611.5, 682.58, 736.64, 844.64, 860.63, 888.62, 600.49, 834.53
  ]
  # Define random seed
  SEED = 42
  random.seed(SEED)
  np.random.seed(SEED)
  # Read metadata csv
  metadata_df = pd.read_csv(METADATA_PATH)
  metadata_df["sample_number"] = metadata_df.sample_file_name.apply(
      lambda s: s.split("_")[0]
  )
  # Loop over each ROI in data frame
  for _, roi in metadata_df.iterrows():
    # Define path to msi imzML file
    msi_path = os.path.join(RAW_DATA, f"{roi.file_name}.imzML")
    # Define path to new msi imzML file after processing
    output_path = os.path.join(PROCESSED_DATA, f"{roi.sample_file_name}")
    # Create output folder if doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # Process msi
    process(
        msi_path, output_path, roi.x_min, roi.x_max, roi.y_min, roi.y_max,
        MZ_START, MZ_END, MASS_RESOLUTION, REPRESENTATIVE_PEAKS
    )
  # Define path to save figures
  PLOT_PATH = Path(CWD / "chip_types/")
  # Create dirs
  PLOT_PATH.mkdir(parents=True, exist_ok=True)
  # Create dict of msi parsers and masks
  parsers = {}
  masks = {}
  for folder in PROCESSED_DATA.iterdir():
    name = folder.stem
    parsers[name] = ImzMLParser(folder.joinpath("meaningful_signal.imzML"))
    masks[name] = np.load(folder.joinpath("segmentation.npy"), mmap_mode='r')
  # Plot figures
  plot_msi_segmentation(metadata_df, masks, PLOT_PATH)
  plot_spectras_corr(PROCESSED_DATA, PLOT_PATH)
  num_features = num_features_df(parsers, masks, np.arange(0.05, 0.55, 0.05))
  plot_num_features_thresholds(num_features, PLOT_PATH)
  plot_num_features(num_features, 0.3, PLOT_PATH)
  plot_area_ratio(masks, PLOT_PATH)


if __name__ == '__main__':
  main()
