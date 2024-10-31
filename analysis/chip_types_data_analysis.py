"""Chip types data analysis
The script should be ran and will read imzML files, process data and create
relevant plot such as feature counts per Chip types.

"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, Tuple
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from pyimzml.ImzMLParser import ImzMLParser
from scipy.signal import find_peaks
from processing import process, aligned_representation
from correlation import correlation_analysis
from utils import read_msi
from analysis.esi_data_analysis import (
    export_spectras_metrics, plot_spectras_best_fit_hex,
    plot_spectras_best_fit_scatter
)


def spectra_significant_features_threshold(
    spectra: np.ndarray, percentage=0.3
) -> int:
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


def spectra_significant_features_z_score(
    spectra: np.ndarray, threshold: float = 2.0
):
  """
  Counts the number of significant values in a spectra based on a z-score 
  threshold.

  Args:
    spectra (np.ndarray): The array of spectra intensities.
    threshold (float, optional): The threshold for determining significance. 
        Defaults to 2 for a 95% confidence level.
  
  Returns:
    int: The number of significant spectra features.     
  
  """
  return np.sum(spectra > threshold)


def msi_sum_spectra_num_features(
    p: ImzMLParser, mask: np.ndarray, threshold=2,
    mz_range: Tuple[float, float] = (600, 900)
) -> int:
  """
  Calculates the number of significant spectra features from the sum of 
  spectra in a masked area.

  Args:
    p (ImzMLParser): The ImzML parser object.
    mask (np.ndarray): A binary mask to select specific regions in the MSI data.
    threshold (float, optional): The threshold for determining significance. 
        Defaults to 2 for a 95% confidence level.
    mz_range (Tuple[float,float]): Start and end m/z value to consider. 
        Defaults to (600, 900).

  Returns:
    int: The number of significant spectra features.

  """
  # Read MSI data and get the intensity matrix
  mzs, msi = read_msi(p)
  # Apply the mask to the MSI data
  spectras = msi[mask]
  # Sum the spectra within the masked region
  sum_spectra = np.sum(spectras, axis=0)
  # Calculate and return the number of significant spectra features
  return spectra_significant_features_z_score(
      sum_spectra[(mzs >= mz_range[0]) & (mzs <= mz_range[1])], threshold
  )


def msi_mean_spectra_num_features(
    p: ImzMLParser, mask: np.ndarray, threshold=2,
    mz_range: Tuple[float, float] = (600, 900)
) -> int:
  """
  Calculates the number of significant spectra features from the mean of 
  spectra in a masked area.

  Args:
    p (ImzMLParser): The ImzML parser object.
    mask (np.ndarray): A binary mask to select specific regions in the MSI data.
    threshold (float, optional): The threshold for determining significance. 
        Defaults to 2 for a 95% confidence level.
    mz_range (Tuple[float,float]): Start and end m/z value to consider. 
        Defaults to (600, 900).

  Returns:
    int: The number of significant spectra features.

  """
  # Read MSI data and get the intensity matrix
  mzs, msi = read_msi(p)
  # Apply the mask to the MSI data
  spectras = msi[mask]
  # Calculate the mean of the spectra within the masked region
  mean_spectra = np.mean(spectras, axis=0)
  # Calculate and return the number of significant spectra features
  return spectra_significant_features_z_score(
      mean_spectra[(mzs >= mz_range[0]) & (mzs <= mz_range[1])], threshold
  )


def msi_mean_num_features(
    p: ImzMLParser, mask: np.ndarray, threshold=2,
    mz_range: Tuple[float, float] = (600, 900)
) -> int:
  """
  Calculates the mean number of spectra features across all spectra in a 
  masked area.

  Args:
    p (ImzMLParser): The ImzML parser object.
    mask (np.ndarray): A binary mask to select specific regions in the MSI data.
    threshold (float, optional): The threshold for determining significance. 
        Defaults to 2 for a 95% confidence level.
    mz_range (Tuple[float,float]): Start and end m/z value to consider. 
        Defaults to (600, 900).

  Returns:
    int: The mean number of significant spectra features.

  """
  # Read MSI data and get the intensity matrix
  mzs, msi = read_msi(p)
  # Apply the mask to the MSI data
  spectras = msi[mask]
  # List to store the number of features for each spectrum
  msi_num_features = []
  # Iterate over each spectrum and calculate its number of features
  for spectra in spectras:
    msi_num_features.append(
        spectra_significant_features_z_score(
            spectra[(mzs >= mz_range[0]) & (mzs <= mz_range[1])], threshold
        )
    )
  # Calculate and return the mean number of features
  return np.mean(msi_num_features)


def msi_spatial_num_features(
    p: ImzMLParser, mask: np.ndarray, threshold=2,
    mz_range: Tuple[float, float] = (600, 900)
) -> np.ndarray:
  """
  Calculates the spatial distribution of spectra features in a masked area of
  MSI data.

  Args:
    p (ImzMLParser): The ImzML parser object.
    mask (np.ndarray): A binary mask to select specific regions in the MSI data.
    threshold (float, optional): The threshold for determining significance. 
        Defaults to 2 for a 95% confidence level.
    mz_range (Tuple[float,float]): Start and end m/z value to consider. 
        Defaults to (600, 900).

  Returns:
    np.ndarray: A 2D array representing the number of spectra features at each 
        position.

  """
  # Read MSI data and get the intensity matrix
  mzs, msi = read_msi(p)
  # Initialize an array to store the spatial number of features
  msi_spatial_num_features = np.zeros(mask.shape)
  # Loop over each pixel in the mask
  for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
      if mask[i, j]:
        # Calculate the number of features for the current pixel
        msi_spatial_num_features[i, j] = spectra_significant_features_z_score(
            msi[i, j, (mzs >= mz_range[0]) & (mzs <= mz_range[1])], threshold
        )
      else:
        # Set the value to NaN if the pixel is not in the mask
        msi_spatial_num_features[i, j] = np.nan
  return msi_spatial_num_features


def plot_msi_segmentation(
    metadata_df: pd.DataFrame, masks, save_path: Path
) -> None:
  """Plots and saves an image of the segmentation mask for all msi's.

  Args:
    metadata (pd.DataFrame): Data frame of metadata.
    masks (Dict[str, np.ndarray]): A dictionary containing masks for each
        sample.
    save_path (Path): Path where the plot will be saved.
  
  """
  # Group the metadata by sample number
  grouped = metadata_df.groupby("sample_number")
  # Determine the layout of the subplot grid based on the data
  max_rows = grouped.size().max()
  num_groups = len(grouped)
  # Creating a grid of subplots
  _, axs = plt.subplots(
      num_groups, max_rows, figsize=(max_rows * 2, num_groups * 2)
  )
  # Plotting each mask image in the appropriate subplot
  for group_index, (_, group) in enumerate(grouped):
    for image_index, (_, row) in enumerate(group.iterrows()):
      # Access the subplot axis
      ax = axs[group_index, image_index]
      try:
        # Get the mask image
        img = masks[row.sample_file_name]
        # Display the image
        ax.imshow(img, cmap="gray_r", aspect="auto")
        # Customizing the ticks for better readability
        ax.set_xticks([])
        ax.set_yticks([])
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
        for axis in ['bottom', 'left', 'top', 'right']:
          ax.spines[axis].set_visible(True)
          ax.spines[axis].set_linewidth(2.5)
          ax.spines[axis].set_color('0.2')
        # Add labels to to grid
        if group_index == num_groups - 1:
          ax.set_xlabel(
              "\n".join(get_sample_type(row.sample_file_name).split("_")
                       ).capitalize(), fontsize=14, weight='bold', color='0.2'
          )
        if image_index == 0:
          ax.set_ylabel(
              row.sample_file_name.split("_")[0], fontsize=14, weight='bold',
              color='0.2', rotation=0, labelpad=15
          )
      except FileNotFoundError:
        # Hide the axis if the file is not found and print a warning
        ax.axis('off')
        print(f"File not found: {row['ImagePath']}")
  # Add space between subplots
  plt.tight_layout()
  # Save the plot
  plt.savefig(
      save_path / "segmentations.png", bbox_inches='tight', dpi=1200,
      transparent=True
  )
  # Close the plot
  plt.close()


def plot_spectras_corr(processed_path: str, output_path: str) -> None:
  """Plots and saves a correlation matrix for the chip types.

  Args:
    processed_path (str): Path to processed continuos imzML files. 
    output_path (str): Path to save correlation matrices.

  """
  # Apply correlation analysis
  correlation_analysis(processed_path, output_path, mz_range=(600, 900))
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
      annot_kws={"fontsize": 14, "fontweight": 'bold', 'color': '0.2'}
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
      output_path / "correlations.png", bbox_inches='tight', dpi=1200,
      transparent=True
  )
  # Close plot
  plt.close()


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
    thresholds: np.ndarray
) -> pd.DataFrame:
  """Creates a data frame containing the number of features in each threshold 
      for each parser.

  Args:
    parser (Dict[str, ImzMLParser]): A dictionary containing ImzMLParser 
        objects with keys as sample names.
    masks (Dict[str, np.ndarray]): A dictionary containing masks for each
        sample.
    thresholds (np.ndarray): Array thresholds for determining significance.
  
  """
  # Dictionary to store number of features for each sample
  num_features = {}
  # Loop through each parser and calculate number of features
  for name, p in parsers.items():
    name_num_features = []
    for threshold in thresholds:
      name_num_features.append(msi_mean_num_features(p, masks[name], threshold))
    num_features[name] = name_num_features
  # Creating a DataFrame from the dictionary
  df = pd.DataFrame.from_dict(num_features)
  # Transposing the DataFrame and setting the columns
  df_transposed = df.T
  df_transposed.columns = thresholds
  # Resetting the index to make the original column names a regular column
  df_reset = df_transposed.reset_index()
  # Applying the function to extract the type
  types = [get_sample_type(name) for name in df.columns]
  df_reset['Type'] = types
  # Melting the DataFrame for a suitable format
  df_melted = df_reset.melt(
      id_vars='Type', var_name='Peak Threshold', value_name='Number of Features'
  )
  # Removing non numeric values
  df_melted['Peak Threshold'] = pd.to_numeric(
      df_melted['Peak Threshold'], errors='coerce'
  )
  df_melted = df_melted.dropna(subset=['Peak Threshold'])
  return df_melted


def plot_num_features_thresholds(num_features: pd.DataFrame, save_path: Path):
  """Plots and saves a line plot comparing the number of features between chip
  types in each threshold.

  Args:
    num_features_df (pd.DataFrame): dataframe containing the number of features
        in each threshold for each parser.
    save_path (Path): Path object where the output image will be saved.
  
  """
  # Creating the line plot with the corrected x-ticks
  plt.figure(figsize=(8, 8))
  ax = sns.lineplot(
      data=num_features, x='Peak Threshold', y='Number of Features', hue='Type',
      linewidth=2
  )
  plt.xticks(sorted(num_features['Peak Threshold'].unique()))

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
    yticklabel.set_fontsize(10)
    yticklabel.set_color('0.2')
  for xticklabel in ax.get_xticklabels():
    xticklabel.set_fontweight('bold')
    xticklabel.set_fontsize(10)
    xticklabel.set_color('0.2')
    xticklabel.set_rotation(90)
  # Remove ticks line
  ax.tick_params(axis='both', length=0)
  # Customize the axis labels
  ax.set_ylabel("Number of Features", fontsize=14, weight='bold', color='0.2')
  ax.set_xlabel("Threshold (Z-score)", fontsize=14, weight='bold', color='0.2')

  # Creating legend and Customizing it
  leg = plt.legend(
      loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fancybox=True,
      shadow=True, prop={"size": 14}
  )

  leg.set_frame_on(False)
  # Sort legend
  for line in leg.get_lines():
    line.set_linewidth(3.5)
  for text in leg.get_texts():
    text_parts = text.get_text().split("_")
    if len(text_parts) > 2:
      parts_new = []
      for i in range(0, len(text_parts), 2):
        parts_new.append(" ".join(text_parts[i:i + 2]))
      text.set_text("\n".join(parts_new).capitalize())
    else:
      text.set_text(" ".join(text_parts).capitalize())
    text.set_weight('bold')
    text.set_color('0.2')
  # Add space between subplots
  plt.tight_layout()
  # Save the plot
  plt.savefig(
      save_path / "num_features_by_threshold.png", bbox_inches='tight',
      dpi=1200, transparent=True
  )
  # Close plot
  plt.close()


def plot_num_features(
    num_features: pd.DataFrame, threshold: float, save_path: Path
):
  """Plots and saves a bar chart comparing the number of features between chip
  types in a certain threshold.

  Args:
    num_features_df (pd.DataFrame): dataframe containing the number of features
        in each threshold for each parser.
    threshold (float): The threshold for determining significance.
    save_path (Path): Path object where the output image will be saved.

  """
  # Ensure "Peak Threshold" column is of type float
  num_features["Peak Threshold"] = num_features["Peak Threshold"].astype(float)
  # Filter the DataFrame based on the threshold with tolerance for floating-point precision
  tolerance = 1e-9
  num_features = num_features[
      np.isclose(num_features["Peak Threshold"], threshold, atol=tolerance)]
  # Plot the bars for the mean and add error bars for the SEM.
  _, ax = plt.subplots(1, 1, figsize=(8, 8))
  # create color palette
  palette_dict = {
      label: color for label, color in
      zip(num_features['Type'].unique(), sns.color_palette())
  }
  print(palette_dict)
  # Create a bar plot using seaborn with custom error bars
  ax = sns.barplot(
      data=num_features, x="Type", y="Number of Features", hue="Type",
      errorbar=lambda x: (x.mean() - x.sem(), x.mean() + x.sem()), capsize=0.15,
      palette=palette_dict
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
  plt.xticks(fontsize=10, fontweight='bold', color='0.2')
  plt.xticks(rotation=45)
  plt.yticks(fontsize=10, fontweight='bold', color='0.2')
  ax.set_xticklabels([])
  # Add legend to the plot with a title
  legend_patches = [
      Patch(color=color, label=label) for label, color in palette_dict.items()
  ]
  leg = ax.legend(
      loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fancybox=True,
      shadow=True, handles=legend_patches, prop={"size": 14}
  )

  leg.set_frame_on(False)
  # Sort legend
  for line in leg.get_lines():
    line.set_linewidth(3.5)
  for text in leg.get_texts():
    text_parts = text.get_text().split("_")
    if len(text_parts) > 2:
      parts_new = []
      for i in range(0, len(text_parts), 2):
        parts_new.append(" ".join(text_parts[i:i + 2]))
      text.set_text("\n".join(parts_new).capitalize())
    else:
      text.set_text(" ".join(text_parts).capitalize())
    text.set_weight('bold')
    text.set_color('0.2')
  # Remove ticks line
  ax.tick_params(axis='both', length=0)
  # Set labels
  ax.set_ylabel('Number of Features', fontsize=16, weight='bold', color='0.2')
  ax.set_xlabel("")
  # Add space between subplots and show the plot
  plt.tight_layout()
  # Save the plot
  plt.savefig(
      save_path / "number_of_features_bar_graph.png", bbox_inches='tight',
      dpi=1200, transparent=True
  )
  # Close the plot
  plt.close()


def plot_area_ratio(masks: Dict[str, np.ndarray], save_path: Path):
  """Plots and saves a bar chart comparing the ratio area of chip
  types and tissue sections.

  Args:
    masks (Dict[str, np.ndarray]): A dictionary containing masks for each
        sample.
    save_path (Path): Path object where the output image will be saved.

  """
  # Define a list to store the area ratios
  areas = []
  # Loop through each mask and calculate the area ratio
  for mask_name_1, mask_1 in masks.items():
    for mask_name_2, mask_2 in masks.items():
      mask_1_type = get_sample_type(mask_name_1)
      mask_2_type = get_sample_type(mask_name_2)
      if (mask_name_1[0]
          == mask_name_2[0]) and (mask_1_type != "tissue_section"
                                 ) and (mask_2_type == "tissue_section"):
        areas.append([mask_1_type, mask_1.sum() / mask_2.sum()])
  # Create a DataFrame from the list
  areas = pd.DataFrame(areas, columns=["Type", "Area Ratio"])
  # Save the DataFrame to a CSV file
  areas.to_csv(save_path / "area_ratio.csv", index=False)

  # Create color palette
  palette_dict = {
      label: color
      for label, color in zip(areas['Type'].unique(), sns.color_palette())
  }

  # Plot the bars for the mean and add error bars for the SEM.
  _, ax = plt.subplots(1, 1, figsize=(8, 8))

  # Create a bar plot using seaborn with custom error bars
  ax = sns.barplot(
      data=areas, x="Type", y="Area Ratio", errorbar=lambda x:
      (x.mean() - x.sem(), x.mean() + x.sem()), capsize=0.15,
      palette=palette_dict
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
  plt.xticks(fontsize=10, fontweight='bold', color='0.2')
  plt.xticks(rotation=45)
  plt.yticks(fontsize=10, fontweight='bold', color='0.2')
  ax.set_xticklabels([])

  # Add legend to the plot with a title
  legend_patches = [
      Patch(color=color, label=label) for label, color in palette_dict.items()
  ]
  leg = ax.legend(
      loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fancybox=True,
      shadow=True, handles=legend_patches, prop={"size": 14}
  )

  leg.set_frame_on(False)
  # Sort legend
  for line in leg.get_lines():
    line.set_linewidth(3.5)
  for text in leg.get_texts():
    text_parts = text.get_text().split("_")
    if len(text_parts) > 2:
      parts_new = []
      for i in range(0, len(text_parts), 2):
        parts_new.append(" ".join(text_parts[i:i + 2]))
      text.set_text("\n".join(parts_new).capitalize())
    else:
      text.set_text(" ".join(text_parts).capitalize())
    text.set_weight('bold')
    text.set_color('0.2')

  # Remove ticks line
  ax.tick_params(axis='both', length=0)

  # Set labels font size to 14
  ax.set_ylabel(
      'Replica Area / Tissue Area', fontsize=16, weight='bold', color='0.2'
  )
  ax.set_xlabel("")

  # Add space between subplots and show the plot
  plt.tight_layout()
  # Save the plot
  plt.savefig(
      save_path / "area_ratio_bar_graph.png", bbox_inches='tight', dpi=1200,
      transparent=True
  )
  # Close the plot
  plt.close()


def get_spectras(
    parsers: Dict[str, ImzMLParser], masks: Dict[str, np.ndarray]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
  """Get the spectras for each parser and mask.

  Args:
    parsers (Dict[str, ImzMLParser]): A dictionary containing parsers.
    masks (Dict[str, np.ndarray]): A dictionary containing masks for each 
        parser.

  Returns:
    Dict[str, Tuple[np.ndarray, np.ndarray]]: A dict containing the spectras 
        for each parser and mask.
  
  """
  # Dictionary to store the spectras
  spectras = {}
  # Loop through each parser and mask
  for name, p in parsers.items():
    # Read MSI data and get the intensity matrix
    mzs, msi = read_msi(p)
    # Apply the mask to the MSI data
    spectras[name] = (
        mzs[(mzs >= 600) & (mzs <= 900)],
        np.mean(msi[masks[name]], axis=0)[(mzs >= 600) & (mzs <= 900)]
    )
  return spectras


def main():
  """Function containing main code"""
  # Define current folder using this file
  CWD = Path(os.path.dirname(os.path.abspath(__file__)))
  # Define folder that contains dataset
  CHIP_TYPES_PATH = CWD / ".." / ".." / "data" / "CHIP_TYPES_DESI"
  # Define folder that contains raw data
  RAW_DATA = CHIP_TYPES_PATH / "raw"
  # Define folder to save aligned data
  ALIGNED_DATA = CHIP_TYPES_PATH / "aligned"
  # Define folder to save processed data
  PROCESSED_DATA = CHIP_TYPES_PATH / "processed"
  # Define file that contains metadata
  METADATA_PATH = CHIP_TYPES_PATH / "metadata.csv"
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
  # Define representative peaks
  representative_peaks_map = {
      'flat_porous_substrate': [861.89, 848.89,
                                862.85], 'porous_nNs_with_porous_substrate':
      [600.51, 768.51, 885.55], 'porous_nNs': [794.5, 834.5, 886.6],
      'solid_nNs': [627.53, 834.56, 886.66], 'tissue': [794.5, 834.5, 886.6]
  }
  representative_peaks_map = {
      'flat_porous_substrate': [834.53, 886.62],
      'porous_nNs_with_porous_substrate': [834.53, 886.62], 'porous_nNs':
      [834.53, 886.62], 'solid_nNs': [834.53,
                                      886.62], 'tissue': [834.53, 886.62]
  }
  # Read metadata csv
  metadata_df = pd.read_csv(METADATA_PATH)
  metadata_df["sample_number"] = metadata_df.sample_file_name.apply(
      lambda s: s.split("_")[0]
  )

  # Loop over each unique msi imzML file
  for file_name in metadata_df.file_name.unique():
    # Define path to msi imzML file
    msi_path = RAW_DATA / f"{file_name}.imzML"
    # Define path to new msi imzML file after alignment
    output_path = ALIGNED_DATA / f"{file_name}.imzML"
    # Create output folder if doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Align MSI
    aligned_representation(msi_path, output_path, LOCK_MASS_PEAK, LOCK_MASK_TOL)

  # Loop over each ROI in data frame
  for _, roi in metadata_df.iterrows():
    # Get representative peaks
    representative_peaks = next(
        (
            value for key, value in representative_peaks_map.items()
            if key in roi.sample_file_name
        ), None
    )
    # Define path to msi imzML file
    msi_path = ALIGNED_DATA / f"{roi.file_name}.imzML"
    # Define path to new msi imzML file after processing
    output_path = PROCESSED_DATA / f"{roi.sample_file_name}"
    # Create output folder if doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    # Process msi
    process(
        msi_path, output_path, roi.x_min, roi.x_max, roi.y_min, roi.y_max,
        MZ_START, MZ_END, MASS_RESOLUTION, representative_peaks
    )

  # Define path to save figures
  PLOT_PATH = CWD / "chip_types"
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
  plot_msi_segmentation(metadata_df, masks, PLOT_PATH)
  plot_spectras_corr(PROCESSED_DATA, PLOT_PATH)
  num_features = num_features_df(parsers, masks, np.arange(1.0, 3.1, 0.2))
  num_features.to_csv(PLOT_PATH / "num_features.csv", index=False)
  plot_num_features_thresholds(num_features.copy(), PLOT_PATH)
  plot_num_features(num_features.copy(), 2.0, PLOT_PATH)
  plot_area_ratio(masks, PLOT_PATH)
  spectras = get_spectras(parsers, masks)
  export_spectras_metrics(spectras, PLOT_PATH)
  plot_spectras_best_fit_hex(spectras, PLOT_PATH, lambda x: x)
  plot_spectras_best_fit_scatter(spectras, PLOT_PATH, lambda x: x)


if __name__ == '__main__':
  main()
