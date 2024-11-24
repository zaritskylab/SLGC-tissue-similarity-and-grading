"""DESI human glioma (DHG) data analysis
The script should be ran and will read text files, process data and create
relevant plot such as correlation heatmap.

"""

# Import the necessary libraries
import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import figure_customizer as fc
from pathlib import Path
from typing import List, Tuple, Dict
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from processing import TICNormalizer, MeanSegmentation
from classification.binary_classification import main as binary_classification_main


def process_spectral_data_to_image(
    txt_file_path: Path
) -> Tuple[np.ndarray, np.ndarray]:
  """Function to process the spectral data in a txt file to a 3D image array.

  Args:
    txt_file_path (Path): The path to the txt file containing the spectral data.

  Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing the 3D image array and 
        the mzs array.

  """
  # Read the file and extract the mzs
  with open(txt_file_path, "r") as f:
    mzs = [float(mz) for mz in f.readlines()[3].strip().split("\t") if mz != ""]
  # Read the file and extract the data
  df = pd.read_csv(txt_file_path, skiprows=4, sep="\t", header=None)
  df = df.iloc[:, 1:-2]
  # Rename the columns
  df.columns = ["x", "y"] + mzs
  # Convert the 'x' and 'y' columns to integers
  df['x'] = (df['x'] * 10).astype(int)
  df['y'] = (df['y'] * 10).astype(int)
  # Shift the 'x' and 'y' coordinates to start from 0
  if df['x'].min() < 0:
    df['x'] = df['x'] + abs(df['x'].min())
  if df['y'].min() < 0:
    df['y'] = df['y'] + abs(df['y'].min())
  # Extract the 'x' and 'y' coordinates, and the spectral data
  x_coords = df['x'].values
  y_coords = df['y'].values
  spectra_data = df.drop(['x', 'y'], axis=1).values
  # Preallocate the 3D image array
  img = np.zeros(
      (y_coords.max() + 1, x_coords.max() + 1, spectra_data.shape[1])
  )
  # Use numpy's advanced indexing to place the spectra_data in the img array
  img[y_coords, x_coords, :] = spectra_data
  # Return the 3D image array and the mzs array
  return img, np.array(mzs)


def merge_mzs_and_intensities(
    mzs: np.ndarray, intensities: np.ndarray, threshold: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
  """Function to merge mzs and intensities of a single image based on a
      threshold.

  Args:
    mzs (np.ndarray): Array of mzs.
    intensities (np.ndarray): Image of intensities corresponding to the mzs.
    threshold (float, optional): The threshold to merge mzs. Defaults to 0.02.

  Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing the merged mzs and 
      intensities.

  """
  # Sort mzs and intensities based on mzs order to simplify the process
  sorted_indices = np.argsort(mzs)
  mzs_sorted = mzs[sorted_indices]
  intensities_sorted = intensities[:, :, sorted_indices]
  # Initialize the merged mzs and intensities
  merged_mzs = []
  merged_intensities = []
  # Iterate over the mzs
  i = 0
  while i < len(mzs_sorted):
    # Check if the next mz is within the threshold
    if i < len(mzs_sorted
              ) - 1 and abs(mzs_sorted[i] - mzs_sorted[i + 1]) < threshold:
      # Merge the current and next mz
      new_mz = (mzs_sorted[i] + mzs_sorted[i + 1]) / 2
      new_intensity = intensities_sorted[:, :, i] + intensities_sorted[:, :,
                                                                       i + 1]
      merged_mzs.append(new_mz)
      merged_intensities.append(new_intensity)
      # Skip the next mz as it has been merged
      i += 2
    else:
      # If no merge, keep the current mz and intensity
      merged_mzs.append(mzs_sorted[i])
      merged_intensities.append(intensities_sorted[:, :, i])
      i += 1
  # Convert back to numpy arrays
  merged_mzs = np.array(merged_mzs)
  merged_intensities = np.stack(merged_intensities, axis=2)
  # Return the merged mzs and intensities
  return merged_mzs, merged_intensities


def pair_corr(
    mean_1: np.ndarray, mean_2: np.ndarray, mzs_1: np.ndarray,
    mzs_2: np.ndarray, tolerance: float = 0.02
) -> float:
  """Function to compute the correlation between two mean spectra.

  Args:
    mean_1 (np.ndarray): The mean spectrum of the first image.
    mean_2 (np.ndarray): The mean spectrum of the second image.
    mzs_1 (np.ndarray): The mzs of the first image.
    mzs_2 (np.ndarray): The mzs of the second image.
    tolerance (float, optional): The tolerance to match mzs. Defaults to 0.02.

  Returns:
    float: The correlation between the two mean spectra.

  """
  # Create arrays for new means and matched m/z values
  new_mean_1 = []
  new_mean_2 = []
  # Compute the absolute difference matrix between mzs_1 and mzs_2
  diff_matrix = np.abs(mzs_1[:, np.newaxis] - mzs_2)
  # Find matches within the tolerance
  matches = diff_matrix < tolerance
  # Keep track of matched indices in mzs_2
  matched_indices_2 = set()
  # First pass: match mzs_1 to mzs_2
  for i, _ in enumerate(mzs_1):
    # Check if there is a match
    match_idx = np.where(matches[i])[0]
    if match_idx.size > 0:
      # We know the first match will be the only one due to larger gap > 0.02
      closest_idx = match_idx[0]
      new_mean_1.append(mean_1[i])
      new_mean_2.append(mean_2[closest_idx])
      matched_indices_2.add(closest_idx)
    else:
      # No match for mzs_1, append zero for mean_2
      new_mean_1.append(mean_1[i])
      new_mean_2.append(0)
  # Second pass: add unmatched mzs_2 with 0 for mzs_1
  for j, _ in enumerate(mzs_2):
    if j not in matched_indices_2:
      new_mean_1.append(0)
      new_mean_2.append(mean_2[j])
  # Convert to numpy arrays
  new_mean_1 = np.array(new_mean_1)
  new_mean_2 = np.array(new_mean_2)
  # Calculate and return the correlation
  return np.corrcoef(new_mean_1, new_mean_2)[0, 1]


def plot_corr_matrix(
    corr_df: pd.DataFrame, y_label: str, x_label: str,
    mark_biopsies: bool = False, sort_biopsies: bool = False,
    figsize: Tuple[float, float] = (11.69, 8.27), cbar: bool = False,
    annot: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
  """Function to plot correlation matrix.

  Args:
      corr_df (pd.DataFrame): Correlation matrix.
      y_label (str): Y axis label.
      x_label (str): X axis label.
      mark_biopsies (bool, optional): Indicator if to mark biopsies from the
          same patient in red box. Defaults to False.
      sort_biopsies (bool, optional): Indicator if to sort biopsies in figure.
          Defaults to False.
      figsize (Tuple[float, float], optional): Figure size. Defaults to
          (11.69, 8.27).
      cbar (bool, optional): Indicator to whether to draw a color bar. Defaults
          to False.
      annot (bool, optional): Indicator to whether to write the data value in
          each cell. Defaults to False.

  Returns:
      Tuple[plt.Figure, plt.Axes]: Figure and axes.
  """
  # Create figure
  fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)

  # Sort biopsies by their number
  if sort_biopsies:
    index_keys = corr_df.index.to_series(
    ).apply(lambda s: int(re.sub(r"HG |-s|-r|_.", "", s))).sort_values().index
    col_keys = corr_df.columns.to_series(
    ).apply(lambda s: int(re.sub(r"HG |-s|-r|_.", "", s))).sort_values().index
    corr_df = corr_df.loc[index_keys.to_list(), col_keys.to_list()]

  # Plot correlation matrix
  ax = sns.heatmap(
      corr_df, annot=annot, cmap="YlGn", fmt=".2f", vmin=-1, vmax=1,
      linewidth=.5, linecolor='w', square=True, cbar=cbar, ax=ax
  )
  # Customize plot
  fc.set_titles_and_labels(
      ax, title="", xlabel=x_label.capitalize(), ylabel=y_label.capitalize()
  )
  fc.customize_ticks(ax)
  # Mark cells of biopsies from the same patient
  if mark_biopsies:
    for index_i, i in enumerate(corr_df.columns):
      for index_j, j in enumerate(corr_df.index):
        i_num = re.sub(r"HG |-s|-r|_.", "", i)
        j_num = re.sub(r"HG |-s|-r|_.", "", j)
        if i_num == j_num:
          ax.add_patch(
              Rectangle(
                  (index_i, index_j), 1, 1, fill=False, edgecolor="red", lw=2
              )
          )
  return fig, ax


def plot_corr_ranks(
    corr_df: pd.DataFrame, figsize: Tuple[float, float] = (11.69, 8.27)
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
  """Function to plot correlation ranks for pair biopsies.

  Args:
      corr_df (pd.DataFrame): Correlation matrix.
      figsize (Tuple[float, float], optional): Figure size. Defaults to
          (11.69, 8.27).

  Returns:
      Tuple[plt.Figure, plt.Axes]: Figure, axes and pair ranks dataframe.
  """
  # Get ranks of correlation matrix
  ranks = corr_df.rank(axis=1, method="min", ascending=False).astype(int)
  # Define list to save pair ranks
  pair_ranks = {}
  # Get ranks of pairs
  for i in corr_df.index:
    for j in corr_df.columns:
      i_num = re.sub(r"HG |-s|-r|_.", "", j)
      j_num = re.sub(r"HG |-s|-r|_.", "", i)
      if i_num == j_num:
        pair_rank = pair_ranks.get(i, (None, np.inf))
        if pair_rank[1] > ranks.loc[i, j]:
          pair_ranks[i] = (j, ranks.loc[i, j])
  # Create pair ranks to CSV with columns "replica" and "section"
  pair_ranks_df = pd.DataFrame(
      [
          {"replica": i, "section": value[0], "corr_rank": value[1]}
          for i, value in pair_ranks.items()
      ]
  )
  # Create figure
  fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
  # Plot pair correlation rank count
  ax = sns.countplot(
      data=pd.DataFrame({"ranks": [i[1] for i in pair_ranks.values()]}),
      x="ranks", ax=ax, color="#3274a1"
  )
  # Customize plot
  fc.set_titles_and_labels(
      ax, title="", xlabel="Pairs correlation rank", ylabel="Count"
  )
  fc.customize_spines(ax)
  fc.customize_ticks(ax)
  return fig, ax, pair_ranks_df


def plot_corr_distribution(
    corr_df: pd.DataFrame, figsize: Tuple[float, float] = (11.69, 8.27)
) -> Tuple[plt.Figure, plt.Axes]:
  """Function to plot correlation distribution.

  Args:
      corr_df (pd.DataFrame): Correlation matrix.
      figsize (Tuple[float, float], optional): Figure size. Defaults to
          (11.69, 8.27).

  Returns:
      Tuple[plt.Figure, plt.Axes]: Figure and axes.
  """
  # Create figure
  fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
  # Plot pair correlation rank count
  ax = sns.histplot(
      corr_df.values.reshape(-1), bins=np.arange(-0.05, 1.1, 0.1),
      stat="probability", ax=ax, color="#3274a1"
  )
  # Customize plot
  fc.set_titles_and_labels(
      ax, title="", xlabel="Correlation", ylabel="Probability"
  )
  fc.customize_spines(ax)
  fc.customize_ticks(ax)
  ax.set_ylim((0, 1))
  return fig, ax


def get_dataset_mzs(metadata_df: pd.DataFrame,
                    processed_data: Path) -> List[Tuple[np.ndarray, str]]:
  """Function to get the mzs for all files in the dataset.

  Args:
    metadata_df (pd.DataFrame): The metadata dataframe.
    processed_data (Path): The path to the processed data folder.

  Returns:
    List[Tuple[np.ndarray, str]]: A list of tuples containing the mzs and the
        sample type.

  """
  # Define list to store all mzs
  all_mzs = []
  # Go over all files
  for p in Path(processed_data).iterdir():
    # Get mzs
    mzs = np.load(p / "mzs.npy")
    # Append to all mzs
    if metadata_df[metadata_df.sample_file_name == p.stem
                  ].sample_type.values[0] == 'replica':
      all_mzs.append((mzs, "r"))
    else:
      all_mzs.append((mzs, "s"))
  # Return all mzs
  return all_mzs


def find_common_mzs(
    *mz_lists: List[np.ndarray], tolerance: float = 0.02
) -> np.ndarray:
  """Function to find the common m/z values across multiple lists of m/z values.

  Args:
    tolerance (float, optional): The tolerance for closeness in m/z values.

  Returns:
    np.ndarray: An array of common m/z values.

  """
  # Start with the first list as a base reference
  common_mzs = []
  # Iterate through each m/z in the first list
  for mz in mz_lists[0]:
    # Assume the current m/z value is common
    is_common = True
    # Compare it with the corresponding values in all other lists
    for mz_list in mz_lists[1:]:
      # Check if at least one value in the current list is within the tolerance
      if not np.any(np.abs(mz - mz_list) < tolerance):
        # If not, the current m/z value is not common
        is_common = False
        break
    # If the value is within tolerance in all lists, add to the common list
    if is_common:
      common_mzs.append(mz)
  # Return the common m/z values as a numpy array
  return np.array(common_mzs)


def build_mz_mapping(mz_lists: List[np.ndarray],
                     tolerance: float = 0.02) -> List[Dict[float, float]]:
  """Function to build a mapping of common m/z values across multiple lists.

  Args:
    mz_lists (List[np.ndarray]): A list of m/z value lists.
    tolerance (float, optional): The tolerance for closeness in m/z values.

  Returns:
    List[Dict[float, float]]: A list of dictionaries mapping common m/z values
        to corresponding m/z values in each list

  """
  # Step 1: Get the common m/z values using the updated function
  max_common_mzs = find_common_mzs(*mz_lists, tolerance=tolerance)
  # Step 2: Build the mapping for each list
  mappings = []
  # For each list of m/z values
  for mz_list in mz_lists:
    # Initialize the mapping for the current list
    mz_mapping = {}
    # For each common m/z value, find the closest m/z value in the current list
    for common_mz in max_common_mzs:
      # Find the m/z value in the current list that is closest to the current
      # common m/z value
      differences = np.abs(common_mz - mz_list)
      min_diff_index = np.argmin(differences)
      # If the closest m/z value is within tolerance, map it
      if differences[min_diff_index] <= tolerance:
        mz_mapping[common_mz] = mz_list[min_diff_index]
      else:
        mz_mapping[common_mz] = None  # No m/z value within tolerance
    # Append the mapping for the current list
    mappings.append(mz_mapping)
  # Return the list of mappings
  return mappings


def normalize_and_segment_rois(
    raw_data: Path, metadata_df: pd.DataFrame, mass_res: float,
    representative_peaks: List[float], processed_data: Path
):
  #  Get normalizer object
  normalizer = TICNormalizer()
  # Get the aw files
  raw_files = list(Path(raw_data).iterdir())
  # Loop through the files
  for p in raw_files:
    # Subset metadata for the current file
    l_metadata_df = metadata_df.loc[metadata_df["file_name"] == p.stem].copy()
    # Get image and mzs
    img, mzs = process_spectral_data_to_image(p)
    # Apply normalization to the intensities
    img_norm = np.apply_along_axis(
        lambda intensities: (normalizer.normalize((mzs, intensities)))[1],
        axis=2, arr=img
    )
    # Merge mz channels with difference less than 0.02 as they are likely to be
    # the same mz
    mzs_merged, img_norm_merged = merge_mzs_and_intensities(
        mzs, img_norm, threshold=mass_res
    )
    # Apply segmentation
    segmenter = MeanSegmentation(mzs_merged, representative_peaks, mass_res * 4)
    img_segmented = segmenter.segment(img_norm_merged)
    # Loop through the ROIs
    for index, roi in l_metadata_df.iterrows():
      # Define path to ROI file after processing
      output_path = processed_data / f"{roi.sample_file_name}"
      # Create output folder if doesn't exist
      output_path.mkdir(parents=True, exist_ok=True)
      # Get the ROI coordinates
      x_min, x_max = roi["x_min"], roi["x_max"]
      y_min, y_max = roi["y_min"], roi["y_max"]
      # Get the ROI image
      roi_img_norm = img_norm_merged[y_min:y_max, x_min:x_max, :]
      # Get the ROI segmented image
      roi_img_segmented = img_segmented[y_min:y_max, x_min:x_max]
      # Save the information
      np.save(output_path / f"tic_normalized.npy", roi_img_norm)
      np.save(output_path / f"segmentation.npy", roi_img_segmented)
      np.save(output_path / f"mzs.npy", mzs_merged)


def create_common_mzs_rois(
    metadata_df: pd.DataFrame, mass_res: float, processed_data: Path
):
  # Get the mzs for all files
  all_mzs = get_dataset_mzs(metadata_df, processed_data)
  # Build the mz mapping for all files to the common mzs
  mappings = build_mz_mapping(
      [np.array(mzs) for mzs, _ in all_mzs], tolerance=mass_res
  )
  # Get the common mzs
  common_mzs = np.array(list(mappings[0].keys()))
  # Loop through the files
  for i, p in enumerate(list(Path(processed_data).iterdir())):
    # Get mzs
    img = np.load(p / "tic_normalized.npy")
    mzs = np.load(p / "mzs.npy")
    # Get the mapping for the current file
    mapping = mappings[i]
    # Create a new image for the mapped mzs
    mapped_img = np.zeros((img.shape[0], img.shape[1], len(mapping)))
    # Map the intensities to the new image
    for j, (common_mz, mz) in enumerate(mapping.items()):
      if mz is not None:
        mz_index = np.where(mzs == mz)[0][0]
        mapped_img[:, :, j] = img[:, :, mz_index]
    # Save the mapped image
    np.save(p / "mapped_tic_normalized.npy", mapped_img)
    # Save the common m/z values
    np.save(p / "common_mzs.npy", common_mzs)


def load_and_segment_mean(file: List[Path], common_mzs: bool = False):
  prefix = "mapped_" if common_mzs else ""
  img = np.load(file / f"{prefix}tic_normalized.npy")
  seg = np.load(file / "segmentation.npy")
  return img[seg == 1].mean(axis=0)


def compute_correlation_matrix(
    files: List[Path], mass_resolution: float, common_mzs: bool = False
):
  matrix = np.zeros((len(files), len(files)))
  for i, p1 in enumerate(files):
    for j, p2 in enumerate(files):
      mean_1 = load_and_segment_mean(p1, common_mzs=common_mzs)
      mean_2 = load_and_segment_mean(p2, common_mzs=common_mzs)
      if common_mzs:
        correlation = np.corrcoef(mean_1, mean_2)[0, 1]
      else:
        mzs1, mzs2 = np.load(p1 / "mzs.npy"), np.load(p2 / "mzs.npy")
        correlation = pair_corr(mean_1, mean_2, mzs1, mzs2, mass_resolution)
      matrix[i, j] = correlation
  return pd.DataFrame(
      matrix, columns=[p.stem for p in files], index=[p.stem for p in files]
  )


def subset_and_sort_corr_df(corr_df: pd.DataFrame, suffix: str):
  samples = sorted(
      [col for col in corr_df.columns if suffix in col], key=lambda x: int(
          x.replace("HG ", "").replace("_", " ").replace("-", " ").split(' ')[0]
      )
  )
  return corr_df.loc[samples, samples]


def plot_and_save_corr_matrices(
    corr_dfs: pd.DataFrame, figure_path: Path, file_suffix: str
):
  for label, corr_df in corr_dfs.items():
    fig, ax = plot_corr_matrix(
        corr_df, label if label != "Replica-Section" else "Replica",
        label if label != "Replica-Section" else "Section",
        mark_biopsies=label == "Replica-Section", sort_biopsies=True
    )
    plt.tight_layout()
    plt.savefig(
        figure_path /
        f"{label.lower().replace('-', '_')}_correlation_heatmap_{file_suffix}.png",
        transparent=True, bbox_inches='tight', dpi=1200
    )
    plt.show()


def plot_multiple_replicas_correlation(
    corr_df: pd.DataFrame, figure_path: Path, suffix: str,
    figsize: Tuple[float, float] = (11.69, 8.27)
):
  # Define the biopsies with multiple replicas
  multiple_replicas = ["HG 6_1-r", "HG 6_2-r", "HG 18_1-r", "HG 18_2-r"]
  # Define the color palette
  palette = {
      "HG 6_1-r": "navy", "HG 6_2-r": "lightsteelblue", "HG 18_1-r":
      "tab:orange", "HG 18_2-r": "bisque"
  }
  # Create figure
  fig, ax = plt.subplots(1, figsize=figsize, tight_layout=True)
  # Create scatter plot
  ax = sns.scatterplot(
      data=corr_df.T.loc[:, multiple_replicas], ax=ax, s=100,
      markers=["v", "^", "v", "^"], palette=palette
  )
  # Add vertical lines for representative sections
  for tick in ["HG 6-s", "HG 18-s"]:
    # Get the index of the tick and plot vertical line
    tick_pos = corr_df.columns.get_loc(tick)
    ax.axvline(x=tick_pos, color="gray", linestyle="--", linewidth=2.5)
  # Set the legend
  lgnd = ax.legend(
      title="", loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=2,
      fancybox=True, shadow=True, prop={'size': 14, 'weight': 'bold'}
  )
  lgnd.legend_handles[0]._sizes = [100]
  lgnd.legend_handles[1]._sizes = [100]
  # Customize plot
  fc.set_titles_and_labels(ax, title="", xlabel="", ylabel="Correlation")
  fc.customize_ticks(ax, rotate_x_ticks=90)
  fc.customize_spines(ax)
  ax.set_ylim(-0.1, 1.1)
  # Show plot
  plt.tight_layout()
  plt.savefig(
      figure_path / f"multiple_replicas_correlation_{suffix}.png",
      transparent=True, bbox_inches='tight', dpi=1200
  )
  plt.show()


def correlation_analysis(
    processed_dir: Path, figure_path: Path, mass_resolution: float
):
  # Get the processed files
  processed_files = list(Path(processed_dir).iterdir())
  # Define the file suffixes
  file_suffixes = ["all_mzs", "common_mzs"]
  # Loop through the file suffixes
  for common_mzs, suffix in zip([False, True], file_suffixes):
    # Compute correlation matrix
    """
    corr_df = compute_correlation_matrix(
        processed_files, mass_resolution=mass_resolution, common_mzs=common_mzs
    )
    # Subset and sort for replicas and sections
    corr_df_r = subset_and_sort_corr_df(corr_df, "-r")
    corr_df_s = subset_and_sort_corr_df(corr_df, "-s")
    corr_df_rs = corr_df.loc[corr_df_r.index, corr_df_s.columns]
    """
    # Load the correlation matrices
    corr_df_r = pd.read_csv(
        figure_path / f"corr_df_r_{suffix}.csv", index_col=0
    )
    corr_df_s = pd.read_csv(
        figure_path / f"corr_df_s_{suffix}.csv", index_col=0
    )
    corr_df_rs = pd.read_csv(
        figure_path / f"corr_df_rs_{suffix}.csv", index_col=0
    )
    # Plot and save all matrices
    plot_and_save_corr_matrices(
        {
            "Section": corr_df_s, "Replica": corr_df_r, "Replica-Section":
            corr_df_rs
        }, figure_path, suffix
    )
    # Plot ranks and distribution
    fig, ax, pair_ranks_df = plot_corr_ranks(corr_df_rs)
    plt.tight_layout()
    plt.savefig(
        figure_path / f"replica_section_correlation_ranks_{suffix}.png",
        transparent=True, bbox_inches='tight', dpi=1200
    )
    plt.show()
    fig, ax = plot_corr_distribution(corr_df_rs)
    plt.tight_layout()
    plt.savefig(
        figure_path / f"replica_section_correlation_distribution_{suffix}.png",
        transparent=True, bbox_inches='tight', dpi=1200
    )
    plt.show()
    # Plot multiple replicas correlation
    plot_multiple_replicas_correlation(corr_df_rs, figure_path, suffix)
    # Save the correlation matrices
    corr_df_r.to_csv(figure_path / f"corr_df_r_{suffix}.csv")
    corr_df_s.to_csv(figure_path / f"corr_df_s_{suffix}.csv")
    corr_df_rs.to_csv(figure_path / f"corr_df_rs_{suffix}.csv")
    pair_ranks_df.to_csv(
        figure_path / f"pair_ranks_df_rs_{suffix}.csv", index=False
    )


def classification_analysis(
    figure_path: Path, processed_files: List[Path], model: str,
    n_permutations: int, n_splits: int
):
  binary_classification_main(
      figure_path, processed_files, model, n_permutations, n_splits
  )


def main():
  """Function containing main code"""
  # Define current folder using this file
  CWD = Path(os.path.dirname(os.path.abspath(__file__)))
  # Define folder that contains dataset
  DHG_PATH = CWD / ".." / ".." / "data" / "DHG"
  # Define folder that contains raw data
  RAW_DATA = DHG_PATH / "raw_txt"
  # Define folder to save processed data
  PROCESSED_DATA = DHG_PATH / "processed_txt"
  # Define file that contains metadata
  METADATA_PATH = DHG_PATH / "txt_metadata.csv"
  # Define path to save plots and results
  FIGURES_PATH = CWD / "dhg"
  FIGURES_PATH.mkdir(exist_ok=True, parents=True)
  # Define mass range start value
  MZ_START = 600
  # Define mass range end value
  MZ_END = 900
  # Define mass resolution of the data
  MASS_RESOLUTION = 0.02
  # Define representative peaks
  REPRESENTATIVE_PEAKS = [794.5, 834.5, 886.6]
  # Read metadata csv
  metadata_df = pd.read_csv(METADATA_PATH)
  """
  print("Starting processing of spectral data")
  # Normalize and segment ROIs
  normalize_and_segment_rois(
      RAW_DATA, metadata_df, MASS_RESOLUTION, REPRESENTATIVE_PEAKS,
      PROCESSED_DATA
  )
  # Create common mzs ROIs
  create_common_mzs_rois(metadata_df, MASS_RESOLUTION, PROCESSED_DATA)
  print("Finished processing of spectral data")
  
  # Perform correlation analysis
  print("Starting correlation analysis")
  correlation_analysis(
      PROCESSED_DATA, FIGURES_PATH / "correlation", MASS_RESOLUTION
  )
  print("Finished correlation analysis")
  """

  # Perform binary classification
  print("Starting classification analysis")
  classification_analysis(
      FIGURES_PATH / "classification", list(Path(PROCESSED_DATA).iterdir()),
      "lightgbm", 100, 1000
  )
  print("Finished classification analysis")


if __name__ == '__main__':
  main()
