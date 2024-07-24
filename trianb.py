"""TRIANB data analysis
The script should be ran and will read imzML files, process data and create
relevant plot such as segmentations maps.

"""

import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict
from pathlib import Path
from matplotlib import pyplot as plt
from processing import process, aligned_representation
from correlation import correlation_analysis


def get_plot_name(sample_file_name: str) -> str:
  """Function to fet the plot name.

  Args:
    sample_file_name (str): Sample file name.

  Returns:
    str: Plot name.
  
  """
  return "Rep_" + sample_file_name.split("Rep")[1][0]

def plot_corr(output_path: Path) -> None:
  """Function to plot the correlation matrix.

  Args:
    output_path (Path): Path to save the plot and to read the correlation 
        matrix from.
  
  """
  # Read the correlation matrix
  corr_matrix = pd.read_csv(
      output_path / 'meaningful-signal_tissue_tissue.csv', index_col=0
  )
  # Generate the heatmap
  _, ax = plt.subplots(1, 1, figsize=(15, 5), tight_layout=True)
  # Plot correlation matrix
  ax = sns.heatmap(
      corr_matrix, annot=True, cbar=False, cmap="YlGn", fmt=".2f",
      vmin=-1, vmax=1, linewidth=2, linecolor='w', square=True, ax=ax,
      annot_kws={"fontsize": 14, "fontweight": 'bold', 'color': '0.2'}
  )
  # Set ticks
  x_ticks_labels = [get_plot_name(label) for label in corr_matrix.columns]
  y_ticks_labels = [get_plot_name(label) for label in corr_matrix.index]
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
  ax.set_ylabel("Replica", fontsize=14, weight='bold', color='0.2')
  ax.set_xlabel("Replica", fontsize=14, weight='bold', color='0.2')
  # Add space between subplots
  plt.tight_layout()
  # Save the plot
  plt.savefig(
      output_path.joinpath("correlations.png"), bbox_inches='tight', dpi=1200,
      transparent=True
  )
  # Show plot
  plt.show()


def plot_msi_segmentation(
   masks: Dict[str, np.ndarray], save_path: str
) -> None:
  """Plot the segmentation masks

  Args:
    masks (Dict[str, np.ndarray]): Dictionary containing the masks.
    save_path (str): Path to save the plot.
  
  """
  # Create a plot with the segmentation masks
  _, axs = plt.subplots(1, len(masks))
  for i, (name, mask) in enumerate(masks.items()):
    ax = axs[i]
    ax.imshow(mask, cmap="gray_r")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(name, fontsize=14, weight='bold', color='0.2')
  # Add space between subplots
  plt.tight_layout()
  # Save the plot
  plt.savefig(
      save_path.joinpath("segmentations.png"), bbox_inches='tight', dpi=1200,
      transparent=True
  )
  # Show the plot
  plt.show()



def main():
  """Function containing main code"""
  # Define current folder using this file
  CWD = Path(os.path.dirname(os.path.abspath(__file__)))
  # Define folder that contains the revision chip type dataset
  TRIANB_PATH = Path(os.path.join(CWD, "..", "data", "TRIANB"))
  # Define folder that contains raw data
  RAW_DATA = TRIANB_PATH.joinpath("raw")
  # Define folder to save aligned data
  ALIGNED_DATA = TRIANB_PATH.joinpath("aligned")
  # Define folder to save processed data
  PROCESSED_DATA = TRIANB_PATH.joinpath("processed")
  # Define file that contains dhg metadata
  METADATA_PATH = TRIANB_PATH.joinpath("metadata.csv")
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
  REPRESENTATIVE_PEAKS = [794.5, 834.5, 886.6]
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
    # Create output folder if doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Align MSI
    aligned_representation(
        msi_path, output_path, LOCK_MASS_PEAK, LOCK_MASK_TOL
    )
  
  # Loop over each ROI in data frame
  for index, roi in metadata_df.iterrows():
    # Define path to msi imzML file
    msi_path = os.path.join(ALIGNED_DATA, f"{roi.file_name}.imzML")
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
  PLOT_PATH = Path(CWD / "trianb/")
  # Create dirs
  PLOT_PATH.mkdir(parents=True, exist_ok=True)
  # Run correlation analysis
  correlation_analysis(PROCESSED_DATA, PLOT_PATH)
  
  # Define path to save figures
  PLOT_PATH = Path(CWD / "trianb/")
  # Create dirs
  PLOT_PATH.mkdir(parents=True, exist_ok=True)
  # Read correlation matrix
  plot_corr(PLOT_PATH)
  # Plot the segmentation masks
  masks = {}
  for folder in PROCESSED_DATA.iterdir():
    name = folder.stem
    masks[get_plot_name(name)] = np.load(
      folder.joinpath("segmentation.npy"), mmap_mode='r')
  plot_msi_segmentation(masks, PLOT_PATH)

if __name__ == '__main__':
  main()
