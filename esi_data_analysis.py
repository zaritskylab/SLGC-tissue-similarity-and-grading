"""ESI data analysis
The script should be ran and will read mzML files, process data and create
relevant plot such as line plots of the spectras.

"""
import os
import string
import itertools
import pymzml
import numpy as np
import seaborn as sns
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Callable
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from processing import EqualWidthBinning, TICNormalizer


def aggregate_spectra(spectra: Dict[float, List],
                      aggregator: Callable) -> Tuple[np.ndarray, np.ndarray]:
  """
  Aggregates spectra using a specified aggregation function.

  Args:
    spectra (Dict[float, List]): Dictionary with m/z values as keys and lists
        of intensities.
    aggregator (Callable): Function to aggregate intensities (e.g., sum,
        np.mean).

  Returns:
    Tuple[np.ndarray, np.ndarray]: Two arrays, one for m/z values and one for
        aggregated intensities.

  """
  # Aggregate intensities for each m/z value using the provided aggregator
  # function
  aggregated_spectra = {
      key: aggregator(values) for key, values in spectra.items()
  }
  # Convert the aggregated data into numpy arrays for m/z values and
  # intensities
  return np.asarray(list(aggregated_spectra.keys())
                   ), np.asarray(list(aggregated_spectra.values()))


def get_spectra(mzml_path: Path) -> Dict[float, List]:
  """
  Extracts spectra from a mzml file.

  Args:
    mzml_path (Path): Path to the mzml file.

  Returns:
    Dict[float, List]: Dictionary with m/z values and corresponding intensities.

  """
  try:
    # Read the mzml file and initialize a defaultdict for efficient data
    # aggregation
    run = pymzml.run.Reader(str(mzml_path))
    spectra = defaultdict(list)
    # Iterate through each spectrum in the file
    for _, spec in enumerate(run):
      # Extract raw peaks
      peaks = spec.peaks("raw")
      # Aggregate intensities for each m/z value
      for mz, intensity in zip(peaks[:, 0], peaks[:, 1]):
        spectra[mz].append(intensity)
    # Return the spectra
    return dict(spectra)
  except Exception as e:
    # Handle any errors during file reading
    print(f"Error reading {mzml_path}: {e}")
    return {}


def file_plot_name(file_name: str) -> str:
  """
  Generates a plot name based on a file name.

  Args:
    file_name (str): The file name to process.

  Returns:
    str: A processed string suitable for use as a plot name.

  """
  # Split the file name by spaces
  parts = file_name.split(" ")
  # Return a part of the file name depending on its length
  return " ".join(parts[3:]) if len(parts) > 4 else " ".join(parts[2:])


def plot_spectras(
    spectras: Dict[str, Dict[float, List]], save_path: Path
) -> None:
  """
  Plots the spectras and saves the figure to a given path.

  Args:
    spectras (Dict[str, Dict[float, List]]): Dictionary of spectras.
    save_path (Path): Path where the plot will be saved.

  """
  # Create a figure with subplots
  fig, axes = plt.subplots(len(spectras.items()), 2, figsize=(7, 10))
  # Loop through the list and plot each spectra
  n = 0
  for ax, key in zip(axes.flat, spectras.keys()):
    # Plot the spectra on the current subplot
    ax.plot(spectras[key][0], spectras[key][1], linewidth=2)
    # Add labels and title for the current subplot
    ax.set_xlabel('m/z')
    ax.set_ylabel('Intensity')
    # Customize the spines
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(2.5)
      ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Customize the ticks
    t = [
        int(round(spectras[key][0].min(), 0)),
        int(round(spectras[key][0].max(), 0))
    ]
    ax.set_xticks(t, t)
    t = [round(spectras[key][1].min(), 2), round(spectras[key][1].max(), 2)]
    ax.set_yticks(t, t)
    for yticklabel in ax.get_yticklabels():
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
    ax.set_xlabel(ax.get_xlabel(), fontsize=14, weight='bold', color='0.2')
    ax.set_ylabel(ax.get_ylabel(), fontsize=14, weight='bold', color='0.2')
    # Annotate the subplots
    ax.text(
        -0.3, 1.1, string.ascii_lowercase[n], transform=ax.transAxes, size=16,
        weight='bold'
    )
    n += 1
  # Delete leftover plot
  for j, ax in enumerate(axes.flat):
    if j >= len(spectras.items()):
      fig.delaxes(ax)
  # Add space between subplots
  plt.tight_layout()
  # Save the plot
  plt.savefig(
      save_path.joinpath("spectras_line_plot.png"), bbox_inches='tight',
      dpi=1200, transparent=True
  )
  # Show the plot
  plt.show()


def plot_spectras_corr(
    spectras: Dict[str, Dict[float, List]], save_path: Path
) -> None:
  """
  Plots a correlation matrix for the given spectras and saves the figure.

  Args:
    spectras (Dict[str, Dict[float, List]]): Dictionary of spectras.
    save_path (Path): Path where the correlation plot will be saved.

  """
  # Compute correlation between spectras
  correlation_matrix = pd.DataFrame(
      {
          file_plot_name(file): spectra[1] for file, spectra in spectras.items()
      }
  ).corr()
  # correlation_matrix = correlation_matrix.drop("TISSUE 3-2", axis=0).drop(
  # "TISSUE 3-2", axis=1)
  # Create figure
  fig, ax = plt.subplots(1, 1, figsize=(7, 7), tight_layout=True)
  # Plot correlation matrix
  ax = sns.heatmap(
      correlation_matrix, annot=True, cbar=False, cmap="YlGn", fmt=".2f",
      vmin=-1, vmax=1, linewidth=2, linecolor='w', square=True, ax=ax,
      annot_kws={
          "fontsize": 14,
          "fontweight": 'bold'
      }
  )
  # Customize the ticks
  for yticklabel in ax.get_yticklabels():
    yticklabel.set_fontweight('bold')
    yticklabel.set_fontsize(14)
    yticklabel.set_color('0.2')
  for xticklabel in ax.get_xticklabels():
    xticklabel.set_fontweight('bold')
    xticklabel.set_fontsize(14)
    xticklabel.set_color('0.2')
  # Remove ticks line
  ax.tick_params(axis='both', length=0)
  # Add space between subplots
  plt.tight_layout()
  # Save the plot
  plt.savefig(
      save_path.joinpath("spectras_correlation.png"), bbox_inches='tight',
      dpi=1200, transparent=True
  )
  # Show the plot
  plt.show()


def plot_spectras_best_fit(
    spectras: Dict[str, Dict[float, List]], save_path: Path
) -> None:
  """
  Plots the best fit line for the given spectras and saves the figure.

  Args:
    spectras (Dict[str, Dict[float, List]]): Dictionary of spectras.
    save_path (Path): Path where the best fit plot will be saved.

  """
  # Create directory to save figures
  save_path = save_path.joinpath("best_fit_plots")
  save_path.mkdir(parents=True, exist_ok=True)
  # Loop over all combinations
  for i, j in itertools.combinations(spectras.keys(), 2):
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    # Plot identity diagonal
    min_axis = min([min(spectras[i][1]), min(spectras[j][1])])
    max_axis = max([max(spectras[i][1]), max(spectras[j][1])])
    ax.plot(
        [min_axis, max_axis], [min_axis, max_axis], linestyle='--',
        color='gray', linewidth=2, label="identity"
    )
    # Plot scatter
    ax = sns.scatterplot(
        y=spectras[i][1], x=spectras[j][1], ax=ax, alpha=0.7, s=70,
        label=fr"$R^2$ = {round(r2_score(spectras[i][1], spectras[j][1]), 2)}"
    )
    # Customize the spines
    for axis in ['bottom', 'left']:
      ax.spines[axis].set_linewidth(2.5)
      ax.spines[axis].set_color('0.2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Customize the ticks
    t = [round(min_axis, 2), round(max_axis, 2)]
    ax.set_xticks(t, t)
    ax.set_yticks(t, t)
    for yticklabel in ax.get_yticklabels():
      yticklabel.set_fontweight('bold')
      yticklabel.set_fontsize(14)
      yticklabel.set_color('0.2')
    for xticklabel in ax.get_xticklabels():
      xticklabel.set_fontweight('bold')
      xticklabel.set_fontsize(14)
      xticklabel.set_color('0.2')
    # Customize the axis labels
    ax.set_ylabel(file_plot_name(i), fontsize=14, weight='bold', color='0.2')
    ax.set_xlabel(file_plot_name(j), fontsize=14, weight='bold', color='0.2')
    # Remove ticks line
    ax.tick_params(axis='both', length=0)
    # Add legend
    plt.legend(prop={"size": 14}, loc="upper left")
    # Add space between subplots
    plt.tight_layout()
    # Save the plot
    plt.savefig(
        save_path.joinpath(f"{i}_{j}_scatter.png"), bbox_inches='tight',
        dpi=300, transparent=True
    )
    # Show the plot
    plt.show()


def main():
  """Function containing main code"""
  # Define current folder using this file
  CWD = Path(os.path.dirname(os.path.abspath(__file__)))
  # Define path to data
  ESI_PATH = Path(("D:/Thesis/chapter_one/data/ESI"))
  # Define mass range start value
  MZ_START = 50
  # Define mass range end value
  MZ_END = 1200
  # Define mass resolution of the data
  MASS_RESOLUTION = 4

  # Process each mzml file in the specified directory
  raw_spectras = {
      file.stem: get_spectra(file)
      for file in ESI_PATH.iterdir()
      if file.suffix.lower() == '.mzml'
  }
  # Aggregate the spectra using sum as the aggregation function
  aggregated_spectras = {
      file: aggregate_spectra(spectra, np.sum)
      for file, spectra in raw_spectras.items()
  }
  # Get normalizer object
  normalizer = TICNormalizer()
  # Get binning object
  binning = EqualWidthBinning(MZ_START, MZ_END, MASS_RESOLUTION / 2)
  # Create process pipe
  process_pipe = (
      lambda mzs, intensities:
      (binning.bin(normalizer.normalize((mzs, intensities))))
  )
  # Apply process pipe
  processed_spectras = {
      file: process_pipe(*spectra)
      for file, spectra in aggregated_spectras.items()
  }
  # Define path to save figures
  PLOT_PATH = Path(CWD / f"esi/bin_width_{MASS_RESOLUTION / 2}")
  # Create dirs
  PLOT_PATH.mkdir(parents=True, exist_ok=True)
  # Create plots
  plot_spectras(processed_spectras, PLOT_PATH)
  plot_spectras_corr(processed_spectras, PLOT_PATH)
  plot_spectras_best_fit(processed_spectras, PLOT_PATH)


if __name__ == "__main__":
  main()
