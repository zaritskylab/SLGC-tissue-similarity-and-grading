""" Module to create images of mz value plot for each imzML file"""

import argparse
import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from itertools import product
from pyimzml.ImzMLParser import ImzMLParser
from typing import Tuple

sns.set_style("white")


def mz_length_plot(p: ImzMLParser) -> Tuple[plt.Figure, plt.Axes]:
  """
  Method to create intensity plot of mz lengths

  Args:
      p (ImzMLParser): parse object of an imzML file.
      title (str): title of the plot.

  Returns:
      Tuple[plt.Figure, plt.Axes]: matplotlib figure and axis.
  """

  # get dimensions
  max_y = p.imzmldict["max count of pixels y"]
  max_x = p.imzmldict["max count of pixels x"]

  # create indexes for each mz length
  points = np.array(list(product(range(0, max_y), range(0, max_x))))

  # get mz lengths
  mz_lengths = np.asarray(p.mzLengths)

  # create plot
  fig_f, ax_f = plt.subplots(figsize=(10, 5))

  # plot intensity
  sc = ax_f.scatter(points[:, 1],
                    points[:, 0],
                    c=mz_lengths.reshape((max_y, max_x)),
                    edgecolors="none",
                    cmap="crest",
                    marker="s",
                    linewidth=0,
                    vmin=mz_lengths.min(),
                    vmax=mz_lengths.max())

  # add color bar
  ax_f.figure.colorbar(sc,
                       ticks=np.linspace(mz_lengths.min(), mz_lengths.max(),
                                         10))

  # set plot title
  ax_f.set_title("Number of MZ Values")

  # change axis
  ax_f.set_xticks([0, max_x], [0, max_x])
  ax_f.set_yticks([0, max_y], [0, max_y])

  return fig_f, ax_f


if __name__ == "__main__":
  # get command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("-i",
                      required=True,
                      help="path to folder containing imzML files")
  parser.add_argument("-o",
                      required=True,
                      help="path to folder to output images")
  args = parser.parse_args()

  # get all imzML files in input folder
  files = [file for file in os.listdir(args.i) if file.endswith(".imzML")]

  # loop over all imzML files
  for file in files:
    # parse imzML file
    with ImzMLParser(os.path.join(args.i, file)) as p_l:
      # get file name of imzML file
      title_l = os.path.splitext(file)[0]
      # get plot mz Lengths
      fig, ax = mz_length_plot(p_l)
      # save plot
      fig.savefig(os.path.join(args.o, title_l + ".png"),
                  transparent=True,
                  bbox_inches="tight",
                  pad_inches=0)
