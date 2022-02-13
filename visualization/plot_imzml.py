"""imzML Images Plotting

This module enables the user to plot the imzML images in a given folder.

This file can also be imported as a module and contains the following
functions:

    * main - module main function.
    * get_ion_image - Function to get an image representation of the intensity
    distribution of the ion with specified m/z values.

"""

import configparser
import argparse
import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from os.path import dirname, realpath, join
from pyimzml.ImzMLParser import ImzMLParser, _bisect_spectrum
from typing import List, Callable


def get_ion_image(p: ImzMLParser,
                  mz_values: List[float],
                  tol: float = 0.1,
                  z: int = 1,
                  reduce_func: Callable = sum) -> np.ndarray:
  """
  Get an image representation of the intensity distribution
  of the ion with specified m/z values.

  By default, the intensity values within the tolerance region are summed.

  Args:
      p (ImzMLParser): the ImzMLParser (or anything else with similar
      attributes) for the desired dataset.
      mz_values (List[float]): m/z values for which the ion image
      shall be returned
      tol (float, optional): Absolute tolerance for the m/z value,
      such that all ions with values mz_value-|tol| <= x <= mz_value+|tol|
      are included. Defaults to 0.1
      z (int, optional): z Value if spectrogram is 3-dimensional.
      Defaults to 1.
      reduce_func (Callable, optional): the bahaviour for reducing the
      intensities between mz_value-|tol| and mz_value+|tol| to a single
      value. Must be a function that takes a sequence as input and outputs
      a number. By default, the values are summed. Defaults to sum.

  Returns:
      [np.ndarray]: numpy matrix with each element representing the
      ion intensity in this pixel. Can be easily plotted with matplotlib

  """
  tol = abs(tol)
  max_y = p.imzmldict["max count of pixels y"]
  max_x = p.imzmldict["max count of pixels x"]
  ims = [np.zeros((max_y, max_x)) for _ in range(0, len(mz_values))]
  for i, (x, y, z_) in enumerate(p.coordinates):
    if z_ == 0:
      UserWarning(("z coordinate = 0 present, if you're getting blank "
                   "images set getionimage(.., .., z=0)"))
    if z_ == z:
      mzs, ints = map(np.asarray, p.getspectrum(i))
      for j, mz_value in enumerate(mz_values):
        im = ims[j]
        min_i, max_i = _bisect_spectrum(mzs, mz_value, tol)
        im[y - 1, x - 1] = reduce_func(ints[min_i:max_i + 1])
  return ims


def main() -> None:
  # get command line arguments
  parser = argparse.ArgumentParser(__doc__)
  parser.add_argument("-i", required=True, help="imzML dataset folder path")
  parser.add_argument("-o", required=True, help="Output path")
  parser.add_argument("-p", required=True, nargs="+", help="Peaks to plot")
  args = parser.parse_args()

  # open the configuration file
  nano_biopsy_dir = dirname(dirname(realpath(__file__)))
  config_file = join(nano_biopsy_dir, "config.ini")
  config = configparser.ConfigParser()
  config.read(config_file)

  # set plot style
  sns.set_style("white")

  # get all imzML images names from the input folder
  images_names = [img for img in os.listdir(args.i) if img.endswith(".imzML")]

  # convert peaks to float from string
  peaks = [float(peak) for peak in args.p]

  # apply preprocessing
  mz_resolution = config["Default"]["MASS_RESOLUTION"]

  # loop over all imzML images name
  for img_name in images_names:
    # parse imzML image
    with ImzMLParser(os.path.join(args.i, img_name)) as p_l:
      # get imzML image name without extension
      title = os.path.splitext(img_name)[0]

      # get image for each peak
      ims_l = get_ion_image(p_l, peaks, tol=mz_resolution)

      # save image for each peak and combined peaks image
      for peak, im_l in zip(peaks + ["Combined"], ims_l + [sum(ims_l)]):
        # remove figure frame
        plt.figure(frameon=False)

        # remove axis
        plt.axis("off")

        # plot peak
        pc = plt.pcolormesh(im_l,
                            cmap="inferno",
                            vmin=im_l.min(),
                            vmax=im_l.max())

        # save plot
        plt.savefig(os.path.join(args.o, title + f"-{peak}" + ".png"),
                    transparent=True,
                    bbox_inches="tight",
                    pad_inches=0)

        # add color bar
        _, ax = plt.subplots(frameon=False)
        plt.colorbar(pc, ticks=np.linspace(im_l.min(), im_l.max(), 10), ax=ax)

        # remove peak plot
        ax.remove()

        # save color bar
        plt.savefig(os.path.join(args.o, title + f"-{peak}" + "-bar.png"),
                    transparent=True,
                    bbox_inches="tight",
                    pad_inches=0)

        # clear plot
        plt.close("all")


if __name__ == "__main__":
  main()
