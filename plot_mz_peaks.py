""" Module to create images of mz peaks plot for each imzML file"""

import argparse
import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pyimzml.ImzMLParser import ImzMLParser, _bisect_spectrum
from typing import List, Callable

sns.set_style("white")


def getionimage(p: ImzMLParser,
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

  # define peaks
  peaks = [794.5, 834.5, 888.6]

  # loop over all imzML files
  for file in files[45:]:
    # parse imzML file
    with ImzMLParser(os.path.join(args.i, file)) as p_l:
      # get file name of imzML file
      title = os.path.splitext(file)[0]

      # get image for each peak
      ims_l = getionimage(p_l, peaks, tol=0.025)

      # save image for each peak
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
        fig, ax = plt.subplots(frameon=False)
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
