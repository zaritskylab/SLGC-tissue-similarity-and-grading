""" DESI Human Glioma Cropper

Each image in the DESI Human Glioma dataset contains multiple samples.
This module enables the user to crop all the images to create a
single image for each sample.

This file can also be imported as a module and contains the following
functions:

    * main - the main function of the script
    * crop_image - function to crop an imzML image to its samples
"""

import os
import argparse
import pandas as pd
import numpy as np
from typing import List
from pathlib import Path
from tqdm import tqdm
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter


def crop_image(p: ImzMLParser, samples_names: List[str], samples_type: str,
               samples_x_min: np.ndarray, samples_y_min: np.ndarray,
               samples_x_max: np.ndarray, samples_y_max: np.ndarray,
               output_path: str) -> None:
  """
  Function to crop an imzML image to its samples. For example
  if we have the imzML image 'HG 14-13-s' then we want to
  crop it to '14-s' and '13-s'.

  Args:
      p (ImzMLParser): imzML image parser object.
      samples_names (List[str]): list of samples names.
      samples_type (str): type of sample. should be 'r' or 's'.
      samples_x_min (np.ndarray): list of boundary box minimum x value. index
      should correspond to the sample in samples_names.
      samples_y_min (np.ndarray): list of boundary box minimum y value. index
      should correspond to the sample in samples_names.
      samples_x_max (np.ndarray): list of boundary box maximum x value. index
      should correspond to the sample in samples_names.
      samples_y_max (np.ndarray): list of boundary box maximum y value. index
      should correspond to the sample in samples_names.
      output_path (str): path to save cropped samples images.
  """
  # create path for each sample new imzML image
  samples_paths = [
      os.path.join(output_path, sample + "-" + samples_type)
      for sample in samples_names
  ]

  # open a new imzML image for each path in samples_paths
  writers = np.asarray([ImzMLWriter(path) for path in samples_paths])

  # loop over each spectrum in imzML image
  for idx, (x, y, z) in tqdm(enumerate(p.coordinates),
                             total=len(p.coordinates),
                             desc="Pixels Loop"):

    # get spectrum mz values and intensities
    mzs, intensities = p.getspectrum(idx)

    # find sample index in image that this spectrum belongs to
    sample_idx = ((samples_x_min <= x) & (samples_x_max >= x) &
                  (samples_y_min <= y) & (samples_y_max >= y))

    # write spectrum to sample image
    writers[sample_idx].addSpectrum(
        mzs, intensities,
        (x - samples_x_min[sample_idx][0], y - samples_y_min[sample_idx][0], z))

  # close all writers
  for writer in writers:
    writer.close()


def crop_dhg(i_path: str, o_path: str, bb_df: pd.DataFrame) -> None:
  """
  Function to crop all the images in the DESI Human Glioma dataset. Each image
  contains multiple samples, this function crops all the images to create a
  single image for each sample.

  Args:
      i_path (str): path to folder containing the imzML images.
      o_path (str): output path.
      bb_df (pd.DataFrame): dataframe containing boundary box for each sample.
      should contain the following columns - file_name (the image file name
      containing the sample and can be found in the i_path),
      sample_name (the sample name in the image), x_min (the boundary box
      minimum x value), x_max (the boundary box maximum x value), y_min (the
      boundary box minimum y value), y_max (the boundary box maximum y value)
  """

  # get all imzML images names
  images_names = bb_df.file_name.unique()

  # loop over each imzML image name
  for img_name in tqdm(images_names, desc="Images Loop"):
    # get all samples names in image
    samples_names_l = bb_df.loc[(bb_df.file_name == img_name),
                                "sample_name"].to_list()

    # get samples type
    samples_type_l = "r.imzML" if "r" in img_name else "s.imzML"

    # get bounding box for each sample in image
    samples_bb_l = bb_df.loc[(bb_df.file_name == img_name)]

    # parse the imzML image
    with ImzMLParser(os.path.join(i_path, img_name + ".imzML")) as p_l:
      # crop the imzML image
      crop_image(p_l, samples_names_l, samples_type_l,
                 samples_bb_l.x_min.to_numpy(), samples_bb_l.y_min.to_numpy(),
                 samples_bb_l.x_max.to_numpy(), samples_bb_l.y_max.to_numpy(),
                 o_path)


if __name__ == "__main__":
  # get command line arguments
  parser = argparse.ArgumentParser(__doc__)
  parser.add_argument("-i",
                      required=True,
                      help="DESI Human Glioma dataset folder path")
  parser.add_argument("-o", required=True, help="Output folder")
  parser.add_argument("-b", required=True, help="Bounding boxes csv file path")
  args = parser.parse_args()

  # create output folder if doesn't exist
  Path(args.o).mkdir(parents=True, exist_ok=True)

  # crop all
  crop_dhg(args.i, args.o, pd.read_csv(args.b))
