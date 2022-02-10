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
from typing import List
from pathlib import Path
from tqdm import tqdm
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter


def main() -> None:
  """
  main function
  """
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

  # read bounding boxes csv
  bb_df = pd.read_csv(args.b)

  # get all imzML images names
  images_names = bb_df.file_name.unique()

  # loop over each imzML image name
  for img_name in tqdm(images_names, desc="Images Loop"):
    # get all samples numbers in image
    samples_nums_l = bb_df.loc[(bb_df.file_name == img_name),
                               "sample_num"].to_list()

    # get samples type
    s_type = "r.imzML" if "r" in img_name else "s.imzML"

    # create path for each sample new imzML image
    samples_paths_l = [
        os.path.join(args.o, sample + "-" + s_type) for sample in samples_nums
    ]

    # get bounding box for each sample in image
    samples_bb_l = bb_df.loc[(bb_df.file_name == img_name)]

    # parse the imzML image
    with ImzMLParser(os.path.join(args.i, img_name + ".imzML")) as p_l:
      # crop the imzML image
      crop_image(p_l, samples_nums_l, samples_paths_l, samples_bb_l)


def crop_image(p: ImzMLParser, samples_nums: List[int],
               samples_paths: List[str], samples_bb: pd.DataFrame) -> None:
  """
  Function to crop and imzML image to its samples. 

  Args:
      p (ImzMLParser): imzML image.
      samples_nums (List[int]): list of samples numbers in the same order of 
      samples_paths.
      samples_paths (List[str]): paths to write each new imzML sample image.
      samples_bb (pd.DataFrame): samples inside the image.
  """
  # open a new imzML image for each path in samples_paths
  writers = [ImzMLWriter(path) for path in samples_paths]

  # loop over each spectrum in imzML image
  for idx, (x, y, z) in tqdm(enumerate(p.coordinates),
                             total=len(p.coordinates),
                             desc="Pixels Loop"):

    # get spectrum mz values and intensities
    mzs, intensities = p.getspectrum(idx)

    # find sample in image that this spectrum belongs to
    sample = samples_bb.loc[(samples_bb.x_min <= x) & (samples_bb.x_max >= x) &
                            (samples_bb.y_min <= y) & (samples_bb.y_max >= y)]

    # get sample index
    sample_idx = samples_nums.index(sample.sample_num.iat[0])

    # write spectrum to sample image
    writers[sample_idx].addSpectrum(
        mzs, intensities, (x - sample.x_min.iat[0], y - sample.y_min.iat[0], z))

  # close all writers
  for writer in writers:
    writer.close()


if __name__ == "__main__":
  main()
