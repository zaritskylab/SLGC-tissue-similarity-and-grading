""" This module crops all the samples in the imzML-DESI dataset """

import os
import argparse
import pandas as pd
from tqdm import tqdm
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter

if __name__ == "__main__":
  # get command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("-i",
                      required=True,
                      help="Path to folder containing imzML-DESI images")
  parser.add_argument("-o",
                      required=True,
                      help="Path to output folder of cropped samples")
  parser.add_argument("-bounding",
                      required=True,
                      help="Path to bounding boxes csv file")
  args = parser.parse_args()

  # read bounding boxes csv
  bb_df = pd.read_csv(args.bounding)

  # get all imzML images
  files = bb_df.file_name.unique()

  # loop over each imzML image
  for file in tqdm(files, desc="Images Loop"):
    # get all samples in image
    samples = bb_df.loc[(bb_df.file_name == file), "sample_num"].to_list()

    # get sample type
    s_type = "r.imzML" if "r" in file else "s.imzML"

    # open imzML image for each sample in image
    writers = [
        ImzMLWriter(os.path.join(args.o, sample + "-" + s_type)) for sample in samples
    ]

    # parse the imzML image
    with ImzMLParser(os.path.join(args.i, file + ".imzML")) as p:
      # loop over each spectrum in imzML image
      for idx, (x, y, z) in tqdm(enumerate(p.coordinates),
                                 total=len(p.coordinates),
                                 desc="Pixels Loop"):

        # get mz values and intensities
        mzs, intensities = p.getspectrum(idx)

        # find sample in image that this spectrum belongs to
        sample = bb_df.loc[(bb_df.file_name == file) & (bb_df.x_min <= x) &
                           (bb_df.x_max >= x) & (bb_df.y_min <= y) &
                           (bb_df.y_max >= y)]

        # write spectrum to sample image
        writers[samples.index(sample.sample_num.iat[0])].addSpectrum(
            mzs, intensities,
            (x - sample.x_min.iat[0], y - sample.y_min.iat[0], z))

    # close all writers
    for writer in writers:
      writer.close()
