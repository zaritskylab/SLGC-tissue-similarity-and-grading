""" This module run a pre processing pipline on the imzML-DESI dataset """

import os
import shutil
import configparser
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from spectrum_utils import SpectrumUtils

if __name__ == "__main__":
  # current directory of this python file
  current_directory = os.path.dirname(os.path.realpath(__file__))

  # path to configuration file
  config_file = os.path.join(current_directory, "config.ini")

  # path to folder containing the imzML-DESI files
  imzml_folder = os.path.join(os.path.dirname(current_directory), "imzml-DESI")

  # path to save the preprocessed imzML-DESI files
  preprocessed_imzml_folder = os.path.join(os.path.dirname(current_directory),
                                           "preprocessed-imzml-DESI")

  # delete existing preprocessed imzML-DESI files
  if os.path.exists(preprocessed_imzml_folder):
    shutil.rmtree(preprocessed_imzml_folder)

  # create preprocessed imzML-DESI directory
  os.mkdir(preprocessed_imzml_folder)

  # get configuration
  config = configparser.ConfigParser()
  config.read(config_file)

  # get mz spectrum range
  mz_start = float(config["DEFAULT"]["MZ_START"])
  mz_end = float(config["DEFAULT"]["MZ_END"])

  # get spectrum mass resolution
  mass_resolution = config["DEFAULT"]["MASS_RESOLUTION"]

  # read bounding boxes csv
  bb_df = pd.read_csv(os.path.join(imzml_folder, "imzML-DESI-BB.csv"))

  # get all imzML files
  files = bb_df.file_name.unique()

  # create spectrum bins using spectrum lowest and largest
  # mz value and spectrum mass resultion
  bins = np.around(np.arange(mz_start, mz_end, float(mass_resolution) / 2), 5)

  # loop over each imzML file
  for file in tqdm(files, desc="Images Loop"):
    # get all samples in file
    samples = bb_df.loc[(bb_df.file_name == file), "sample_num"].to_list()

    # get sample type
    s_type = "r.imzML" if "r" in file else "s.imzML"

    # open imzML file for each sample in file
    # because we have the same mz value after binning we can
    # use mode="continuous" to save mz values once
    writers = [
        ImzMLWriter(os.path.join(preprocessed_imzml_folder,
                                 sample + "-" + s_type),
                    mode="continuous") for sample in samples
    ]

    # parse the imzML file
    with ImzMLParser(os.path.join(imzml_folder, file + ".imzML")) as p:
      # loop over each spectrum in imzML file
      for idx, (x, y, z) in tqdm(enumerate(p.coordinates),
                                 total=len(p.coordinates),
                                 desc="Pixels Loop"):

        # apply normalization and bining to spectrum
        mzs, intensities = SpectrumUtils().bining_spectrum(
            SpectrumUtils().normalize_spectrum(p.getspectrum(idx)), bins)

        # find sample in file that this spectrum belongs to
        sample = bb_df.loc[(bb_df.file_name == file) & (bb_df.x_min <= x) &
                           (bb_df.x_max >= x) & (bb_df.y_min <= y) &
                           (bb_df.y_max >= y)]

        # write processed spectrum
        writers[samples.index(sample.sample_num.iat[0])].addSpectrum(
            mzs, intensities,
            (x - sample.x_min.iat[0], y - sample.y_min.iat[0], z))

    # close all writers
    for writer in writers:
      writer.close()
