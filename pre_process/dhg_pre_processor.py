"""DESI Human Glioma Pre Processor

This module enables the user to pre process all the images in the DESI Human
Glioma dataset.

This file can also be imported as a module and contains the following
functions:

    * main - module main function.
    * pre_process_dhg - Function to preprocess all the images in the DESI Human
    Glioma dataset.

"""

import argparse
import configparser
from os.path import dirname, realpath, join, listdir
from pathlib import Path
from tqdm import tqdm
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from normalization import TICNormalizer
from binning import MassResolutionBinning


def main() -> None:
  """Main function.
  
  """
  # get command line arguments
  parser = argparse.ArgumentParser(__doc__)
  parser.add_argument("-i",
                      required=True,
                      help="DESI Human Glioma dataset folder path")
  parser.add_argument("-o", required=True, help="Output folder")
  args = parser.parse_args()

  # create output folder if doesn't exist
  Path(args.o).mkdir(parents=True, exist_ok=True)

  # open the configuration file
  nano_biopsy_dir = dirname(dirname(realpath(__file__)))
  config_file = join(nano_biopsy_dir, "config.ini")
  config = configparser.ConfigParser()
  config.read(config_file)

  # apply preprocessing
  pre_process_dhg(args.i, args.o, float(config["DHG"]["MZ_START"]),
                  float(config["DHG"]["MZ_END"]),
                  float(config["DHG"]["MASS_RESOLUTION"]))


def pre_process_dhg(i_path: str, o_path: str, mz_start: float, mz_end: float,
                    mass_resolution: float) -> None:
  """Function to preprocess all the images in the DESI Human Glioma dataset.

  Args:
      i_path (str): path to folder containing the imzML images.
      o_path (str): output path.
      mz_start (float): mz spectrum range start.
      mz_end (float): mz spectrum range end.
      mass_resolution (float): mass spectrometry resolution.

  """
  # create normalizer
  normalizer = TICNormalizer()

  # create binning object
  binning = MassResolutionBinning(mz_start, mz_end, mass_resolution)

  # get all imzml images names
  images_names = [
      join(o_path, file) for file in listdir(i_path) if file.endswith(".imzML")
  ]

  # loop over each imzML image name
  for img_name in tqdm(images_names, desc="Images Loop"):
    # create a new imzML preprocessed image, because we have the same mz value
    # after binning we can use mode="continuous" to save mz values once
    with ImzMLWriter(join(o_path, img_name), mode="continuous") as writer:
      # parse the imzML image
      with ImzMLParser(join(i_path, img_name)) as p:
        # loop over each spectrum in imzML image
        for idx, (x, y, z) in tqdm(enumerate(p.coordinates),
                                   total=len(p.coordinates),
                                   desc="Pixels Loop"):

          # apply normalization and bining to spectrum
          mzs, intensities = binning.bin(
              normalizer.normalize(p.getspectrum(idx)))

          # write processed spectrum
          writer.addSpectrum(mzs, intensities, (x, y, z))


if __name__ == "__main__":
  main()
