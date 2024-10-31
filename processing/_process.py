"""Mass spectrometry processing
This module should be imported and contains the following:
    
    * process_spectras - Function to process msi.
    * aligned_representation - Function to to create aligned representation 
          for msi spectras.
    * common_representation - Function to to create common representation for
          msi spectras.
    * meaningful_signal - Function to create meaningful signal scaler for msi
          spectras.

"""

import os
import numpy as np
from typing import List
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from processing import (
    EqualWidthBinning, ReferenceLockMass, TICNormalizer, MeanSegmentation, 
    ZScoreCorrection
)
from utils import read_msi, get_mean_spectra
from tqdm import tqdm


def aligned_representation(input_path: str, output_path: str, 
                          original_lock_mass_position: float, 
                          tol: float = 0.3) -> None:
  """Function to create aligned representation for msi spectras. Function 
        creates a new msi file in the given folder.

  Args:
    input_path (str): Path to imzML file that needs to be aligned.
    output_path (str): Path to folder for saving output.
    original_lock_mass_position (float): The original peak value for expected.
    tol (float, optional): Tolerance for searching the shifted peak from expected
        peak. Defaults to 0.3.

  """
  # Parse the MSI file
  with ImzMLParser(input_path) as reader:
    # Get lock mass object
    print("started mean spectra calc")
    mean_spectra = get_mean_spectra(reader)
    lock_mass = ReferenceLockMass(original_lock_mass_position, mean_spectra, tol)
    print(input_path, lock_mass.scale_ratio, lock_mass.diff)
    # Create a new MSI for aligned data
    with ImzMLWriter(output_path, mode="processed") as writer:
        print("started aligning msi")
        # Iterate over all spectra in the file
        for idx, (x,y,z) in enumerate(reader.coordinates):
          # Apply lock mass
          aligned_mzs, intensities = lock_mass.lock_mass(
            reader.getspectrum(idx)
          )
          # Write spectra to new MSI with coordinate
          writer.addSpectrum(aligned_mzs, intensities, (x, y, z))


def common_representation(
    input_path: str, output_path: str, x_min: int, x_max: int, y_min: int,
    y_max: int, mz_start: int, mz_end: int, mass_resolution: float
) -> None:
  """Function to create common representation for msi spectras. Function 
        creates a new msi file in the given folder.

  Args:
      input_path (str): Path to imzML file that needs to be processed.
      output_path (str): Path to folder for saving output.
      x_min (int): X minimum coordinate of the the tissue in the input.
      x_max (int): X maximum coordinate of the the tissue in the input.
      y_min (int): Y minimum coordinate of the the tissue in the input.
      y_max (int): Y maximum coordinate of the the tissue in the input.
      mz_start (int): The start value of the mz range.
      mz_end (int): The end value of the mz range.
      mass_resolution (float): The mass resolution.

  """
  # Get normalizer object
  normalizer = TICNormalizer()
  # Get binning object
  binning = EqualWidthBinning(mz_start, mz_end, mass_resolution)
  # Create process pipe
  process_pipe = (
      lambda mzs, intensities:
      (binning.bin(normalizer.normalize((mzs, intensities))))
  )
  # Parse the MSI file containing ROI
  with ImzMLParser(input_path) as reader:
    # Create a new MSI for ROI. because we apply binning
    # we can use mode="continuous"
    with ImzMLWriter(
        os.path.join(output_path, "common_representation.imzML"),
        mode="continuous"
    ) as writer:
      # Loop over each spectra in MSI
      for idx, (x, y, z) in enumerate(reader.coordinates):
        # Check if spectra is in ROI boundaries
        if ((x_min <= x <= x_max) & (y_min <= y <= y_max)):
          # Read spectra from MSI
          raw_mzs, raw_intensities = reader.getspectrum(idx)
          # Apply processing pipe
          preprocessed_mzs, preprocessed_intensities = process_pipe(
              raw_mzs, raw_intensities
          )
          # Write spectra to new MSI with relative coordinate
          writer.addSpectrum(
              preprocessed_mzs, preprocessed_intensities,
              (x - x_min + 1, y - y_min + 1, z)
          )


def meaningful_signal(
    input_path: str, output_path: str, representative_peaks: List[float],
    mass_resolution: float
):
  """Function to create meaningful signal for msi spectras. Function 
      creates a new msi file in the given folder and a segmentation file.
  Args:
      input_path (str): Path to continuos imzML file that needs to be
              processed.
      output_path (str): Path to folder for saving output.
      representative_peaks (List[float]): Representative peaks (mz values) 
          for getting a single channel image.
      mass_resolution (float): Mass resolution of the msi.
  """
  # Parse the msi file
  with ImzMLParser(input_path) as reader:
    # Get full msi
    mzs, img = read_msi(reader)
    # Segment image
    segment_img = MeanSegmentation(mzs, representative_peaks,
                                   mass_resolution).segment(img)
    # Save segmentation
    np.save(os.path.join(output_path, 'segmentation.npy'), segment_img)

    # Apply image correction
    zscore_img = ZScoreCorrection().correct(img, segment_img)

    # Open writer
    with ImzMLWriter(
        os.path.join(output_path, "meaningful_signal.imzML"), mode="continuous"
    ) as writer:
      # Save zscore image
      for _, (x, y, z) in enumerate(reader.coordinates):
        writer.addSpectrum(mzs, zscore_img[y - 1, x - 1], (x, y, z))


def process(
    input_path: str, output_path: str,
    x_min: int, x_max: int, y_min: int, y_max: int, mz_start: int, 
    mz_end: int, mass_resolution: float, representative_peaks: List[float]
) -> None:
  """Function to process msi.

  Args:
    input_path (str): Path to imzML file that needs to be processed.
    output_path (str): Path to folder for saving output.
    x_min (int): X minimum coordinate of the the tissue in the input.
    x_max (int): X maximum coordinate of the the tissue in the input.
    y_min (int): Y minimum coordinate of the the tissue in the input.
    y_max (int): Y maximum coordinate of the the tissue in the input.
    mz_start (int): The start value of the mz range.
    mz_end (int): The end value of the mz range.
    mass_resolution (float): The mass resolution.
    representative_peaks (List[float]): Representative peaks (mz values) 
        for getting a single channel image.

  """

  ""
  # Create common representation
  common_representation(
      input_path, output_path,
      x_min, x_max, y_min, y_max, mz_start, mz_end, mass_resolution / 2
  )

  # Create meaningful signal
  meaningful_signal(
      os.path.join(output_path, "common_representation.imzML"), output_path,
      representative_peaks, mass_resolution
  )
