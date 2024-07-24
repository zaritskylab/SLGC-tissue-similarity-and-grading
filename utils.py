"""Mass spectrometry utils
This module should be imported and contains the following:
    
    * read_msi - Function to read a continuos msi.

"""
import numpy as np
from typing import Tuple
from pyimzml.ImzMLParser import ImzMLParser


def read_msi(p: ImzMLParser) -> Tuple[np.ndarray, np.ndarray]:
  """
  Function to read a continuos imzML parser object into a numpy array.

  Args:
      p (ImzMLParser): The imzML parser.
  
  Returns:
      Tuple[np.ndarray, np.ndarray]: Numpy 3D matrix where y coordinate
          (axis=0), x coordinate (axis=1), intensities values (axis=2)
          and continuos mzs values.

  """
  # Get shape of mzs values
  max_z = p.mzLengths[0]
  # Get shape of y axis
  max_y = p.imzmldict["max count of pixels y"]
  # Get shape of x axis
  max_x = p.imzmldict["max count of pixels x"]
  # Create empty numpy 3D matrix
  msi = np.zeros((max_y, max_x, max_z))
  # Loop over each coordinate and add to 3D matrix
  for i, (x, y, _) in enumerate(p.coordinates):
    # Get mzs and intensities
    mzs, ints = p.getspectrum(i)
    # Add intensities to x,y coordinate
    msi[y - 1, x - 1, :] = ints
  return mzs, msi

def get_mean_spectra(p):
  """
  Function to calculate mean spectra of processed imzML parser 

  Args:
    p (ImzMLParser): The imzML parser.
  
  Returns:
    Tuple[np.ndarray, np.ndarray]: Mean spectra first element 
        is the mz values array of spectra and second element 
        is the intensity values array of spectra.

  """
  # Initialize a list to store all spectra
  all_mzs = []
  all_intensities = []
  
  # Iterate over all spectra in the file
  for idx, _ in enumerate(p.coordinates):
      mzs, intensities = p.getspectrum(idx)
      all_mzs.append(mzs)
      all_intensities.append(intensities)
      
  # Convert lists to numpy arrays
  all_mzs = np.concatenate(all_mzs)
  all_intensities = np.concatenate(all_intensities)
  
  # Find unique m/z values and their indices in the original array
  unique_mzs, indices = np.unique(all_mzs, return_inverse=True)
  
  # Accumulate intensities at each unique m/z
  summed_intensities = np.bincount(indices, weights=all_intensities)
  
  # Calculate the mean by dividing by the number of spectra
  mean_intensities = summed_intensities / len(p.coordinates)
  
  return unique_mzs, mean_intensities

def get_mean_spectra_old(p: ImzMLParser):
  """
  Function to calculate mean spectra of processed imzML parser 

  Args:
    p (ImzMLParser): The imzML parser.
  
  Returns:
    Tuple[np.ndarray, np.ndarray]: Mean spectra first element 
        is the mz values array of spectra and second element 
        is the intensity values array of spectra.

  """
  # Initialize a dictionary to store the sum of all spectra
  sum_spectra = {}
  count = 0
      
  # Iterate over all spectra in the file
  for idx, (x,y,z) in enumerate(p.coordinates):
    mzs, intensities = p.getspectrum(idx)
    # Aggregate spectra by adding intensities at each m/z
    for m, i in zip(mzs, intensities):
      if m in sum_spectra:
          sum_spectra[m] += i
      else:
          sum_spectra[m] = i
    count += 1
  # Calculate the mean spectrum
  mean_spectrum = {m: i / count for m, i in sum_spectra.items()}
  # Convert the mean spectrum to sorted lists of m/z and intensities
  mzs, intensities = zip(*sorted(mean_spectrum.items()))
  return np.asarray(mzs),  np.asarray(intensities)