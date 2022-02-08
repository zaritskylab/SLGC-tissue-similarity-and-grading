""" TO DO """

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from pyimzml.ImzMLParser import ImzMLParser


def pca_components(p: ImzMLParser) -> pd.DataFrame:
  """
  Method to get a dataframe with pc components, their corresponding feature
  names, the explained_variance_ratio_ value and cumsum
  explained_variance_ratio_ for given imzml samples.

  Args:
      p_lst (List[ImzMLParser]): List of imzml samples

  Returns:
      pd.DataFrame: dataframe with pc components, their corresponding feature
      names, the explained_variance_ratio_ value and cumsum
      explained_variance_ratio_
  """
  # create pca object
  pca = PCA()

  # fit pca
  pca.fit(
      np.asarray([p.getspectrum(idx)[1] for idx, _ in enumerate(p.coordinates)
                 ]))

  # get first index that has a explained variance ratio cumsum greater
  # than threshold
  n_pcs = pca.components_.shape[0]

  # get the index of the most important feature on each component
  most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

  # set initial feature_names
  initial_feature_names = p.getspectrum(0)[0]

  # get the names by most important indexes
  most_important_names = [
      initial_feature_names[most_important[i]] for i in range(n_pcs)
  ]

  # LIST COMPREHENSION HERE AGAIN
  pcs = [f"PC{i}" for i in range(n_pcs)]

  # build the dataframe
  return pd.DataFrame({
      "PC": pcs,
      "Peak": most_important_names,
      "explained_variance_ratio": pca.explained_variance_ratio_,
      "explained_variance_ratio_cumsum": pca.explained_variance_ratio_.cumsum(),
  })


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
    pca_components(ImzMLParser(os.path.join(args.i, file))).to_csv(os.path.join(
        args.o, f"{file}-pca.csv"),
                                                                   index=False)
