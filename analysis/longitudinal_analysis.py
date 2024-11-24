import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy import ndimage
from skimage import filters
from skimage.morphology import disk
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from processing import TICNormalizer
from analysis.dhg_analysis import (
    process_spectral_data_to_image, merge_mzs_and_intensities
)
from typing import List, Tuple


def get_image_and_segmentation(
    raw_files: List[Path], normalizer: TICNormalizer, mass_resolution: float
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
  """Function to get images and segmentations from raw files

  Args:
      raw_files (List[Path]): List of paths to raw files.
      normalizer (TICNormalizer): Normalizer to use.
      mass_resolution (float): Mass resolution to use.

  Returns:
      Tuple[List[np.ndarray], List[np.ndarray]]: Tuple containing list of images
          and list of segmentations.

  """
  # Create a dictionary to store the images
  imgs = {}
  # Create a dictionary to store the segmented images
  tissue_segs = {}
  # Loop through the files
  for p in raw_files:
    # Get image and mzs
    img, mzs = process_spectral_data_to_image(p)
    # Apply normalization to the intensities
    img_norm = np.apply_along_axis(
        lambda intensities: (normalizer.normalize((mzs, intensities)))[1],
        axis=2, arr=img
    )
    # Merge mz channels with difference less than 0.02 as they are likely to be
    # the same mz
    mzs_merged, img_norm_merged = merge_mzs_and_intensities(
        mzs, img_norm, threshold=mass_resolution
    )
    # Define filters for the different mzs
    gray_matter_mzs_filter = (
        (mzs_merged >= 600.51 - mass_resolution * 2) &
        (mzs_merged <= 600.51 + mass_resolution * 2)
    )
    white_matter_mzs_filter = (
        (mzs_merged >= 888.62 - mass_resolution * 2) &
        (mzs_merged <= 888.62 + mass_resolution * 2)
    )
    tumour_mzs_filter = (
        (mzs_merged >= 682.59 - mass_resolution * 2) &
        (mzs_merged <= 682.59 + mass_resolution * 2)
    )
    # Smooth the image
    img_smooth = filters.gaussian(img_norm_merged, sigma=1)
    # Get the different mzs
    gray_matter_img = img_norm_merged[:, :, gray_matter_mzs_filter].sum(axis=2)
    white_matter_img = img_norm_merged[:, :,
                                       white_matter_mzs_filter].sum(axis=2)
    tumour_img = img_norm_merged[:, :, tumour_mzs_filter].sum(axis=2)
    # Smooth the images
    smooth_gray_matter_img = filters.gaussian(gray_matter_img, sigma=1)
    smooth_white_matter_img = filters.gaussian(white_matter_img, sigma=1)
    smooth_tumour_img = filters.gaussian(tumour_img, sigma=1)
    # Threshold the images
    gray_matter_img_thresh = smooth_gray_matter_img >= np.percentile(
        smooth_gray_matter_img.flatten(), 80
    )
    white_matter_img_thresh = smooth_white_matter_img >= np.percentile(
        smooth_white_matter_img.flatten(), 85
    )
    tumour_img_thresh = smooth_tumour_img >= np.percentile(
        smooth_tumour_img.flatten(), 85
    )
    # Combine the segmented images
    img_segmented = gray_matter_img_thresh + white_matter_img_thresh + tumour_img_thresh
    # Fill holes in the segmented image and store it
    tissue_segs[
        p.stem
    ] = filters.median(ndimage.binary_fill_holes(img_segmented), disk(5))
    # Store the image
    imgs[p.stem] = img_smooth
  # Return the images and segmentations
  return imgs, tissue_segs


def apply_clustering(imgs: List[np.ndarray],
                     tissue_segs: List[np.ndarray]) -> List[np.ndarray]:
  """ Function to apply clustering to the images.

  Args:
      imgs (List[np.ndarray]): List of images.
      tissue_segs (List[np.ndarray]): List of tissue and background 
          segmentations.

  Returns:
      List[np.ndarray]: List of segmentations.

  """
  # Create a dictionary to store the segmentations
  segmentations = {}
  # Loop through the images
  for key, img in tqdm(imgs.items()):
    # Get the pixels
    pixels = img[tissue_segs[key]]
    # Reshape the pixels
    clustering = AgglomerativeClustering(
        n_clusters=3, linkage='ward', compute_distances=True
    )
    # Perform hierarchical clustering
    clustering = clustering.fit(pixels)
    # Get the labels
    labels = np.zeros(img.shape[:2], dtype=np.uint8)
    labels[tissue_segs[key]] = clustering.labels_ + 1
    labels = labels.reshape(img.shape[:2])
    # Save the labels
    segmentations[key] = labels
  # Return the segmentations
  return segmentations


def plot_segmentation(
    figures_path: Path, segmentations: List[np.ndarray]
) -> None:
  """Function to plot the segmentations.

  Args:
      figures_path (Path): Path to save the figures.
      segmentations (List[np.ndarray]): List of segmentations.

  """
  # Define the color map
  cluster_labels_map = {"white": 1, "gray": 2, "tumour": 3}
  """
  # No image smoothing
  color_map = {
  '20240425_B4_T_0_S1(Day0S2)_Rep3_600-900_TopMax': {1: "white", 2: "gray", 3: "tumour"},
  '20240425_B4_T_1000_S1(Day0S3)_Rep2_600-900_TopMax': {1: "tumour", 2: "white", 3: "gray"},
  '20240510_B5_T_0_S1(Day0S1)_Rep3_600-900_TopMax': {1: "gray", 2: "white", 3: "tumour"},
  '20240510_B5_T_1000_S2(Day0S3)_Rep3_600-900_TopMax': {1: "tumour", 2: "gray", 3: "white"},
  '20240425_B4_T_Day0_Rep2_600-900_TopMax': {1: "gray", 2: "white", 3: "tumour"},
  '20240521_B8_R_T_1000_S1(Day0S2)_Rep2_600-900_TopMax': {1: "white", 2: "tumour", 3: "gray"},
  '20240514_B8_R_T_0_S1(Day0S1)_Rep3_600-900_TopMax': {1: "white", 2: "tumour", 3: "gray"},
    '20240425_B4_T_Day0_Rep3_600-900_TopMax': {1: "white", 2: "tumour", 3: "gray"},
    '20240510_B5_T_Day0_Rep1_600-900_TopMax': {1: "gray", 2: "tumour", 3: "white"},
    '20240510_B5_T_Day0_Rep3_600-900_TopMax': {1: "white", 2: "gray", 3: "tumour"},
    '20240523_B8_R_T_Day0_Rep2_600-900_TopMax': {1: "white", 2: "tumour", 3: "gray"},
    '20240523_B8_R_T_Day0_Rep1_600-900_TopMax': {1: "tumour", 2: "white", 3: "gray"}
  }

  # Gaussian smoothing with sigma=0.5
  color_map = {
  '20240425_B4_T_0_S1(Day0S2)_Rep3_600-900_TopMax': {1: "white", 2: "tumour", 3: "gray"},
  '20240425_B4_T_1000_S1(Day0S3)_Rep2_600-900_TopMax': {1: "white", 2: "tumour", 3: "gray"},
  '20240510_B5_T_0_S1(Day0S1)_Rep3_600-900_TopMax': {1: "white", 2: "gray", 3: "tumour"},
  '20240510_B5_T_1000_S2(Day0S3)_Rep3_600-900_TopMax': {1: "gray", 2: "white", 3: "tumour"},
  '20240425_B4_T_Day0_Rep2_600-900_TopMax': {1: "white", 2: "gray", 3: "tumour"},
  '20240521_B8_R_T_1000_S1(Day0S2)_Rep2_600-900_TopMax': {1: "gray", 2: "tumour", 3: "white"},
  '20240514_B8_R_T_0_S1(Day0S1)_Rep3_600-900_TopMax': {1: "white", 2: "tumour", 3: "gray"},
    '20240425_B4_T_Day0_Rep3_600-900_TopMax': {1: "tumour", 2: "gray", 3: "white"},
    '20240510_B5_T_Day0_Rep1_600-900_TopMax': {1: "white", 2: "tumour", 3: "gray"},
    '20240510_B5_T_Day0_Rep3_600-900_TopMax': {1: "gray", 2: "white", 3: "tumour"},
    '20240523_B8_R_T_Day0_Rep2_600-900_TopMax': {1: "gray", 2: "white", 3: "tumour"},
    '20240523_B8_R_T_Day0_Rep1_600-900_TopMax': {1: "white", 2: "gray", 3: "tumour"}
  }
  """
  # Define cluster labels map for gaussian smoothing with sigma=1
  color_map = {
      '20240425_B4_T_0_S1(Day0S2)_Rep3_600-900_TopMax':
      {1: "gray", 2: "white", 3:
       "tumour"}, '20240425_B4_T_1000_S1(Day0S3)_Rep2_600-900_TopMax':
      {1: "white", 2: "gray", 3:
       "tumour"}, '20240510_B5_T_0_S1(Day0S1)_Rep3_600-900_TopMax':
      {1: "white", 2: "gray", 3:
       "tumour"}, '20240510_B5_T_1000_S2(Day0S3)_Rep3_600-900_TopMax':
      {1: "white", 2: "gray", 3:
       "tumour"}, '20240425_B4_T_Day0_Rep2_600-900_TopMax':
      {1: "gray", 2: "white", 3:
       "tumour"}, '20240521_B8_R_T_1000_S1(Day0S2)_Rep2_600-900_TopMax':
      {1: "tumour", 2: "white", 3:
       "gray"}, '20240514_B8_R_T_0_S1(Day0S1)_Rep3_600-900_TopMax':
      {1: "gray", 2: "white", 3:
       "tumour"}, '20240425_B4_T_Day0_Rep3_600-900_TopMax':
      {1: "white", 2: "gray", 3:
       "tumour"}, '20240510_B5_T_Day0_Rep1_600-900_TopMax':
      {1: "white", 2: "tumour", 3:
       "gray"}, '20240510_B5_T_Day0_Rep3_600-900_TopMax':
      {1: "gray", 2: "tumour", 3:
       "white"}, '20240523_B8_R_T_Day0_Rep2_600-900_TopMax': {
           1: "gray", 2: "white", 3: "tumour"
       }, '20240523_B8_R_T_Day0_Rep1_600-900_TopMax':
      {1: "white", 2: "gray", 3: "tumour"}
  }
  # Create the colormap
  cmap = ListedColormap([(1, 1, 1, 0), "#3BAF42", "#5757F9", "#F94040"])
  # Loop through the segmentations
  for key, seg in segmentations.items():
    # Create a color image
    color_img = np.zeros(seg.shape, dtype=np.uint8)
    for cluster, color in color_map[key].items():
      color_img[seg == cluster] = cluster_labels_map[color]
    # Save the segmentation
    plt.imshow(color_img, cmap=cmap)
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(
        figures_path / f"{key}_segmentation.png", dpi=1200, bbox_inches="tight",
        transparent=True
    )
    plt.close()


def main():
  """Function containing main code"""
  # Define current folder using this file
  CWD = Path(os.path.dirname(os.path.abspath(__file__)))
  # Define folder that contains dataset
  LONGITUDINAL_PATH = CWD / ".." / ".." / "data" / "LONGITUDINAL"
  # Define folder that contains raw data
  RAW_DATA = LONGITUDINAL_PATH / "raw_txt"
  # Define path to save plots and results
  FIGURES_PATH = CWD / "longitudinal"
  FIGURES_PATH.mkdir(exist_ok=True, parents=True)
  # Define mass resolution of the data
  MASS_RESOLUTION = 0.02
  # Create a normalizer
  normalizer = TICNormalizer()
  # Get the aw files
  raw_files = list(Path(RAW_DATA).iterdir())
  # Get the images and segmentations
  imgs, tissue_segs = get_image_and_segmentation(
      raw_files, normalizer, MASS_RESOLUTION
  )
  # Apply clustering
  segmentations = apply_clustering(imgs, tissue_segs)
  # Plot the segmentations
  plot_segmentation(FIGURES_PATH, segmentations)


if __name__ == '__main__':
  main()
