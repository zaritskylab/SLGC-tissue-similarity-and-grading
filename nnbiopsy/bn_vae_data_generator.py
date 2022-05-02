"""Data Generator.

This module should be imported and contains the following:

    * DataGenerator - Data generator class.

"""

import numpy as np
from typing import Callable, Tuple
from tensorflow.keras import utils


class DataGenerator(utils.Sequence):
  """Data Generator for Keras.

  """

  def __init__(self,
               ids: np.ndarray,
               load_sample_fn: Callable,
               batch_size: int = 32,
               dim: int = 92000,
               shuffle: bool = True) -> None:
    """Initialization method.

    Args:
        ids (np.ndarray): Sample ids to get sample using load_sample_fn.
        load_sample_fn (Callable): Function to get sample using id
        batch_size (int, optional): Generator batch size. Defaults to 32.
        dim (int, optional): Sample dimension. Defaults to 92000.
        shuffle (bool, optional): Boolean to indicate whether to shuffle
        data at the end of each epoch. Defaults to True.

    """
    # Save ids
    self.ids = ids
    # Save load_sample_fn
    self.load_sample_fn = load_sample_fn
    # Save batch_size
    self.batch_size = batch_size
    # Save dim
    self.dim = dim
    # Save shuffle
    self.shuffle = shuffle
    # Trigger on_epoch_end at the beginning
    self.on_epoch_end()

  def on_epoch_end(self) -> None:
    """ Method to update indexes after each epoch.

    """
    # Get index of each id
    self.indexes = np.arange(len(self.ids))
    if self.shuffle == True:
      # Shuffle  indexes
      np.random.shuffle(self.indexes)

  def __len__(self) -> int:
    """Method to denote the number of batches per epoch

    Returns:
        int: number of batches per epoch

    """
    return int(np.ceil(len(self.ids) / self.batch_size))

  def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
    """Method to generate one batch of data.

    Args:
        index (int): batch index.

    Returns:
        Tuple[np.ndarray, np.ndarray]: batch samples and labels
    """
    # Generate indexes of the batch
    batch_indexes = self.indexes[(index * self.batch_size):((index + 1) *
                                                            self.batch_size)]
    # Get ids of the batch
    batch_ids = [self.ids[k] for k in batch_indexes]
    print(len(batch_ids))
    # Generate data
    batch_x, batch_y = self.__data_generation(batch_ids)
    return batch_x, batch_y

  def __data_generation(self,
                        batch_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Method to generate one batch of data with samples from ids.

    Args:
        batch_ids (np.ndarray): _description_

    Returns:
        Tuple[np.ndarray, np.ndarray]: batch samples and labels
    """
    # Initialize empty arrays
    batch_x = np.empty((self.batch_size, self.dim))
    # Generate data
    for i, ID in np.ndenumerate(batch_ids):
      # Store sample
      batch_x[i,] = self.load_sample_fn(ID)
    return batch_x, batch_x
