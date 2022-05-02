"""Batch Normalization Variational Autoencoder

This module should be imported and contains the following:

    * Sampling - Sampling layer class for VAE.
    * Encoder - Encoder class for VAE.
    * Decoder - Decoder class for VAE.
    * BNVAE - Batch Normalization VAE class.

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K


class Sampling(layers.Layer):
  """Sampling layer for VAE, Uses (z_mean, z_log_var) to sample z
  (vector encoding).

  """

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    """Override of call method. Calls the model on new inputs and returns
    the outputs as tensors.

    Args:
        inputs (tf.Tensor): Model inputs.

    Returns:
        tf.Tensor: Model outputs.

    """
    # Unpack z_mean, z_log_var
    z_mean, z_log_var = inputs
    # Sample noise from normal distribution
    epsilon = K.random_normal(shape=tf.shape(z_mean))
    # return re-parameterization
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
  """Encoder for VAE.

  """

  def __init__(self,
               latent_dim: int,
               intermediate_dim: int,
               name: str = "encoder",
               **kwargs) -> None:
    """Initialization method.

    Args:
        latent_dim (int): Encoder latent dimension size.
        intermediate_dim (int): Encoder intermediate dimension size.
        name (str, optional): Encoder name. Defaults to "encoder".
    
    """
    # Super class initialization
    super(Encoder, self).__init__(name=name, **kwargs)
    # Define intermidiate dense layer
    self.dense_proj = layers.Dense(intermediate_dim)
    # Define intermidiate batch normalization layer
    self.dense_batch_norm = layers.BatchNormalization()
    # Define intermidiate relu layer
    self.dense_relu = layers.ReLU()
    # Define mean dense layer
    self.dense_mean = layers.Dense(latent_dim)
    # Define variance dense layer
    self.dense_log_var = layers.Dense(latent_dim)
    # Define sampleing layer
    self.sampling = Sampling()

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    """Override of call method. Calls the model on new inputs and returns
    the outputs as tensors.

    Args:
        inputs (tf.Tensor): Model inputs.

    Returns:
        tf.Tensor: Model outputs.

    """
    # Apply intermidiate dense layer on inputs
    x = self.dense_proj(inputs)
    # Apply intermidiate batch normalization layer on
    # intermidiate dense layer outputs
    x = self.dense_batch_norm(x)
    # Apply intermidiate relu layer on intermidiate
    # batch normalization layer outputs
    x = self.dense_relu(x)
    # Apply mean dense layer on intermidiate
    # relu layer outputs
    z_mean = self.dense_mean(x)
    # Apply variance dense layer on intermidiate
    # relu layer outputs
    z_log_var = self.dense_log_var(x)
    # Apply sampling layer on latent mean
    # dense layer outputs and latent
    # variance dense layer outputs
    z = self.sampling((z_mean, z_log_var))
    # Return z_mean, z_log_var, z
    return z_mean, z_log_var, z


class Decoder(layers.Layer):
  """Decoder for VAE.

  """

  def __init__(self,
               original_dim: int,
               intermediate_dim: int,
               name: str = "decoder",
               **kwargs) -> None:
    """Initialization method.

    Args:
        original_dim (int): Decoder original dimension size.
        intermediate_dim (int): Decoder intermediate dimension size.
        name (str, optional): Decoder name. Defaults to "decoder".
    
    """
    # Super class initialization
    super(Decoder, self).__init__(name=name, **kwargs)
    # Define intermidiate dense layer
    self.dense_proj = layers.Dense(intermediate_dim)
    # Define reconstruction dense layer
    self.dense_output = layers.Dense(original_dim, activation="sigmoid")

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    """Override of call method. Calls the model on new inputs and returns
    the outputs as tensors.

    Args:
        inputs (tf.Tensor): Model inputs.

    Returns:
        tf.Tensor: Model outputs.

    """
    # Apply intermidiate dense layer on inputs
    x = self.dense_proj(inputs)
    # Return reconstruction dense layer of intermidiate layer outputs
    return self.dense_output(x)


class BNVAE(keras.Model):
  """Batch Normalization VAE class.

  """

  def __init__(self,
               original_dim: int,
               intermediate_dim: int,
               latent_dim: int,
               name="autoencoder",
               **kwargs) -> None:
    """Initialization method.

    Args:
        original_dim (int): AutoEncoder original dimension size.
        intermediate_dim (int): AutoEncoder intermediate dimension size.
        latent_dim (int): AutoEncoder latent dimension size.
        name (str, optional): AutoEncoder name. Defaults to "autoencoder".
    """
    # Super class initialization
    super(BNVAE, self).__init__(name=name, **kwargs)
    # Save original dimension size
    self.original_dim = original_dim
    # Define encoder
    self.encoder = Encoder(latent_dim=latent_dim,
                           intermediate_dim=intermediate_dim)
    # Define decoder
    self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    """Override of call method. Calls the model on new inputs and returns
    the outputs as tensors.

    Args:
        inputs (tf.Tensor): Model inputs.

    Returns:
        tf.Tensor: Model outputs.

    """
    # Unpack z_mean, z_log_var, z
    z_mean, z_log_var, z = self.encoder(inputs)
    # Get decoder reconstruction
    reconstructed = self.decoder(z)
    # Define KL divergence regularization loss
    kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) -
                                    tf.exp(z_log_var) + 1)
    # Add KL divergence regularization loss
    self.add_loss(kl_loss)
    # Return decoder reconstructed output
    return reconstructed
