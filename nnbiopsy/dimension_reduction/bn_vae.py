"""Batch Normalization Variational AutoEncoder

This module should be imported and contains the following:

    * Sampling - Sampling layer class for VAE.
    * Encoder - Encoder class for VAE.
    * Decoder - Decoder class for VAE.
    * BNVAE - Batch Normalization VAE class.

"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers


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
    # Get batch size
    batch = tf.shape(z_mean)[0]
    # Get layer dimensions
    dim = tf.shape(z_mean)[1]
    # Sample noise from normal distribution
    epsilon = tf.random.normal(shape=(batch, dim))
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
    super(Encoder, self).__init__(name=name, **kwargs)
    self.dense_proj = layers.Dense(intermediate_dim)
    self.batch_norm_proj = layers.BatchNormalization()
    self.relu_proj = layers.ReLU()
    self.dense_mean = layers.Dense(latent_dim)
    self.batch_norm_mean = layers.BatchNormalization()
    self.dense_log_var = layers.Dense(latent_dim)
    self.batch_norm_log_var = layers.BatchNormalization()
    self.sampling = Sampling()

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    """Override of call method. Calls the model on new inputs and returns
    the outputs as tensors.

    Args:
        inputs (tf.Tensor): Model inputs.

    Returns:
        tf.Tensor: Model outputs.

    """
    # Intermediate layer
    h = self.dense_proj(inputs)
    h = self.batch_norm_proj(h)
    h = self.relu_proj(h)

    # Mean layer
    z_mean = self.dense_mean(h)
    z_mean = self.batch_norm_mean(z_mean)

    # Log var layer
    z_log_var = self.dense_log_var(h)
    z_log_var = self.batch_norm_log_var(z_log_var)

    # Sampling layer
    z = self.sampling((z_mean, z_log_var))
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
    super(Decoder, self).__init__(name=name, **kwargs)
    self.dense_proj = layers.Dense(intermediate_dim)
    self.batch_norm_proj = layers.BatchNormalization()
    self.relu_proj = layers.ReLU()
    self.dense_output = layers.Dense(original_dim, activation="sigmoid")

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    """Override of call method. Calls the model on new inputs and returns
    the outputs as tensors.

    Args:
        inputs (tf.Tensor): Model inputs.

    Returns:
        tf.Tensor: Model outputs.

    """
    # Intermediate layer
    h = self.dense_proj(inputs)
    h = self.batch_norm_proj(h)
    h = self.relu_proj(h)

    # Reconstruction layer
    outputs = self.dense_output(h)
    return outputs


class BNVAE(Model):
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
    super(BNVAE, self).__init__(name=name, **kwargs)
    self.encoder = Encoder(latent_dim=latent_dim,
                           intermediate_dim=intermediate_dim)
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

    # Add KL divergence regularization loss
    kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) -
                                    tf.exp(z_log_var) + 1)
    self.add_loss(kl_loss)

    # Return decoder reconstructed output for reconstruction loss
    return reconstructed
