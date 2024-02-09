import torch
import torch.nn as nn
import pandas as pd

from genrisk.generation.base import TorchGenerator
from genrisk.generation.gan import GANModule
from genrisk.generation.tcn import TCNModule


class _TCNGenerator(nn.Module):
    def __init__(self, latent_dim, condition_dim, kernel_size, hidden_dim, target_dim, num_layers):
        super().__init__()
        self.tcn = TCNModule(
            input_size=latent_dim + condition_dim,
            kernel_size=kernel_size,
            num_filters=hidden_dim,
            num_layers=num_layers,
            dilation_base=2,
            weight_norm=False,
            target_size=hidden_dim,
            dropout=0,
        )
        self.projection = nn.Linear(hidden_dim, target_dim)

    def forward(self, z, cond):
        z_cond = torch.cat([z, cond], dim=2)
        output = self.tcn(z_cond)
        fake = self.projection(output)
        return fake


class _TCNDiscriminator(nn.Module):
    def __init__(self, target_dim, condition_dim, kernel_size, hidden_dim, num_layers):
        super().__init__()
        self.tcn = TCNModule(
            input_size=target_dim + condition_dim,
            kernel_size=kernel_size,
            num_filters=hidden_dim,
            num_layers=num_layers,
            dilation_base=2,
            weight_norm=False,
            target_size=hidden_dim,
            dropout=0,
        )
        self.projection = nn.Linear(hidden_dim, 1)

    def forward(self, taget, cond):
        target_cond = torch.cat([taget, cond], dim=2)
        output = self.tcn(target_cond)
        logits = self.projection(output)
        return logits, output


class TCNGAN(TorchGenerator):
    """GAN generator based on TCN module."""
    def __init__(
            self, 
            target_columns: list[str], 
            conditional_columns: list[str],
            window_size: int=10, 
            batch_size: int=16, 
            num_epochs: int=1,
            verbose: bool=False,
            hidden_dim: int=16,
            latent_dim: int=4, 
            num_disc_steps: int=4,
            num_layers: int=2,
            kernel_size: int=3,
            lr: float=0.001,
        ):
        """
        Args:
            target_columns (list): A list of columns for generation.
            target_columns (list): A list of columns for conditioning.
            window_size (int): A window size to train the generator.
            batch_size (int): A batch size to train the generator.
            num_epochs (int): A number of epochs to train the generator.
            verbose (bool): An indicator to show the progressbar in training.
            hidden_dim (int): The hidden dimensionality of TCN module.
            latent_dim (int): The dimensionality of latent space in GAN.
            num_disc_steps (int): The number of steps to train a discriminator 
                for one step of training a generator.
            num_layers (int): The number of layers in TCN module.
            kernel_size (int): The kernel size in TCN module.
            lr (int): The learning rate to train the generator.
        """
        super().__init__(
            target_columns, 
            conditional_columns,
            window_size, 
            batch_size, 
            num_epochs,
            verbose,
        )
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_disc_steps = num_disc_steps
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.lr = lr

    def fit(self, data: pd.DataFrame):
        """Fit the generator
        
        Args:
            data (pd.DataFrame): A dataframe with time series data.
        """
        gen = _TCNGenerator(
            self.latent_dim, 
            len(self.conditional_columns),
            self.kernel_size, 
            self.hidden_dim, 
            len(self.target_columns), 
            self.num_layers
        )
        disc = _TCNDiscriminator(
            len(self.target_columns), 
            len(self.conditional_columns),
            self.kernel_size, 
            self.hidden_dim, 
            self.num_layers,
        )
        self.model = GANModule(
            gen,
            disc,
            latent_dim=self.latent_dim,
            lr=self.lr,
            num_disc_steps=self.num_disc_steps,
        )
        super().fit(data)
