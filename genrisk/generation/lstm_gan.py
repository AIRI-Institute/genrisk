import torch
import torch.nn as nn
import pandas as pd

from genrisk.generation.base import TorchGenerator
from genrisk.generation.gan import GANModule


class _LSTMGenerator(nn.Module):
    def __init__(self, latent_dim, condition_dim, hidden_dim, target_dim, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(latent_dim + condition_dim, hidden_dim, num_layers, batch_first=True)
        self.projection = nn.Linear(hidden_dim, target_dim)

    def forward(self, z, cond):
        z_cond = torch.cat([z, cond], dim=2)
        output, _ = self.rnn(z_cond)
        fake = self.projection(output)
        return fake


class _LSTMDiscriminator(nn.Module):
    def __init__(self, target_dim, condition_dim, hidden_dim, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(target_dim + condition_dim, hidden_dim, num_layers, batch_first=True)
        self.projection = nn.Linear(hidden_dim, 1)

    def forward(self, target, cond):
        target_cond = torch.cat([target, cond], dim=2)
        output, _ = self.rnn(target_cond)
        logits = self.projection(output)
        return logits, output


class LSTMGAN(TorchGenerator):
    """GAN generator based on LSTM module."""
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
            num_disc_steps: int=1,
            num_layers: int=1,
            lr: float=0.01,
        ):
        """
        Args:
            target_columns (list): A list of columns for generation.
            target_columns (list): A list of columns for conditioning.
            window_size (int): A window size to train the generator.
            batch_size (int): A batch size to train the generator.
            num_epochs (int): A number of epochs to train the generator.
            verbose (bool): An indicator to show the progressbar in training.
            hidden_dim (int): The hidden dimensionality of LSTM module.
            latent_dim (int): The dimensionality of latent space in GAN.
            num_disc_steps (int): The number of steps to train a discriminator 
                for one step of training a generator.
            num_layers (int): The number of layers in LSTM module.
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
        self.lr = lr

    def fit(self, data: pd.DataFrame):
        """Fit the generator
        
        Args:
            data (pd.DataFrame): A dataframe with time series data.
        """
        gen = _LSTMGenerator(
            latent_dim=self.latent_dim, 
            condition_dim=len(self.conditional_columns), 
            hidden_dim=self.hidden_dim,
            target_dim=len(self.target_columns),
            num_layers=self.num_layers,
        )
        disc = _LSTMDiscriminator(
            target_dim=len(self.target_columns),
            condition_dim=len(self.conditional_columns),
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
        self.model = GANModule(
            gen,
            disc,
            latent_dim=self.latent_dim,
            lr=self.lr,
            num_disc_steps=self.num_disc_steps,
        )
        super().fit(data)
