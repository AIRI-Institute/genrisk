import torch
import torch.nn as nn
import pandas as pd

from genrisk.generation.base import TorchGenerator
from genrisk.generation.vae import VAEModule


class _LSTMEncoder(nn.Module):
    def __init__(self, target_dim, condition_dim, hidden_dim, latent_dim, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=target_dim+condition_dim, 
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=num_layers,
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.sigma = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, target, cond):
        target_cond = torch.cat([target, cond], dim=2)
        h, _ = self.rnn(target_cond) # (batch size, seq len, hidden_dim)
        mu, sigma = self.mu(h), torch.relu(self.sigma(h))
        return mu, sigma
    

class _LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, condition_dim, hidden_dim, target_dim, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=latent_dim+condition_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=num_layers,
        )
        self.projection = nn.Linear(hidden_dim, target_dim)
    
    def forward(self, z, cond):
        z_cond = torch.cat([z, cond], dim=2)
        h, _ = self.rnn(z_cond)
        rec_target = self.projection(h)
        return rec_target


class LSTMVAE(TorchGenerator):
    """VAE generator based on LSTM module."""
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
            num_layers: int=1,
            lr: float=0.01,
        ):
        """
        Args:
            target_columns (list): A list of columns for generation.
            conditional_columns (list): A list of columns for conditioning.
            window_size (int): A window size to train the generator.
            batch_size (int): A batch size to train the generator.
            num_epochs (int): A number of epochs to train the generator.
            verbose (bool): An indicator to show the progressbar in training.
            hidden_dim (int): The hidden dimensionality of LSTM module.
            latent_dim (int): The dimensionality of latent space in VAE.
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
        self.num_layers = num_layers
        self.lr = lr

    def fit(self, data: pd.DataFrame):
        """Fit the generator
        
        Args:
            data (pd.DataFrame): A dataframe with time series data.
        """
        enc = _LSTMEncoder(
            len(self.target_columns), 
            len(self.conditional_columns), 
            self.hidden_dim, 
            self.latent_dim, 
            self.num_layers
        )
        dec = _LSTMDecoder(
            self.latent_dim, 
            len(self.conditional_columns), 
            self.hidden_dim, 
            len(self.target_columns), 
            self.num_layers
        )
        self.model = VAEModule(enc, dec, self.latent_dim, self.lr)
        super().fit(data)
