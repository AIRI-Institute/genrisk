import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from genrisk.generation.base import TorchGenerator
from genrisk.generation.vae import VAEModule

class TSPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.2, max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

class TSEncoder(nn.Module):
  def __init__(self, target_dim = 7, cond_dim = 6, d_model = 256, nhead = 4, num_layers = 1, dropout = 0.2, latent_dim = 8):
    super().__init__()
    self.ts_encoder = nn.Linear(target_dim, d_model - cond_dim)
    self.ts_pos_encoder = TSPositionalEncoding(d_model - cond_dim, dropout)
    encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
    self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)

    self.projection = nn.Linear(d_model, d_model // 4)
    self.relu = nn.ReLU()
    self.mu = nn.Linear(d_model // 4, latent_dim)
    self.sigma = nn.Linear(d_model // 4, latent_dim)

  def forward(self, target, cond):
    x = self.ts_encoder(target)
    x = self.ts_pos_encoder(x)
    x = torch.cat([x, cond], dim = 2)
    x = self.transformer(x)
    x = self.relu(self.projection(x))

    mu = self.mu(x)
    sigma = torch.relu(self.sigma(x))

    return mu, sigma


class TSDecoder(nn.Module):
    def __init__(self, seq_len, target_dim = 7, cond_dim = 6, d_model = 256, nhead = 4, num_layers = 1, dropout = 0.2, latent_dim = 8):
        super().__init__()
        self.latent_decoder = nn.Linear(latent_dim, d_model - cond_dim)
        self.ts_pos_encoder = TSPositionalEncoding(d_model - cond_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.projection = nn.Linear(d_model, target_dim)
        device_mask = 'cuda' if torch.cuda_is_available() else 'cpu'
        self.att_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device = device_mask)


    def forward(self, x, cond):
        x = self.latent_decoder(x)
        x = self.ts_pos_encoder(x)
        x = torch.cat([x, cond], dim = 2)

        x = self.transformer(x, mask = self.att_mask, is_causal = True)
        x = self.projection(x)
        return x


class TransformerVAE(TorchGenerator):
    def __init__(
            self,
            target_columns: list[str],
            conditional_columns: list[str],
            window_size: int=200,
            batch_size: int=64,
            num_epochs: int=20,
            verbose: bool=False,
            hidden_dim: int=256,
            latent_dim: int=8,
            num_layers: int=2,
            lr: float=0.001,
            att_head : int = 4,
            dropout : float = 0.2
        ):

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
        self.att_head = att_head
        self.dropout = dropout
        self.seq_len = window_size

    def fit(self, data: pd.DataFrame):
        """Fit the generator

        Args:
            data (pd.DataFrame): A dataframe with time series data.
        """
        enc = TSEncoder(
            len(self.target_columns),
            len(self.conditional_columns),
            self.hidden_dim,
            self.att_head,
            self.num_layers,
            self.dropout,
            self.latent_dim
        )

        dec = TSDecoder(
            self.seq_len,
            len(self.target_columns),
            len(self.conditional_columns),
            self.hidden_dim,
            self.att_head,
            self.num_layers,
            self.dropout,
            self.latent_dim
        )

        self.model = VAEModule(enc, dec, self.latent_dim, self.lr)
        super().fit(data)
