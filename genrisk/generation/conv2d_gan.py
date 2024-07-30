import pandas as pd
import torch
import torch.nn as nn
from genrisk.generation.base import TorchGenerator
from genrisk.generation.gan import GANModule


class _Conv2dGenerator(nn.Module):
    def __init__(self, latent_dim, condition_dim, hidden_dim, target_dim,
                 num_layers):
        super().__init__()
        self.pad = torch.nn.functional.pad
        self.dim = latent_dim + condition_dim
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_dim,
                               kernel_size=(3, latent_dim + condition_dim))
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim,
                               kernel_size=(3, latent_dim + condition_dim))
        self.projection = nn.Linear(hidden_dim, target_dim)

    def forward(self, z, cond):
        z_cond = torch.cat([z, cond], dim=2)
        z_cond = z_cond.unsqueeze(1)
        if (self.dim) % 2 != 0:
            z_cond = self.pad(z_cond,
                              ((self.dim - 1) // 2, (self.dim - 1) // 2, 1, 1))
        else:
            z_cond = self.pad(z_cond,
                              ((self.dim) // 2 - 1, self.dim // 2, 1, 1))

        output = self.relu(self.conv1(z_cond))
        output = self.pad(output, (0, 0, 1, 1))
        output = self.relu(self.conv2(output))
        output = output.squeeze(3).permute(0, 2, 1)  #
        fake = self.projection(output)
        return fake


class _Conv2dDiscriminator(nn.Module):
    def __init__(self, target_dim, condition_dim, hidden_dim, num_layers):
        super().__init__()
        self.pad = torch.nn.functional.pad
        self.dim = target_dim + condition_dim
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_dim,
                               kernel_size=(3, target_dim + condition_dim))
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim,
                               kernel_size=(3, target_dim + condition_dim))
        self.projection = nn.Linear(hidden_dim, 1)

    def forward(self, target, cond):
        target_cond = torch.cat([target, cond], dim=2)
        target_cond = target_cond.unsqueeze(1)
        if (self.dim) % 2 != 0:
            target_cond = self.pad(target_cond, (
            (self.dim - 1) // 2, (self.dim - 1) // 2, 1, 1))
        else:
            target_cond = self.pad(target_cond,
                                   ((self.dim) // 2 - 1, self.dim // 2, 1, 1))
        output = self.relu(self.conv1(target_cond))
        output = self.pad(output, (0, 0, 1, 1))
        output = self.relu(self.conv2(output))
        output = output.squeeze(3).permute(0, 2, 1)
        logits = self.projection(output)
        return logits, output


class Conv2dGAN(TorchGenerator):
    """GAN generator based on Conv2d module."""

    def __init__(
            self,
            target_columns: list[str],
            conditional_columns: list[str],
            window_size: int = 10,
            batch_size: int = 16,
            num_epochs: int = 1,
            verbose: bool = False,
            hidden_dim: int = 16,
            latent_dim: int = 4,
            num_disc_steps: int = 1,
            num_layers: int = 1,
            lr: float = 0.01,
    ):
        """
        Args:
            target_columns (list): A list of columns for generation.
            target_columns (list): A list of columns for conditioning.
            window_size (int): A window size to train the generator.
            batch_size (int): A batch size to train the generator.
            num_epochs (int): A number of epochs to train the generator.
            verbose (bool): An indicator to show the progressbar in training.
            hidden_dim (int): The hidden dimensionality of Conv2d module.
            latent_dim (int): The dimensionality of latent space in GAN.
            num_disc_steps (int): The number of steps to train a discriminator
                for one step of training a generator.
            num_layers (int): The number of layers in Conv2d module.
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
        gen = _Conv2dGenerator(
            latent_dim=self.latent_dim,
            condition_dim=len(self.conditional_columns),
            hidden_dim=self.hidden_dim,
            target_dim=len(self.target_columns),
            num_layers=self.num_layers,
        )
        disc = _Conv2dDiscriminator(
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
            num_disc_steps=self.num_disc_steps
        )
        super().fit(data)
