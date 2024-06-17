import os
import warnings
from abc import ABC
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from genrisk.generation.utils import SlidingWindowDataset


warnings.filterwarnings("ignore", ".*does not have many workers.*")


class BaseGenerator(ABC):
    """Base generator class."""

    @abstractmethod
    def __init__(self, target_columns: list[str], conditional_columns: list[str]):
        """
        Args:
            target_columns (list[str]): A list of columns for generation.
            conditional_columns (list[sre]): A list of columns for conditioning.
        """
        self.target_columns = target_columns
        self.conditional_columns = conditional_columns

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """Fit the generator

        Args:
            data (pd.DataFrame): A dataframe with time series data.
        """
        pass

    @abstractmethod
    def sample(self, data: pd.DataFrame, n_samples: int) -> list[pd.DataFrame]:
        """Sample time series data.

        Args:
            data (pd.DataFrame): A dataframe with columns for conditioning.
            n_samples (int): The number of generated to sample.

        Returns:
            list[pd.DataFrame]: A list with sampled dataframes.
        """
        pass

    def postprocess_fake(self, data: pd.DataFrame, target_fakes: list):
        fakes = []
        for target_fake in target_fakes:
            fake = pd.concat(
                [
                    target_fake,
                    data[list(set(data.columns).difference(self.target_columns))],
                ],
                axis=1,
            )
            fakes.append(fake)
        return fakes


class TorchGenerator(BaseGenerator, ABC):
    """Base torch generator class."""

    def __init__(
        self,
        target_columns: list[str],
        conditional_columns: list[str],
        window_size: int,
        batch_size: int,
        num_epochs: int,
        verbose: bool,
    ):
        """
        Args:
            target_columns (list[str]): A list of columns for generation.
            conditional_columns (list[str]): A list of columns for conditioning.
            window_size (int): A window size to train the generator.
            batch_size (int): A batch size to train the generator.
            num_epochs (int): A number of epochs to train the generator.
            verbose (bool): An indicator to show the progressbar in training.
        """

        super().__init__(target_columns, conditional_columns)
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.model = None

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """Fit the generator

        Args:
            data (pd.DataFrame): A dataframe with time series data.
        """
        self.dataset = SlidingWindowDataset(
            df=data,
            target_columns=self.target_columns,
            conditional_columns=self.conditional_columns,
            window_size=self.window_size,
            step_size=1,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.trainer = Trainer(
            enable_progress_bar=self.verbose,
            accelerator="auto",
            max_epochs=self.num_epochs,
            log_every_n_steps=np.ceil(len(self.dataloader) * 0.1),
            logger=CSVLogger("."),
        )
        self.trainer.fit(
            model=self.model,
            train_dataloaders=self.dataloader,
        )

    def sample(self, data: pd.DataFrame, n_samples: int) -> list[pd.DataFrame]:
        """Sample time series data.

        Args:
            data (pd.DataFrame): A dataframe with columns for conditioning.
            n_samples (int): A number of samples to sample.

        Returns:
            list[pd.DataFrame]: A list with samples dataframes.
        """
        _fake = self.model.sample(
            data[self.conditional_columns].values, data.shape[0], n_samples
        )  # (n_samples, seq_len, target_dim)
        target_fakes = []
        for fake in _fake:
            fake_df = pd.DataFrame(fake, index=data.index, columns=self.target_columns)
            target_fakes.append(fake_df)
        return super().postprocess_fake(data, target_fakes)

    def save_model(self, path="generation_models/gen_model.pth"):
        """Save the model to a file.

        Args:
            path (str): The path where the model will be saved.
        """
        self.ensure_directory(path)
        torch.save(self.model.state_dict(), path)

    def ensure_directory(self, path):
        """Ensures that the directory exists."""
        os.makedirs(path, exist_ok=True)

    def load_model(self, path="generation_models/gen_model.pth"):
        """Load the model from a file.

        Args:
            path (str): The path where the model is saved.
        """
        self.model.load_state_dict(torch.load(path))
