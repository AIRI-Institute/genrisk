import pandas as pd
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import Adam

from genrisk.generation.base import TorchGenerator


class _RandomLSTMGenerator(nn.Module):
    __using_hx = False

    def __init__(self, latent_dim, condition_dim, hidden_dim, target_dim, num_layers, butch_size):
        super().__init__()
        self.rnn = nn.LSTM(latent_dim + condition_dim, hidden_dim, num_layers, batch_first=True)
        self.projection = nn.Linear(hidden_dim, target_dim)
        self.latent_dim = latent_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.butch_size = butch_size
        self.h_0 = None
        self.c_0 = None

    def forward(self, z, cond):
        z_cond = torch.cat([z, cond], dim=2)
        if self.__using_hx and self.h_0 is None:
            self.h_0 = torch.randn(self.num_layers, z.shape[0], self.hidden_dim)
            self.c_0 = torch.randn(self.num_layers, z.shape[0], self.hidden_dim)

        if self.h_0 is not None:
            output, (self.h_0, self.c_0) = self.rnn(z_cond, (self.h_0, self.c_0))
        else:
            output, _ = self.rnn(z_cond)

        fake = self.projection(output)
        return fake

    def set_using_hx(self, flag: bool):
        self.__using_hx = flag
        if not flag:
            self.h_0 = self.c_0 = None


class LSTMModule(LightningModule):
    def __init__(
            self, gen, latent_dim, lr, num_disc_steps,
    ):
        super().__init__()
        self.gen: _RandomLSTMGenerator = gen
        self.latent_dim = latent_dim
        self.automatic_optimization = False
        self.lr = lr
        self.num_disc_steps = num_disc_steps

    def configure_optimizers(self):
        optimizer = Adam(self.gen.parameters(), lr=self.lr)
        return optimizer

    def gen_loss(self, target, fake):
        return nn.MSELoss()(target, fake)

    def training_step(self, batch, batch_idx):
        gen_opt = self.optimizers()
        target, cond = batch
        batch_size = target.shape[0]
        seq_len = target.shape[1]
        z = torch.randn(batch_size, seq_len, self.latent_dim,
                        device=self.device)

        fake = self.gen(z, cond)
        g_loss = self.gen_loss(target, fake)

        gen_opt.zero_grad()
        self.manual_backward(g_loss)
        gen_opt.step()

        self.log_dict({'train_gen_loss': g_loss},
                      prog_bar=True)

    def sample(self, cond, seq_len, n_samples):
        cond = torch.FloatTensor(cond)[None, ...].repeat(n_samples, 1, 1).to(
            self.device)
        z = torch.randn(n_samples, seq_len, self.latent_dim, device=self.device)
        with torch.no_grad():
            self.gen.set_using_hx(True)
            fake = self.gen(z, cond).cpu().numpy()
        return fake


class LSTM(TorchGenerator):
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
            conditional_columns (list): A list of columns for conditioning.
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
        gen = _RandomLSTMGenerator(
            latent_dim=self.latent_dim,
            condition_dim=len(self.conditional_columns),
            hidden_dim=self.hidden_dim,
            target_dim=len(self.target_columns),
            num_layers=self.num_layers,
            butch_size=self.batch_size
        )
        self.model = LSTMModule(
            gen=gen,
            latent_dim=self.latent_dim,
            lr=self.lr,
            num_disc_steps=self.num_disc_steps)
        super().fit(data)
