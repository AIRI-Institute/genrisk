import torch
from torch.optim import Adam
from pytorch_lightning import LightningModule


class VAEModule(LightningModule):
    def __init__(
        self, enc, dec, latent_dim, lr,
    ):
        super().__init__()
        self.enc = enc
        self.dec = dec
        self.latent_dim = latent_dim
        self.lr = lr

    def neg_elbo(self, ts, rec_ts, sigma, mu):
        likelihood = -((rec_ts - ts)**2).sum((1,2))
        kld = -0.5 * (1 + (sigma**2 + 1e-15).log() - mu**2 - sigma**2).sum((1,2))
        elbo = likelihood - kld
        return -elbo.mean()

    def reparametrization_trick(self, mu, sigma):
        z = mu + torch.randn_like(mu) * sigma
        return z

    def training_step(self, batch, batch_idx):
        target, cond = batch
        mu, sigma = self.enc(target, cond)
        z = self.reparametrization_trick(mu, sigma)
        rec_target = self.dec(z, cond)
        loss = self.neg_elbo(target, rec_target, sigma, mu)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    def sample(self, cond, seq_len, n_samples):
        cond = torch.FloatTensor(cond)[None, ...].repeat(n_samples, 1, 1)
        z = torch.randn(n_samples, seq_len, self.latent_dim, device=self.device)
        with torch.no_grad():
            fake = self.dec(z, cond).cpu()
        return fake
