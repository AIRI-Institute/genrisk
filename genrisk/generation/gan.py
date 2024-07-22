import torch
from torch.optim import SGD
from pytorch_lightning import LightningModule


class GANModule(LightningModule):
    def __init__(
        self, gen, disc, latent_dim, lr, num_disc_steps,
    ):
        super().__init__()
        self.gen = gen
        self.disc = disc
        self.latent_dim = latent_dim
        self.automatic_optimization = False
        self.lr = lr
        self.num_disc_steps = num_disc_steps

    def disc_loss(self, real_logits, fake_logits):
        real_is_real = torch.log(torch.sigmoid(real_logits) + 1e-10)
        fake_is_fake = torch.log(1 - torch.sigmoid(fake_logits) + 1e-10)
        return -(real_is_real + fake_is_fake).mean() / 2
    
    def gen_loss(self, fake_logits):
        fake_is_real = torch.log(torch.sigmoid(fake_logits) + 1e-10)
        return -fake_is_real.mean()

    def training_step(self, batch, batch_idx):
        gen_opt, disc_opt = self.optimizers()
        target, cond = batch
        batch_size = target.shape[0]
        seq_len = target.shape[1]
        z = torch.randn(batch_size, seq_len, self.latent_dim, device=self.device)
        for _ in range(self.num_disc_steps):
            real_logits, _ = self.disc(target, cond)
            with torch.no_grad():
                fake = self.gen(z, cond)
            fake_logits, _ = self.disc(fake, cond)
            d_loss = self.disc_loss(real_logits, fake_logits)

            disc_opt.zero_grad()
            self.manual_backward(d_loss)
            disc_opt.step()
        
        fake = self.gen(z, cond)
        fake_logits, _ = self.disc(fake, cond)
        g_loss = self.gen_loss(fake_logits)

        gen_opt.zero_grad()
        self.manual_backward(g_loss)
        gen_opt.step()

        self.log_dict({'train_gen_loss': g_loss, 'train_disc_loss': d_loss}, prog_bar=True)

    def configure_optimizers(self):
        gen_opt = SGD(self.gen.parameters(), lr=self.lr)
        disc_opt = SGD(self.disc.parameters(), lr=self.lr)
        return gen_opt, disc_opt

    def sample(self, cond, seq_len, n_samples):
        cond = torch.FloatTensor(cond)[None, ...].repeat(n_samples, 1, 1).to(self.device)
        z = torch.randn(n_samples, seq_len, self.latent_dim, device=self.device)
        with torch.no_grad():
            fake = self.gen(z, cond).cpu().numpy()
        return fake
