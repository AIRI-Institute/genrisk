import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from diffusers import UNet2DConditionModel, DDPMScheduler
import wandb
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

def batch_generator(data, batch_size):
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]
    X_mb = data[train_idx]
    return X_mb

class RNNDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(RNNDiscriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        y_hat_logit = self.fc(last_hidden)
        y_hat = self.sigmoid(y_hat_logit)
        return y_hat_logit, y_hat

def discriminative_score_metrics(ori_data, generated_data):

    
    no, dim = ori_data.shape
    hidden_dim = int(dim / 2)
    iterations = 2000
    batch_size = 128

    discriminator = RNNDiscriminator(input_size=dim, hidden_dim=hidden_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(discriminator.parameters())

    train_x, train_x_hat, test_x, test_x_hat = train_test_divide(ori_data, generated_data)

    for itt in tqdm(range(iterations), desc='training', total=iterations):
        X_mb = batch_generator(train_x, batch_size)
        X_hat_mb = batch_generator(train_x_hat, batch_size)

        X_mb = torch.tensor(X_mb, dtype=torch.float32).unsqueeze(1).requires_grad_(True)
        X_hat_mb = torch.tensor(X_hat_mb, dtype=torch.float32).unsqueeze(1).requires_grad_(True)

        optimizer.zero_grad()

        y_logit_real, y_pred_real = discriminator(X_mb)
        y_logit_fake, y_pred_fake = discriminator(X_hat_mb)

        real_labels = torch.ones_like(y_logit_real)
        fake_labels = torch.zeros_like(y_logit_fake)
        d_loss_real = criterion(y_pred_real, real_labels)
        d_loss_fake = criterion(y_pred_fake, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        d_loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_x = torch.tensor(test_x, dtype=torch.float32).unsqueeze(1)
        test_x_hat = torch.tensor(test_x_hat, dtype=torch.float32).unsqueeze(1)

        y_pred_real_curr = discriminator(test_x)[1].numpy()
        y_pred_fake_curr = discriminator(test_x_hat)[1].numpy()

    y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis=0))
    y_label_final = np.concatenate((np.ones([len(y_pred_real_curr)]), np.zeros([len(y_pred_fake_curr)])), axis=0)

    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))

    fake_acc = accuracy_score(np.zeros([len(y_pred_fake_curr)]), (y_pred_fake_curr > 0.5))
    real_acc = accuracy_score(np.ones([len(y_pred_real_curr)]), (y_pred_real_curr > 0.5))

    discriminative_score = np.abs(0.5 - acc)
    return discriminative_score, fake_acc, real_acc


def cacf_torch(x, max_lag, dim=(0, 1)):
    def get_lower_triangular_indices(n):
        return [list(x) for x in torch.tril_indices(n, n)]

    ind = get_lower_triangular_indices(x.shape[2])
    x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)
    x_l = x[..., ind[0]]
    x_r = x[..., ind[1]]
    cacf_list = list()
    for i in range(max_lag):
        y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r
        cacf_i = torch.mean(y, (1))
        cacf_list.append(cacf_i)
    cacf = torch.cat(cacf_list, 1)
    return cacf.reshape(cacf.shape[0], -1, len(ind[0]))

class Loss(nn.Module):
    def __init__(self, name, reg=1.0, transform=lambda x: x, threshold=10., backward=False, norm_foo=lambda x: x):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()

    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)

class CrossCorrelLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(CrossCorrelLoss, self).__init__(norm_foo=lambda x: torch.abs(x).sum(0), **kwargs)
        self.cross_correl_real = cacf_torch(self.transform(x_real), 1).mean(0)[0]

    def compute(self, x_fake):
        cross_correl_fake = cacf_torch(self.transform(x_fake), 1).mean(0)[0]
        loss = self.norm_foo(cross_correl_fake - self.cross_correl_real.to(x_fake.device))
        return loss / 10.

class DiffusionModel(nn.Module):
    def __init__(self, dim_time_series, dim_cond):
        super(DiffusionModel, self).__init__()
        self.dim_time_series = dim_time_series
        self.dim_cond = dim_cond
        self.unet = UNet2DConditionModel(
            conv_in_kernel=7, 
            conv_out_kernel=7, 
            sample_size=1, 
            in_channels=dim_time_series, 
            out_channels=dim_time_series,
            down_block_types=('DownBlock2D', 'DownBlock2D'), 
            mid_block_type='UNetMidBlock2D', 
            up_block_types=('UpBlock2D', 'UpBlock2D'), 
            block_out_channels=(16, 32), 
            layers_per_block=2, 
            norm_num_groups=4, 
            cross_attention_dim=dim_cond, 
            attention_head_dim=4
        )
        
        self.scheduler = DDPMScheduler(num_train_timesteps=100)

    def forward(self, x, cond, t):
        x = x.float().unsqueeze(2).unsqueeze(3)
        cond = cond.float().unsqueeze(1)
        return self.unet(x, t, encoder_hidden_states=cond).sample

    def train_model(self, train_data, train_cond, val_data, val_cond, epochs=10, batch_size=64, lr=0.001):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        train_data, train_cond = train_data.to(device), train_cond.to(device)
        val_data, val_cond = val_data.to(device), val_cond.to(device)

        train_loader = DataLoader(TensorDataset(train_data, train_cond), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_data, val_cond), batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self._setup_wandb(epochs, lr)

        for epoch in range(epochs):
            train_loss = self._train_one_epoch(train_loader, optimizer, criterion, device)
            if (epoch + 1) % 5 == 0:
                val_loss, train_corr, val_corr, train_discr, val_discr = self._validate(
                    train_data, train_cond, val_data, val_cond, criterion, device)
                self._log_epoch(epoch, train_loss, val_loss, train_corr, val_corr, train_discr, val_discr)

        wandb.finish()

    def sample(self, cond_test, num_sample):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        cond_test = cond_test.to(device)

        self.eval()
        samples = []

        for _ in range(num_sample):
            x = torch.randn((cond_test.shape[0], self.dim_time_series), device=device).float()
            for t in reversed(range(self.scheduler.config.num_train_timesteps)):
                timesteps = torch.tensor([t], device=device).long()
                with torch.no_grad():
                    model_output = self.forward(x, cond_test, timesteps)
                scheduler_output = self.scheduler.step(model_output, t, x.unsqueeze(2).unsqueeze(3))
                x = scheduler_output.prev_sample.squeeze(3).squeeze(2)
            samples.append(x)

        return torch.stack(samples)

    def _setup_wandb(self, epochs, lr):
        """Настройка логирования в wandb."""
        wandb.login(key="YOUR_WANDB_API_KEY")
        run_name = f"{sum(p.numel() for p in self.parameters())} param {epochs} epochs id{np.random.randint(999)}"
        wandb.init(project="DiffusionTimeSeries", name=run_name, config={
            "epochs": epochs,
            "learning_rate": lr,
            "model_parameters": sum(p.numel() for p in self.parameters()),
            "architecture": self.__repr__()
        })

    def _train_one_epoch(self, train_loader, optimizer, criterion, device):
        """Обучение модели за одну эпоху."""
        self.train()
        epoch_loss = 0
        for batch_data, batch_cond in train_loader:
            batch_data, batch_cond = batch_data.to(device), batch_cond.to(device)

            optimizer.zero_grad()
            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_data.size(0),), device=device).long()
            noise = torch.randn_like(batch_data)
            noisy_data = self.scheduler.add_noise(batch_data, noise, timesteps)
            output = self.forward(noisy_data, batch_cond, timesteps)
            loss = criterion(output.squeeze(2).squeeze(2), noise)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss / len(train_loader)

    def _validate(self, train_data, train_cond, val_data, val_cond, criterion, device):
        """Валидация модели."""
        self.eval()
        with torch.no_grad():
            train_samples = self.sample(train_cond, num_sample=1)
            val_samples = self.sample(val_cond, num_sample=1)

            train_corr = self._compute_correlation(train_data, train_samples)
            val_corr = self._compute_correlation(val_data, val_samples)

            val_loss = criterion(val_samples.squeeze(2).squeeze(2), val_data)

            train_discr, _, _ = discriminative_score_metrics(train_data.cpu().numpy(),
                                                             train_samples.squeeze(0).cpu().numpy())
            val_discr, _, _ = discriminative_score_metrics(val_data.cpu().numpy(),
                                                           val_samples.squeeze(0).cpu().numpy())

        return val_loss.item(), train_corr, val_corr, train_discr, val_discr

    def _compute_correlation(self, real_data, generated_samples):
        """Вычисление корреляции."""
        cross_corr_loss = CrossCorrelLoss(real_data.unsqueeze(1), name='CrossCorrelLoss')
        return cross_corr_loss.compute(generated_samples).item()

    def _log_epoch(self, epoch, train_loss, val_loss, train_corr, val_corr, train_discr, val_discr):
        """Логгирование результатов эпохи в wandb."""
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_cross_correlation": train_corr,
            "val_cross_correlation": val_corr,
            "train_discriminative_score": train_discr,
            "val_discriminative_score": val_discr
        })
