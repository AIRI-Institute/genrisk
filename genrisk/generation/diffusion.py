import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd


class DenoisingDiffusion(nn.Module):
    def __init__(self, target_columns, conditional_columns, kernel_size, hidden_dim, latent_dim, num_layers, lr, num_epochs, num_timesteps, verbose):
        super(DenoisingDiffusion, self).__init__()
        
        self.target_columns = target_columns
        self.conditional_columns = conditional_columns
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.lr = lr
        self.num_epochs = num_epochs
        self.num_timesteps = num_timesteps
        self.verbose = verbose

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.L1Loss()


    def _build_model(self):
        layers = []
        input_dim = len(self.target_columns) + len(self.conditional_columns)
        
        for i in range(self.num_layers):
            layers.append(nn.Conv1d(input_dim, self.hidden_dim, self.kernel_size, padding='same'))
            layers.append(nn.BatchNorm1d(self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.5))
            input_dim = self.hidden_dim
        
        layers.append(nn.Conv1d(self.hidden_dim, len(self.target_columns), 1))
        return nn.Sequential(*layers)
    

    def fit(self, train_data):
        train_dataset = self._prepare_data(train_data)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}", leave=False):
                self.optimizer.zero_grad()
                batch = batch[0].to(self.device)
                
                x_target = batch[:, :len(self.target_columns), :]
                x_cond = batch[:, len(self.target_columns):, :]
                
                noise = torch.randn_like(x_target).to(self.device)
                noisy_target = x_target + noise
                
                output = self.model(torch.cat((x_cond, noisy_target), dim=1))
                
                loss = self.criterion(output, x_target)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss / len(train_loader)}")


    def _prepare_data(self, data):
        data = data.values.T
        data = torch.tensor(data, dtype=torch.float32)
        return TensorDataset(data.unsqueeze(0))
    

    def sample(self, conditional_data, n_samples=5):
        conditional_data_length = len(conditional_data)
        conditional_data_prepared = self._prepare_data(conditional_data)
        data_loader = DataLoader(conditional_data_prepared, batch_size=1, shuffle=False)
        
        samples = []
        for i in range(n_samples):
            batch = next(iter(data_loader))[0].to(self.device)
            x_cond = batch[:, len(self.target_columns):, :]
            
            noise = torch.randn((1, len(self.target_columns), conditional_data_length)).to(self.device)
            generated = noise
            
            for t in reversed(range(self.num_timesteps)):
                noisy_input = torch.cat((x_cond, generated), dim=1)
                generated = self.model(noisy_input)
                if t > 0:
                    beta = self._get_beta_schedule(t)
                    noise = torch.randn_like(generated).to(self.device)
                    generated = generated + noise * beta.sqrt() * (t / self.num_timesteps)
            
            generated = generated.detach().cpu().numpy()
            samples.append(pd.DataFrame(generated[0, :, :].T, columns=self.target_columns))

        return samples
    

    def _get_beta_schedule(self, t):
        return torch.tensor(0.1 * t / self.num_timesteps, dtype=torch.float32).to(self.device)
