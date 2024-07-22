from torch.utils.data import Dataset
import torch

class SlidingWindowDataset(Dataset):
    def __init__(self, df, target_columns, conditional_columns, window_size, step_size):
        self.window_size = window_size
        self.target_columns = target_columns
        self.conditional_columns = conditional_columns
        self.df = df
        self.step_size = step_size
    
    def __len__(self):
        return (len(self.df) - self.window_size + 1) // self.step_size
    
    def __getitem__(self, idx):
        target = self.df[self.target_columns].iloc[range(idx*self.step_size, idx*self.step_size + self.window_size)]
        cond = self.df[self.conditional_columns].iloc[range(idx*self.step_size, idx*self.step_size + self.window_size)]
        return torch.FloatTensor(target.values), torch.FloatTensor(cond.values)
