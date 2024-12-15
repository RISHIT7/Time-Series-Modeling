import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np

from .utils import load_dataset

class HARDataset(Dataset):
    """
    HAR dataset class for PyTorch without any augmentation
    """
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize the dataset with the given data
        
        Parameters:
        --------
        - X: numpy array of shape (n_samples, n_features)
        - y: numpy array of shape (n_samples,)
        
        Returns:
        --------
        - None
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]        
    
def _compute_mean_std(train_loader: HARDataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the mean and standard deviation of the dataset
    
    Parameters:
    --------
    - train_loader: DataLoader object
    
    Returns:
    --------
    - mean: mean of the dataset
    - std: standard deviation of the dataset
    """
    mean = torch.zeros(128, 9)
    sum_sq = torch.zeros(128, 9)
    
    for i, (X, y) in enumerate(train_loader):
        mean += X.mean(dim=0)
        sum_sq += (X**2).mean(dim=0)
        
        
    mean /= len(train_loader)
    std = torch.sqrt(sum_sq/len(train_loader) - mean**2)
    
    return mean, std

class Normalize:
    """
    Class to normalize the data.
    """
    def __init__(self, mean: np.ndarray, std: np.ndarray, device='cpu') -> None:
        """
        Initialize the class with the mean and standard deviation
        
        Parameters:
        --------
        - mean: mean of the dataset
        - std: standard deviation of the dataset
        - device: device to move the mean and std (Default: 'cpu')
        
        Returns:
        --------
        - None
        """
        self.mean = mean.to(device)
        self.std = std.to(device)
        
    def __call__(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        
        return (X - self.mean) / self.std

class AddNoise:
    """
    Add noise to the data
    """
    def __init__(self, noise_level=0.01) -> None:
        """
        Initialize the class with the noise level
        
        Parameters:
        --------
        - noise_level: noise level to add to the data (Default: 0.01)
        
        Returns:
        --------
        - None
        """
        self.noise_level = noise_level
        
    def __call__(self, X):
        noise = torch.randn_like(X) * self.noise_level
        return X + noise
    
class HARDatasetWithAugmentation(Dataset):
    """
    HAR dataset class for PyTorch with augmentation
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, transforms=None) -> None:
        """
        Initialize the dataset with the given data
        
        Parameters:
        --------
        - X: numpy array of shape (n_samples, n_features)
        - y: numpy array of shape (n_samples,)
        - transforms: list of transformations to apply to the data
        
        Returns:
        --------
        - None
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.transforms = transforms
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        X, y = self.X[index], self.y[index]
        
        if self.transforms:
            for transform in self.transforms:
                X = transform(X)
        
        return X, y
    
if __name__ == "__main__":
    BATCH = 256
    trainX, trainy, testX, testy = load_dataset()
    
    train_loader = HARDataset(trainX, trainy)
    
    mean, std = _compute_mean_std(train_loader)
    
    normalize_transform = Normalize(torch.Tensor(mean), torch.Tensor(std))
    add_noise_transform = AddNoise(noise_level=0.02)

    transforms = [normalize_transform, add_noise_transform]
    # transforms = [normalize_transform]

    train_dataset = HARDatasetWithAugmentation(trainX, trainy, transforms=transforms)
    test_dataset = HARDatasetWithAugmentation(testX, testy, transforms=transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)