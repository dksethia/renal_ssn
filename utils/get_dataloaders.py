import time
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from utils.dataset import BasicDataset

def get_train_loaders(train_path,
                      batch_size,
                      augment,
                      calc_edge_map=False,
                      split_ratio=0.9,
                      oversample=False,
                      oversample_classes=[2, 3],
                      oversample_weight=4,
                      downsample_empty=False,
                      downsample_weight=0.5,
                      seed=42):
    """ Get train and validation loaders for kidney dataset, with optional oversampling of minority classes

    Args:
        train_path (str): path to training data
        batch_size (int): batch size
        augment (bool): whether to apply data augmentation
        calc_edge_map (bool): whether to calculate edge map
        split_ratio (float): fraction of data to use for training
        oversample (bool): whether to oversample minority classes
        oversample_classes (list): list of classes to oversample
        oversample_weight (int): weight to assign to minority classes
        downsample_empty (bool): whether to downsample empty masks
        downsample_weight (int): weight to assign to empty masks
        seed (int): random seed for reproducibility

    Returns:
        train_loader (DataLoader): training DataLoader
        val_loader (DataLoader): validation DataLoader
    """

    if not (0 < split_ratio < 1):
        raise ValueError("split_ratio must be between 0 and 1.")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    
    train_dataset = BasicDataset(train_path, augment=augment, calc_edge_map=calc_edge_map)
    val_dataset = BasicDataset(train_path, augment=False, calc_edge_map=calc_edge_map)
    print(f"Number of training samples: {len(train_dataset)}")

    # Split data into training and validation sets using seed
    train_len = int(split_ratio * len(train_dataset))
    val_len = len(train_dataset) - train_len

    train_set, _ = random_split(train_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(seed))
    _ , val_set = random_split(val_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(seed))

    # Minority class oversampling
    sampler = None
    if oversample:
        weights = get_oversampling_weights(train_set, oversample_classes, oversample_weight, downsample_empty, downsample_weight)
        sampler = WeightedRandomSampler(weights, len(train_set), replacement=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=(not oversample), num_workers=0, pin_memory=True, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader


def get_oversampling_weights(dataset, oversample_classes, oversample_weight, downsample_empty, downsample_weight):
    """ Get weights for each sample in dataset for minority class oversampling

    Args:
        dataset (Dataset): dataset to oversample
        classes (list): list of classes to oversample
        oversample_weight (int): weight to assign to minority classes
        downsample_empty (bool): whether to downsample empty masks
        downsample_weight (int): weight to assign to empty masks

    Returns:
        weights (list): list containing the weight for each sample in dataset
    """

    print(f"Oversampling classes {oversample_classes} with weight={oversample_weight}")
    if downsample_empty:
        print(f"Downsampling empty masks with weight={downsample_weight}")

    weights = [1] * len(dataset)

    start = time.time()

    for idx, (_, mask, _) in enumerate(dataset):
        if any(c in mask for c in oversample_classes):
            weights[idx] = oversample_weight
        elif downsample_empty and mask.sum() == 0: # or np.all(mask == 0) ?
            weights[idx] = downsample_weight

    end = time.time()
    print(f"Time taken to calculate weights: {end - start:.2f}s")

    return weights


def get_test_loader(test_path):
    """ Get test loader for kidney dataset

    Args:
        test_path (str): path to test data

    Returns:
        test_loader (DataLoader): test DataLoader
    """

    test_dataset = BasicDataset(test_path, augment=False, calc_edge_map=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return test_loader


if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)

    # Hyperparameters
    train_path = '/vol/bitbucket/dks20/renal_ssn/labelbox_download/train_data_random.h5'
    oversample_classes = [2, 3]
    oversample_weight = 4
    downsample_weight = 0.5

    train_loader, val_loader = get_train_loaders(train_path=train_path,
                                                batch_size=1,
                                                augment=True,
                                                calc_edge_map=False,
                                                oversample=False, 
                                                oversample_classes=oversample_classes, 
                                                oversample_weight=oversample_weight,
                                                downsample_empty=False,
                                                downsample_weight=downsample_weight,
                                                seed=SEED)
    

