import pickle

from torch.utils.data import DataLoader

# Dataset class
from datasets.mosei_dataset import MOSEIDataset


def get_dataloaders(
    data_path,
    batch_size=32
):
    """
    Loads CMU-MOSEI / CMU-MOSI dataset
    and creates train, validation,
    and test dataloaders.
    """

    # Load dataset file
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # Train dataset
    train_dataset = MOSEIDataset(
        data,
        split="train"
    )

    # Validation dataset
    valid_dataset = MOSEIDataset(
        data,
        split="valid"
    )

    # Test dataset
    test_dataset = MOSEIDataset(
        data,
        split="test"
    )

    # Training loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Validation loader
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return (
        train_loader,
        valid_loader,
        test_loader
    )