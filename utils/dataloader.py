import pickle

from torch.utils.data import DataLoader

from datasets.mosei_dataset import MOSEIDataset


def get_dataloaders(
    data_path,
    batch_size=32
):

    # load dataset once
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    train_dataset = MOSEIDataset(
        data,
        split="train"
    )

    valid_dataset = MOSEIDataset(
        data,
        split="valid"
    )

    test_dataset = MOSEIDataset(
        data,
        split="test"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

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