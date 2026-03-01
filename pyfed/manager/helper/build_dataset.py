import torch
import torchvision.transforms as transforms
from pyfed.dataset.dataset import (
    Prostate, ProstatePre, Fundus
)


def build_dataset(config, site):
    assert site in config.INNER_SITES + config.OUTER_SITES
    if config.DATASET == "prostate":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        train_set = ProstatePre(
            site=site,
            base_path=config.DIR_DATA,
            train_ratio=config.TRAIN_RATIO,
            split="train",
            transform=transform,
        )

        valid_set = ProstatePre(
            site=site,
            base_path=config.DIR_DATA,
            train_ratio=config.TRAIN_RATIO,
            split="valid",
            transform=transform,
        )
        test_set = ProstatePre(
            site=site,
            base_path=config.DIR_DATA,
            train_ratio=config.TRAIN_RATIO,
            split="test",
            transform=transform,
        )
    elif config.DATASET == "fundus":
        transform = None
        train_set = Fundus(
            site=site, base_path=config.DIR_DATA, split="train", transform=transform
        )
        valid_set = Fundus(
            site=site, base_path=config.DIR_DATA, split="test", transform=transform
        )
        test_set = Fundus(
            site=site, base_path=config.DIR_DATA, split="test", transform=transform
        )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config.TRAIN_BATCHSIZE, shuffle=True, drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=config.TRAIN_BATCHSIZE, shuffle=False, drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=config.TRAIN_BATCHSIZE, shuffle=False, drop_last=False
    )

    return train_loader, valid_loader, test_loader


def build_central_dataset(config, sites):
    train_sets, valid_sets, test_sets = [], [], []
    train_loaders, valid_loaders, test_loaders = [], [], []
    if config.DATASET == "prostate":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        for site in sites:
            train_set = Prostate(
                site=site, base_path=config.DIR_DATA, split="train", transform=transform
            )
            valid_set = Prostate(
                site=site, base_path=config.DIR_DATA, split="valid", transform=transform
            )
            test_set = Prostate(
                site=site, base_path=config.DIR_DATA, split="test", transform=transform
            )

            train_sets.append(train_set)
            valid_sets.append(valid_set)
            test_sets.append(test_set)

        train_set = torch.utils.data.ConcatDataset(train_sets)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=config.TRAIN_BATCHSIZE, shuffle=True
        )
        valid_loaders, test_loaders = [], []
        for valid_set, test_set in zip(valid_sets, test_sets):
            valid_loaders.append(
                torch.utils.data.DataLoader(
                    valid_set, batch_size=config.TRAIN_BATCHSIZE, shuffle=False
                )
            )
            test_loaders.append(
                torch.utils.data.DataLoader(
                    test_set, batch_size=config.TRAIN_BATCHSIZE, shuffle=False
                )
            )


    return train_loader, valid_loaders, test_loaders
