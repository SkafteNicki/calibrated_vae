import torch
import torchvision


def get_dataset(dataset_name):
    if dataset_name == "svhn":
        train = torchvision.datasets.SVHN(
            root=f"data/{dataset_name}/",
            download=True,
            split="train",
            transform=torchvision.transforms.ToTensor(),
        )
        test = torchvision.datasets.SVHN(
            root=f"data/{dataset_name}/",
            download=True,
            split="test",
            transform=torchvision.transforms.ToTensor(),
        )
    elif dataset_name == "cifar10":
        train = torchvision.datasets.CIFAR10(
            root=f"data/{dataset_name}/",
            download=True,
            train=True,
            transform=torchvision.transforms.ToTensor(),
        )
        test = torchvision.datasets.CIFAR10(
            root=f"data/{dataset_name}/",
            download=True,
            train=False,
            transform=torchvision.transforms.ToTensor(),
        )
    else:
        raise ValueError("Unknown dataset")

    n_train = int(len(train) * 0.9)
    train, val = torch.utils.data.random_split(train, [n_train, len(train) - n_train])
    return train, val, test
