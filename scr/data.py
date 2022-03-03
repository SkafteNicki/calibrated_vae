import torch
from torchvision import datasets, transforms

def rgb_transform(dataset):
    data = torch.tensor(dataset.data) / 255.0
    if hasattr(dataset, 'targets'):
        targets = torch.tensor(dataset.targets)
        data = data.permute(0, 3, 1, 2)
    else:
        targets = torch.tensor(dataset.labels)
    return torch.utils.data.TensorDataset(data, targets)


def gray_transform(dataset):
    data = torch.tensor(dataset.data[:,None].repeat(1,3,1,1)) / 255.0
    targets = torch.tensor(dataset.targets)
    return torch.utils.data.TensorDataset(data, targets)


def get_dataset(dataset_name):
    if dataset_name == "svhn":
        train = datasets.SVHN(
            root=f"data/{dataset_name}/",
            download=True,
            split="train",
        )
        test = datasets.SVHN(
            root=f"data/{dataset_name}/",
            download=True,
            split="test",
        )
        train = rgb_transform(train)
        test = rgb_transform(test)
    elif dataset_name == "cifar10":
        train = datasets.CIFAR10(
            root=f"data/{dataset_name}/",
            download=True,
            train=True,
        )
        test = datasets.CIFAR10(
            root=f"data/{dataset_name}/",
            download=True,
            train=False,
        )
        train = rgb_transform(train)
        test = rgb_transform(test)
    elif dataset_name == "mnist":
        train = datasets.MNIST(
            root=f"data/{dataset_name}/",
            download=True,
            train=True,
        )
        test = datasets.MNIST(
            root=f"data/{dataset_name}/",
            download=True,
            train=False,
        )
        train = gray_transform(train)
        test = gray_transform(test)
    elif dataset_name == "fmnist":
        train = datasets.FashionMNIST(
            root=f"data/{dataset_name}/",
            download=True,
            train=True,
        )
        test = datasets.FashionMNIST(
            root=f"data/{dataset_name}/",
            download=True,
            train=False,
        )
        train = gray_transform(train)
        test = gray_transform(test)
    else:
        raise ValueError("Unknown dataset")

    n_train = int(len(train) * 0.9)
    train, val = torch.utils.data.random_split(train, [n_train, len(train) - n_train])
    return train, val, test


if __name__ == "__main__":
    for dataset_name in ["svhn", "cifar10", "mnist", "fmnist"]:
        train, val, test = get_dataset(dataset_name)
        for data in [train, val, test]:
            dataloader = torch.utils.data.DataLoader(data, batch_size=1)
            batch = next(iter(dataloader))

            assert len(batch) == 2
            assert batch[0].shape[:2] == torch.Size([1, 3])
            assert batch[1].shape == torch.Size([1])
            assert batch[0].min() >= 0.0
            assert batch[0].max() <= 1.0