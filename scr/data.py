import torch
from torchvision import datasets, transforms

rbg_transform = lambda : transforms.ToTensor()
gray_transform = lambda : transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

def get_dataset(dataset_name):
    if dataset_name == "svhn":
        train = datasets.SVHN(
            root=f"data/{dataset_name}/",
            download=True,
            split="train",
            transform=rbg_transform(),
        )
        test = datasets.SVHN(
            root=f"data/{dataset_name}/",
            download=True,
            split="test",
            transform=rbg_transform(),
        )
    elif dataset_name == "cifar10":
        train = datasets.CIFAR10(
            root=f"data/{dataset_name}/",
            download=True,
            train=True,
            transform=rbg_transform(),
        )
        test = datasets.CIFAR10(
            root=f"data/{dataset_name}/",
            download=True,
            train=False,
            transform=rbg_transform(),
        )
    elif dataset_name == "mnist":
        train = datasets.MNIST(
            root=f"data/{dataset_name}/",
            download=True,
            train=True,
            transform=gray_transform()
        )
        test = datasets.MNIST(
            root=f"data/{dataset_name}/",
            download=True,
            train=False,
            transform=gray_transform(),
        )
    elif dataset_name == "fmnist":
        train = datasets.FashionMNIST(
            root=f"data/{dataset_name}/",
            download=True,
            train=True,
            transform=gray_transform(),
        )
        test = datasets.FashionMNIST(
            root=f"data/{dataset_name}/",
            download=True,
            train=False,
            transform=gray_transform(),
        )
    else:
        raise ValueError("Unknown dataset")

    n_train = int(len(train) * 0.9)
    train, val = torch.utils.data.random_split(train, [n_train, len(train) - n_train])
    return train, val, test


if __name__ == "__main__":
    for dataset_name in ['svhn', 'cifar10', 'mnist', 'fmnist']:
        train, val, test = get_dataset(dataset_name)
        for data in [train, val, test]:
            dataloader = torch.utils.data.DataLoader(data, batch_size=1)
            batch = next(iter(dataloader))

            assert len(batch) == 2
            assert batch[0].shape[:2] == torch.Size([1, 3])
            assert batch[1].shape == torch.Size([1])