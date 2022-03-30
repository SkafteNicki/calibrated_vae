from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

def rgb_transform(dataset: Dataset) -> Dataset:
    data = torch.tensor(dataset.data) / 255.0
    if hasattr(dataset, 'targets'):
        targets = torch.tensor(dataset.targets)
        data = data.permute(0, 3, 1, 2)
    else:
        targets = torch.tensor(dataset.labels)
    return torch.utils.data.TensorDataset(data, targets)


def gray_transform(dataset: Dataset) -> Dataset:
    data = torch.tensor(dataset.data[:,None].repeat(1,3,1,1)) / 255.0
    targets = torch.tensor(dataset.targets)
    return torch.utils.data.TensorDataset(data, targets)


def get_dataset(dataset_name: str) -> Tuple[Dataset, Dataset, Dataset, int]:
    n_labels = 10
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
    elif dataset_name == "cifar100":
        train = datasets.CIFAR100(
            root=f"data/{dataset_name}/",
            download=True,
            train=True,
        )
        test = datasets.CIFAR100(
            root=f"data/{dataset_name}/",
            download=True,
            train=False,
        )
        train = rgb_transform(train)
        test = rgb_transform(test)
        n_labels = 100
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
    elif dataset_name == "celeba":
        train = datasets.CelebA(
            root=f"data/{dataset_name}/",
            split="train",
            download=True,
        )
        val = datasets.CelebA(
            root=f"data/{dataset_name}/",
            split="valid",
            download=True,
        )
        test = datasets.CelebA(
            root=f"data/{dataset_name}/",
            split="test",
            download=True,
        )
    elif dataset_name == "protein":
        from Bio import SeqIO
        import numpy as np
        import re
        import pickle as pkl
        import os
        aa1_to_index = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6,
                        'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12,
                        'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
                        'Y': 19, 'X':20, 'Z': 21, '-': 22, '.': 22}
        important_organisms = {
            'Acidobacteria': 0, 'Actinobacteria': 1, 'Bacteroidetes': 2,
            'Chloroflexi': 3, 'Cyanobacteria': 4, 'Deinococcus-Thermus': 5,
            'Firmicutes': 6, 'Fusobacteria': 7, 'Proteobacteria': 8
        }
        if 'processed_data.pkl' not in os.listdir('data/protein'):
            seqs = [ ]
            labels = [ ]
            ids1, ids2 = [ ], [ ]
            for record in SeqIO.parse('data/protein/PF00144_full.txt', 'fasta'):
                seqs.append(np.array([aa1_to_index[aa] for aa in str(record.seq).upper()]))
                ids1.append(re.findall(r'.*\/',record.id)[0][:-1])
            d1 = dict([(i,s) for i,s in zip(ids1, seqs)])
            for record in SeqIO.parse("data/protein/PF00144_full_length_sequences_labeled.fasta", 'fasta'):
                ids2.append(record.id)
                labels.append(re.findall(r'\[.*\]', record.description)[0][1:-1])
            d2 = dict([(i,l) for i,l in zip(ids2, labels)])

            data = [ ]
            for key in d1.keys():
                if key in d2.keys() and d2[key] in important_organisms:
                    data.append([d1[key], d2[key]])
            with open('data/protein/processed_data.pkl', 'wb') as file:
                pkl.dump(data, file)
        else:
            with open('data/protein/processed_data.pkl', 'rb') as file:
                data = pkl.load(file)

        seqs = torch.tensor(np.array([d[0] for d in data]))
        labels = torch.tensor(np.array([important_organisms[d[1]] for d in data]))

        n_total = len(seqs)
        idx = np.random.permutation(n_total)
        n_train = int(0.9*n_total)
        n_val = int(0.05*n_total)
        train = torch.utils.data.TensorDataset(seqs[idx[:n_train]], labels[idx[:n_train]])
        val = torch.utils.data.TensorDataset(seqs[idx[n_train:n_train+n_val]], labels[idx[n_train:n_train+n_val]])
        test = torch.utils.data.TensorDataset(seqs[idx[n_train+n_val:]], labels[idx[n_train+n_val:]])
        
    else:
        raise ValueError("Unknown dataset")

    if dataset_name not in ("celeba", "protein"):
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