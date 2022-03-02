import torch
from torchvision import datasets, transforms
from timeit import default_timer as timer
from datetime import timedelta


def inside():
    cifar = datasets.CIFAR10(root='data/cifar10', transform=transforms.ToTensor())
    dl = torch.utils.data.DataLoader(cifar, batch_size=64)
    for batch in dl:
        pass

def outside():
    cifar = datasets.CIFAR10(root='data/cifar10')
    data = torch.tensor(cifar.data)
    targets = torch.tensor(cifar.targets)
    cifar = torch.utils.data.TensorDataset(data, targets)
    dl = torch.utils.data.DataLoader(cifar, batch_size=64)
    for batch in dl:
        pass
    
if __name__ == '__main__':    
    for f in [inside, outside]:
        start = timer()
        f()
        end = timer()
        print(timedelta(seconds=end-start))
    