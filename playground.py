from copy import deepcopy
from distutils.ccompiler import new_compiler
import torch
from torchvision import datasets, transforms
from timeit import default_timer as timer
from datetime import timedelta
import pickle 
import numpy as np
from scr.utils import get_all_combinations
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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

def main1():
    for f in [inside, outside]:
        start = timer()
        f()
        end = timer()
        print(timedelta(seconds=end-start))


if __name__ == '__main__':
    with open('results/analyze_weight_space/cifar10_MixLayerEnsembles_3_embeddings.pkl', 'rb') as file:
        emb = pickle.load(file)
    with open('results/analyze_weight_space/cifar10_MixLayerEnsembles_3_results.pkl', 'rb') as file:
        sim, dis, dis2 = pickle.load(file)

    combinations = get_all_combinations(3, 4)

    def where(comb): return np.argmax(np.sum(np.array(combinations) == comb, axis=1))

    
    base = [0,0,0,0]
    for i in range(4):
        idx = [ ]
        for j in range(3):
            temp = deepcopy(base)
            temp[i] = j
            idx.append(temp)
        
        print(idx)
        idx = [where(i) for i in idx]
        print(idx)
    
        tsne = TSNE(n_components=2)
        X = tsne.fit_transform(torch.cat([emb[i] for i in idx]))

        color = plt.cm.rainbow(np.linspace(0, 1, len(idx)))
        plt.figure()
        for i,c in zip(range(len(idx)), color):
            plt.plot(X[1000*i:1000*(i+1), 0], X[1000*i:1000*(i+1), 1], '.', color=c)
    
    base = [0,0,0,0]
    for k in range(1,3):
        comb = [ ]
        for i in range(4):
            temp = deepcopy(base)
            temp[i] = k
            comb.append((base, temp))

        c = np.array([(where(c[0]), where(c[1])) for c in comb])
        for i, (idx1, idx2) in enumerate(c):
            print(comb[i], sim[idx1, idx2], dis[idx1, idx2], dis2[idx1, idx2])
        print()

    plt.show()
    
    
        




#    [0,0,0,0], [0,0,0,1], [0,0,0,2]




#e = embeddings.reshape(81, 50, batch_size, 512)
#e = e.reshape(81, -1, 512)
#tsne2 = TSNE(n_components=2)
#X=tsne2.fit_transform(torch.cat([e[i] for i in idx], 0))


"""
idx = [1, 2, 3, 4, 5]


for i,c in zip(range(len(idx)), color):
    plt.plot(X[1000*i:1000*(i+1), 0], X[1000*i:1000*(i+1), 1], '.', color=c)
"""