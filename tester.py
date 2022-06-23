import torch

recon = [ ]
for i in range(5):
    recon.append([[],[],[],[],[]])
    for _ in range(100):
        for j in range(5):
            recon[-1][j].append(torch.randn(5, 20, 100))

for i in range(5):
    for j in range(5):
        recon[i][j] = torch.cat(recon[i][j], 0)

for i in range(5):
    recon[i] = torch.stack(recon[i], 0)

recon = torch.stack(recon, 0)

recon_f = recon.reshape(-1, *recon.shape[2:])

disagreement = torch.zeros(25, 25)
for i in range(25):
    for j in range(25):
        if i == j:
            continue
        disagreement[i, j] = (recon_f[i].argmax(dim=1) != recon_f[j].argmax(dim=1)).float().sum(1).mean()

score = disagreement.sum() / (disagreement.numel() - 25)  # subtract the diagonal





import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
xx, yy = np.meshgrid(np.arange(-10, 10, 0.1), np.arange(-10, 10, 0.1))

fig, ax = plt.subplots(1, 3)

for i, name in enumerate(['means', 'samples', 'ensemble']):
    x, y = torch.randn(1000, 2), torch.randint(10, (1000,))
    classifier = KNeighborsClassifier(n_neighbors=7)
    classifier.fit(x, y)
    z = classifier.predict(np.c_(xx.ravel(), yy.ravel()))
    z = z.reshape(xx.shape)

    ax[i].contourf(xx, yy, z, alpha=0.4)

plt.show()
