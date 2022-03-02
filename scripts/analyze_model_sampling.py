import argparse
import pickle as pkl

import numpy as np
import torch
from torchmetrics.functional import accuracy
import matplotlib.pyplot as plt

from scr.classification_models import MixLayerEnsembles
from scr.data import get_dataset
from scr.utils import brierscore

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("weight_file")
    parser.add_argument("test_data", default=None)
    args = parser.parse_args()

    if 'mixlayer' in args.weight_file.lower():
        n_ensemble = int(args.weight_file[:-3].split('_')[-1])
        model = MixLayerEnsembles(n_ensemble)
        model.load_state_dict(torch.load(args.weight_file))

    _, _, test_data = get_dataset(args.test_data)
    test_data = torch.utils.data.Subset(
        test_data, list(np.random.permutation(1000))
    )
    test = torch.utils.data.DataLoader(test_data, batch_size=20)

    scores = {'sampling_size': [ ], 'acc': [ ], 'nll': [ ], 'brier': [ ], 'mode_acc': [ ]}
    for sampling_size in [2, 5, 10, 20, 50, 100]:
        print(f"Testing size {sampling_size}")
        acc, nll, brier, mode_acc = 0.0, 0.0, 0.0, 0.0
        for batch in test:
            x, y = batch
            log_probs = torch.stack([model(x) for _ in range(sampling_size)])
            mean = log_probs.mean(dim=0)

            acc += accuracy(mean, y, num_classes=10).item()
            nll += torch.nn.functional.nll_loss(mean, y).item()
            brier += brierscore(mean.softmax(dim=-1), y).item()

            preds = log_probs.argmax(dim=-1)
            corrects = torch.mode(preds, axis=0)[0] == y
            mode_acc += (sum(corrects) / len(corrects)).item()

        acc /= len(test)
        nll /= len(test)
        brier /= len(test)
        mode_acc /= len(test)
        print(acc, nll, brier, mode_acc)

        scores['sampling_size'].append(sampling_size)
        scores['acc'].append(acc)
        scores['nll'].append(nll)
        scores['brier'].append(brier)
        scores['mode_acc'].append(mode_acc)

    fig, ax = plt.subplots(nrows=1, ncols=4)
    ax[0].plot(scores['sampling_size'], scores['acc'])
    ax[1].plot(scores['sampling_size'], scores['nll'])
    ax[2].plot(scores['sampling_size'], scores['brier'])
    

