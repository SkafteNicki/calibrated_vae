import argparse
import os

import numpy as np
import torch
from torchmetrics.functional import accuracy
from torchmetrics import CalibrationError
import matplotlib.pyplot as plt

from scr.classification_models import get_classification_model_from_file
from scr.data import get_dataset
from scr.utils import brierscore


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("results/", exist_ok=True)
    with open("results/sampling_scores.txt", "w") as file:
        file.write("weight_file, sampling_size, acc, nll, brier, calibration\n")
    file_extension = args.weight_file[:-3].split("/")[-1]

    parser = argparse.ArgumentParser()
    parser.add_argument("weight_file")
    parser.add_argument("test_data", default=None)
    parser.add_argu
    args = parser.parse_args()

    model = get_classification_model_from_file(args.weight_file)
    model.to(device)

    _, _, test_data = get_dataset(args.test_data)
    test_data = torch.utils.data.Subset(test_data, list(np.random.permutation(1000)))
    test = torch.utils.data.DataLoader(test_data, batch_size=20)

    reps = 5
    scores = {"sampling_size": [], "acc": [], "nll": [], "brier": [], "calibration": []}
    for sampling_size in [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        print(f"Testing size {sampling_size}")

        scores["sampling_size"].append(sampling_size)
        scores["acc"].append([])
        scores["nll"].append([])
        scores["brier"].append([])
        scores["calibration"].append([])

        for r in range(reps):
            acc, nll, brier = 0.0, 0.0, 0.0
            c = CalibrationError()

            for batch in test:
                with torch.no_grad():
                    x, y = batch
                    log_probs = torch.stack(
                        [model(x.to(device)) for _ in range(sampling_size)]
                    ).cpu()
                    mean = log_probs.mean(dim=0)

                    acc += accuracy(mean, y, num_classes=10).item()
                    nll += torch.nn.functional.nll_loss(mean, y).item()
                    brier += brierscore(mean.softmax(dim=-1), y).item()
                    c.update(mean, y)

            acc /= len(test)
            nll /= len(test)
            brier /= len(test)
            calibration = c.compute().item()
            print(
                f"accuracy={acc:0.4f}, nll={nll:0.4f}, brier={brier:0.4f}, calibration={calibration:0.4f}"
            )

            scores["acc"][-1].append(acc)
            scores["nll"][-1].append(nll)
            scores["brier"][-1].append(brier)
            scores["calibration"][-1].append(calibration)

        with open("results/sampling_scores.txt", "a") as file:
            file.write(
                f"{file_extension}, {sampling_size}, {scores[-1]['acc']}, {scores[-1]['nll']}, {scores[-1]['brier']}, {scores[-1]['calibration']} \n"
            )

    os.makedirs("figures/sampling_performance/", exist_ok=True)
    for name in ["acc", "nll", "brier", "calibration"]:
        fig = plt.figure()
        score_mean = np.array([np.mean(s) for s in scores[name]])
        score_std = np.array([np.std(s) for s in scores[name]])

        plt.plot(scores["sampling_size"], score_mean)
        plt.fill_between(
            scores["sampling_size"],
            score_mean - 3 * score_std,
            score_mean + 3 * score_std,
            alpha=0.1,
        )
        plt.xlabel("Sampling size")
        plt.ylabel(name)

        fig.savefig(
            f"figures/sampling_performance/{name}_{file_extension}.png",
            bbox_inches="tight",
        )
        fig.savefig(
            f"figures/sampling_performance/{name}_{file_extension}.svg",
            bbox_inches="tight",
        )
