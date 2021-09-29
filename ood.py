from pytorch_lightning.core import datamodule
import torch
from torch import distributions as D
import seaborn as sns
import numpy as np

def run_ood(model, datamodule):
    log_probs = []
    for batch in datamodule1:
        x, y = batch
        x_out, z_mu, z_std, q_dist = model.encode_decode(x)
        log_prob = D.Independent(D.Bernoulli(probs=x_out), 3).log_prob(x)
        log_probs.append(log_prob)

    log_probs = torch.stack(log_probs)
    return log_probs

if __name__ == "__main__":
    

    model = model_class.load_from_checkpoint(args.model_checkpoint)
    model.eval()

    datamodule1 = MnistDatamodule(args.data_dir, args.labels_to_use)
    datamodule2 = MnistDatamodule(args.data_dir, args.labels_to_use)
    scores1 = run_ood(model, datamodule1)
    scores2 = run_ood(model, datamodule2)

    torch.save(scores1, 'results1.pt')
    torch.save(scores2, 'results2.pt')

    if args.plot_this:
        combined = np.array([scores1, scores2])

        sns.histplot(combined, x='log prob', y='Procentage samples')



        

    





