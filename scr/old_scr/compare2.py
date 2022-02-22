import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import distributions as D
from tqdm import tqdm

from models import get_model
from scr.old_scr import get_data

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    vae_class = get_model("vae")
    vae = vae_class.load_from_checkpoint("wandb/run-20211111_231056-3lrhwvxa/files/checkpoints/best_model.ckpt")
    vae.eval()
    vae.to(device)
    
    mevae_class = get_model("mevae")
    mevae = mevae_class.load_from_checkpoint("wandb/run-20211105_111253-10lb8792/files\checkpoints/best_model.ckpt")
    mevae.to(device)

    mnist01 = get_data("mnist")()
    mnist01.prepare_data()
    mnist01.setup()
    mnist23 = get_data("fmnist")()
    mnist23.prepare_data()
    mnist23.setup()

    base_log_probs_mnist01 = vae.calc_log_prob(mnist01.test_dataloader()).cpu()
    base_log_probs_mnist23 = vae.calc_log_prob(mnist23.test_dataloader()).cpu()

    def agreement(model, dataloader):
        with torch.no_grad():
            entropys, log_probs = [ ], [ ]
            for batch in tqdm(dataloader):
                entropy, log_prob = [], []
                for _ in range(100):
                    x, y = batch
                    _, _, _, x_hat, _ = model.encode_decode(x.to(device))
                    dist = D.Independent(D.Bernoulli(probs=x_hat), 3)
                    entropy.append(dist.entropy())
                    log_prob.append(dist.log_prob(x.to(device)))
                entropy = torch.stack(entropy, dim=0)
                log_prob = torch.stack(log_prob, dim=0)
                entropys.append(entropy.std(dim=0).cpu())
                log_probs.append(log_prob.std(dim=0).cpu())
            return torch.cat(entropys, dim=0), torch.cat(log_probs, dim=0)

    entropy1, log_probs1 = agreement(mevae, mnist01.test_dataloader())
    entropy2, log_probs2 = agreement(mevae, mnist23.test_dataloader())
    n1, n2 = len(entropy1), len(entropy2)
    n = n1 + n2

    dataframe = pd.DataFrame()
    dataframe['base_log_probs'] = torch.cat([base_log_probs_mnist01, base_log_probs_mnist23])
    dataframe['entropy'] = torch.cat([entropy1, entropy2]).numpy()
    dataframe['log_probs'] = torch.cat([log_probs1, log_probs2]).numpy()
    dataframe['dataset'] = n1 * ['mnist01'] + n2 * ['mnist23']
    dataframe['dataset_class'] = n1 * [0] + n2 * [1]

    #plt.figure()
    #sns.histplot(data=dataframe, x='entropy', hue='dataset')

    #plt.figure()
    #sns.histplot(data=dataframe, x='log_probs', hue='dataset')
    
    #plt.show()

#    acc1, acc2 = [ ], [ ]
#    thredshols = np.linspace(0, 10, 1000)
#    for t in thredshols:
#        classifier = dataframe['entropy'] > t
#        acc1.append(
#            (classifier.to_numpy().astype('int') == dataframe['dataset_class'].to_numpy()).mean()
#        )
#        classifier = dataframe['log_probs'] > t
#        acc2.append(
#            (classifier.to_numpy().astype('int') == dataframe['dataset_class'].to_numpy()).mean()
#        )
#    acc1, acc2 = np.array(acc1), np.array(acc2)
#    print(f"best threshold for entropy {thredshols[np.argmax(acc1)]} with acc {np.max(acc1)}")
#    print(f"best threshold for entropy {thredshols[np.argmax(acc2)]} with acc {np.max(acc2)}")

    from sklearn.metrics import RocCurveDisplay, roc_auc_score, roc_curve
    roc_base = roc_curve(dataframe['dataset_class'].to_numpy(), -dataframe['base_log_probs'].to_numpy())
    roc_entropy = roc_curve(dataframe['dataset_class'].to_numpy(), dataframe['entropy'].to_numpy())
    roc_log_probs = roc_curve(dataframe['dataset_class'].to_numpy(), dataframe['log_probs'].to_numpy())
    auroc_base = roc_auc_score(dataframe['dataset_class'].to_numpy(), -dataframe['base_log_probs'].to_numpy())
    auroc_entropy = roc_auc_score(dataframe['dataset_class'].to_numpy(), dataframe['entropy'].to_numpy())
    auroc_roc_log_probs = roc_auc_score(dataframe['dataset_class'].to_numpy(), dataframe['log_probs'].to_numpy())

    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(roc_base[0], roc_base[1], label = f'Base log probs (AUROC={auroc_base})')
    plt.plot(roc_entropy[0], roc_entropy[1], label = f'Agreement in entropy (AUROC={auroc_entropy})')
    plt.plot(roc_log_probs[0], roc_log_probs[1], label = f'Agreement in log probs (AUROC={auroc_roc_log_probs})')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()

    def get_encoding(model, dataloader):
        with torch.no_grad():
            z_mu, z_std = [ ], [ ]
            for batch in tqdm(dataloader):
                out = model.encode(batch[0].to(device), use_all=True)
                z_mu.append(out[0]); z_std.append(out[1])
            z_mu = torch.cat(z_mu, dim=1)
            z_std = torch.cat(z_std, dim=1)
        return z_mu.cpu(), z_std.cpu()
    
    z_mu_in, z_std_in = get_encoding(mevae, mnist01.test_dataloader())
    z_mu_out, z_std_out = get_encoding(mevae, mnist23.test_dataloader())

    plt.figure()
    plt.scatter(z_mu_in[0,:,0], z_mu_in[0,:,1], alpha=0.1, c='g')

    for i, c in enumerate(['navy', 'blue', 'royalblue']):
        plt.scatter(z_mu_in[:,i,0], z_mu_in[:,i,1], s=100*z_std_in.mean(dim=-1)[:,i], facecolors='none', edgecolors=c)
    for i, c in enumerate(['darkred', 'red', 'lightcoral']):
        plt.scatter(z_mu_out[:,i,0], z_mu_out[:,i,1], s=100*z_std_in.mean(dim=-1)[:,i], facecolors='none', edgecolors=c)
    plt.axis([-4, 4, -4, 4])
    plt.grid(True)

    #x_hat_in = mevae(z_mu_in[:3,1,:].to(device), use_all=True).detach().cpu()
    #x_hat_out = mevae(z_mu_out[:3,1,:].to(device), use_all=True).detach().cpu()
    
    #fig, ax = plt.subplots(nrows=3, ncols=10)
    #for i in range(3):
    #    for j in range(10):
    #        ax[i, j].imshow(x_hat_in[j, i, 0], cmap='gray')
    #        ax[i, j].axis('off')

    #fig, ax = plt.subplots(nrows=3, ncols=10)
    #for i in range(3):
    #    for j in range(10):
    #        ax[i, j].imshow(x_hat_out[j, i, 0], cmap='gray')
    #        ax[i, j].axis('off')

    
    n = mevae.hparams.n_ensemble
    fig, ax = plt.subplots(nrows=n, ncols=n)
    for i in range(n):
        decoded = mevae(z_mu_in[i, 0, :].unsqueeze(0).to(device), use_all=True).detach().cpu()
        for j in range(n):
            ax[i, j].imshow(decoded[j, 0, 0], cmap='gray')
            ax[i, j].axis('off')

    fig, ax = plt.subplots(nrows=n, ncols=n)
    for i in range(n):
        decoded = mevae(z_mu_out[i, 1, :].unsqueeze(0).to(device), use_all=True).detach().cpu()
        for j in range(n):
            ax[i, j].imshow(decoded[j, 0, 0], cmap='gray')
            ax[i, j].axis('off')

    plt.show()
    

