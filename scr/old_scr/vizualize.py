import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    vae_mnist01 = pd.read_csv('vae_mnist01.csv')
    vae_mnist23 = pd.read_csv('vae_mnist23.csv')
    mevae_mnist01 = pd.read_csv('mevae_mnist01.csv')
    mevae_mnist23 = pd.read_csv('mevae_mnist23.csv')

    vae_mnist01['model'] = 'vae'
    vae_mnist23['model'] = 'vae'
    mevae_mnist01['model'] = 'mevae'
    mevae_mnist23['model'] = 'mevae'

    mnist01 = pd.concat([vae_mnist01, mevae_mnist01], ignore_index=True)
    mnist23 = pd.concat([vae_mnist23, mevae_mnist23], ignore_index=True)

    fig, ax = plt.subplots()
    sns.kdeplot(data=mnist01, x='log_probs', hue='model', ax=ax)
    
    fig, ax = plt.subplots()
    sns.kdeplot(data=mnist23, x='log_probs', hue='model', ax=ax)
    
    plt.show()

#    fig, ax = plt.subplots()
#    sns.kdeplot(data=vae_mnist01, x='log_probs', hue='split', ax=ax, ls='--')
#    sns.kdeplot(data=vae_mnist01, x='log_probs_refit', hue='split', ax=ax)
#    sns.kdeplot(data=vae_mnist01, x='mixture_score', hue='split', ax=ax, ls=':')

#    fig, ax = plt.subplots()
#    sns.kdeplot(data=vae_mnist23, x='log_probs', hue='split', ax=ax, ls='--')
#    sns.kdeplot(data=vae_mnist23, x='log_probs_refit', hue='split', ax=ax)
#    sns.kdeplot(data=vae_mnist23, x='mixture_score', hue='split', ax=ax, ls=':')

#    fig, ax = plt.subplots()
#    sns.kdeplot(data=mevae_mnist01, x='log_probs', hue='split', ax=ax, ls='--')
#    sns.kdeplot(data=mevae_mnist01, x='log_probs_refit', hue='split', ax=ax)
#    sns.kdeplot(data=mevae_mnist01, x='mixture_score', hue='split', ax=ax, ls=':')

#    fig, ax = plt.subplots()
#    sns.kdeplot(data=mevae_mnist23, x='log_probs', hue='split', ax=ax, ls='--')
#    sns.kdeplot(data=mevae_mnist23, x='log_probs_refit', hue='split', ax=ax)
#    sns.kdeplot(data=mevae_mnist23, x='mixture_score', hue='split', ax=ax, ls=':')

#    plt.show()
