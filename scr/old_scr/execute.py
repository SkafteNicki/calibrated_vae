import os

import matplotlib.pyplot as plt
import torch

from models import get_model

if __name__ == "__main__":
#    for seed in range(5):
#        os.system(f'python train.py VAE --gpus 1 --gradient_clip_val 1.0 --seed {seed}')
    vae_class = get_model("vae")
    
    vaes = [ ]
    for path in ["wandb/run-20211105_140631-11nwlscd/files/checkpoints/best_model.ckpt",
                 "wandb/run-20211105_141515-e0mj9pcp/files/checkpoints/best_model.ckpt",
                 "wandb/run-20211105_142044-yr6nd74u/files/checkpoints/best_model.ckpt",
                 "wandb/run-20211105_142807-1w6evb3b/files/checkpoints/best_model.ckpt",
                 "wandb/run-20211105_143409-3ka5eyk1/files/checkpoints/best_model.ckpt"]:
        
        vae = vae = vae_class.load_from_checkpoint(path)
        vae.eval()
        vaes.append(vae)

    plot_bound = -7.0
    n_points = 30
    linspaces = [torch.linspace(-plot_bound, plot_bound, n_points) for _ in range(2)]
    meshgrid = torch.meshgrid(linspaces)
    z_sample = torch.stack(meshgrid).reshape(2, -1).T

    out = [ ]
    for vae in vaes:
        x_out = vae(z_sample)
        x_var = torch.distributions.Bernoulli(probs=x_out).entropy().sum(dim=[1, 2, 3])
        out.append(x_var)
    out = torch.stack(out, dim=0)
    var = out.std(dim=0)

    plt.contourf(
        z_sample[:, 0].reshape(n_points, n_points),
        z_sample[:, 1].reshape(n_points, n_points),
        var.reshape(n_points, n_points).detach().cpu(),
        levels=50,
        zorder=0,
    )
    plt.show()
