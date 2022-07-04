import enum
from protein_vae import TOKEN_SIZE, MixVAE
import pytorch_lightning as pl
import torch
from scr.data import get_dataset, important_organisms
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse
import torch.distributions as D
from torch import nn
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

train, val, test, _ = get_dataset('protein')
train_dl = torch.utils.data.DataLoader(train, batch_size=32)
val_dl = torch.utils.data.DataLoader(val, batch_size=32)
test_dl = torch.utils.data.DataLoader(test, batch_size=32)

model = MixVAE()
model.load_state_dict(torch.load("models/generative_models/MixVAE.pt"))

trainer = pl.Trainer(logger=False)
predictions = trainer.predict(model, dataloaders=train_dl)
embeddings = torch.cat([p['encoder_mean'] for p in predictions], dim=0)
labels = torch.cat([p['label'] for p in predictions], dim=0)

fig = plt.figure()
def plot_embeddings(ax=None, alpha=1.0):
    if ax is None:
        ax = plt
    for name in important_organisms:
        current_label = important_organisms[name]
        ax.plot(
            embeddings[labels == current_label, 0],
            embeddings[labels == current_label, 1],
            '.',
            alpha=alpha,
        )
plot_embeddings()

in_datapoint = train[0][0].reshape(1, -1)
idx = torch.where(in_datapoint != 22)[1]
n = len(idx)
r_idx = torch.randint(n, (int(n * 0.8),))
out_datapoint = in_datapoint.clone()
out_datapoint[0][idx[r_idx]] = torch.randint(0, 20, (len(r_idx),)).int()
# out_datapoint = torch.randint(0, 23, in_datapoint.shape)

def get_full_encoding(datapoint):
    with torch.no_grad():
        x = nn.functional.one_hot(datapoint.long(), TOKEN_SIZE)
        m, s = [ ], [ ]
        for i in range(5):
            h = model.encoder[i](x.float().reshape(x.shape[0], -1))
            m.append(model.encoder_mu[i](h))
            s.append(model.encoder_scale[i](h))
    return m, s

m_in, s_in = get_full_encoding(in_datapoint)
m_out, s_out = get_full_encoding(out_datapoint)

fig, ax = plt.subplots(1, 1)
plot_embeddings(ax, alpha=0.1)
for l, c, m, s in zip(['in', 'out'], ['red', 'blue'], [m_in, m_out], [s_in, s_out]):
    for mean, std in zip(m, s):
        ax.plot(mean[0][0].item(), mean[0][1].item(), 'x', color=c, label=l)
        e = Ellipse(
            (mean[0][0].item(), mean[0][1].item()),
            width=std[0][0].item(),
            height=std[0][1].item(),
            linewidth=2,
            fill=False,
            alpha=0.5,
            color=c
        )
        ax.add_patch(e)
ax.legend()
ax.set_xlim(-6,6)
ax.set_ylim(-6,6)
ax.set_title("In/out distribution encoding")

fig, ax = plt.subplots(1, 5)
with torch.inference_mode():
    decoded = [ ]
    for i in range(5):
        embeddings_i, labels_i = [ ], [ ]
        decoded.append([[ ], [ ], [ ], [ ], [ ]])
        for batch_idx, batch in tqdm(enumerate(train_dl), desc=f'Decoding using decoder {i}', total=len(train_dl)):
            x, y = batch
            z_mu, _ = model.encode(x.long(), i)
            for j in range(5):
                if batch_idx > 50:
                    break
                recon = model.decoder[j](z_mu)
                decoded[-1][j].append(recon.reshape(*z_mu.shape[:-1], TOKEN_SIZE, -1).argmax(dim=1))
            embeddings_i.append(z_mu)
            labels_i.append(y)
        embeddings_i = torch.cat(embeddings_i, dim=0)
        labels_i = torch.cat(labels_i, dim=0)
        for name in important_organisms:
            current_label = important_organisms[name]
            ax[i].plot(
                embeddings_i[labels_i == current_label, 0],
                embeddings_i[labels_i == current_label, 1],
                '.',
            )
            ax[i].set_title(f"Encoder {i+1}")

for i in range(5):
    for j in range(5):
        decoded[i][j] = torch.cat(decoded[i][j], 0)
    decoded[i] = torch.stack(decoded[i], 0)
decoded = torch.stack(decoded, 0)
decoded_f = decoded.reshape(25, *decoded.shape[2:])

disagreement = torch.zeros(25, 25)
for i in range(25):
    for j in range(25):
        disagreement[i,j] = (decoded_f[i] != decoded_f[j]).float().sum(1).mean()
        disagreement[i,j] = float("-inf") if i == j else disagreement[i,j]
fig = plt.figure()
img = plt.imshow(disagreement)
plt.text(
    0.0, 0.1, 
    '\n'.join([f"E={i+1},D={j+1}" for i in range(5) for j in range(5)]), 
    fontsize=10, transform=plt.gcf().transFigure
)
plt.title("Disagreement score")
plt.colorbar()
plt.subplots_adjust(left=0.2)

del decoded, decoded_f

with torch.inference_mode():
    plot_bound_x = (1.1*embeddings[:,0].min(), 1.1*embeddings[:,0].max())
    plot_bound_y = (1.1*embeddings[:,1].min(), 1.1*embeddings[:,1].max())
    n_points = 65
    linspaces = [
        torch.linspace(plot_bound_x[0], plot_bound_x[1], n_points),
        torch.linspace(plot_bound_y[0], plot_bound_y[1], n_points),
    ]
    meshgrid = torch.meshgrid(linspaces)
    z_sample = torch.stack(meshgrid).reshape(2, -1).T

    samples = [ ]
    print('Decoding z_grid')
    for i in range(5):
        x_out = model.decoder[i](z_sample).reshape(*z_sample.shape[:-1], TOKEN_SIZE, -1)
        samples.append(x_out)

    entropy = torch.zeros(z_sample.shape[0], 5)
    for i in range(5):
        dist = D.Independent(D.Categorical(probs=(samples[i] / 10).softmax(dim=1).permute(0, 2, 1)), 1)
        entropy[:, i] = dist.entropy()
    entropy_mean_score = entropy.mean(dim=-1)
    entropy_std_score = entropy.std(dim=-1)
    entropy_std_score[torch.isnan(entropy_std_score)] = 0.0

    kl = torch.zeros(z_sample.shape[0], 5, 5)
    for i in range(5):
        for j in range(5):
            if i==j:
                continue
            # TODO: this temperature scaling is not ideal
            dist1 = D.Independent(D.Categorical(probs=(samples[i] / 10).softmax(dim=1).permute(0, 2, 1)), 1)
            dist2 = D.Independent(D.Categorical(probs=(samples[j] / 10).softmax(dim=1).permute(0, 2, 1)), 1)
            kl[:, i, j] = D.kl_divergence(dist1, dist2)
    kl_mean_score = kl.mean(dim=[-1, -2])
    kl_std_score = kl.std(dim=[-1, -2])
    kl_std_score[torch.isnan(kl_std_score)] = 0.0
    
    disagreement = torch.zeros(z_sample.shape[0], 5, 5)
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            preds1 = samples[i].argmax(dim=1)
            preds2 = samples[j].argmax(dim=1)
            disagreement[:, i, j] = (preds1 != preds2).float().sum(dim=1)
    disagreement_mean_score = disagreement.mean(dim=[-1, -2])
    disagreement_std_score = disagreement.std(dim=[-1, -2])
    disagreement_std_score[torch.isnan(disagreement_std_score)] = 0.0

    for (score, name) in [
        (entropy_mean_score, 'entropy_mean'),
        (entropy_std_score, 'entropy_std'),
        (kl_mean_score, 'kl_mean'),
        (kl_std_score, 'kl_std'),
        (disagreement_mean_score, 'disagreement_mean'),
        (disagreement_std_score, 'disagreement_std'),
    ]:
        fig = plt.figure()
        plot_embeddings(alpha=0.5)
        plt.contourf(
            z_sample[:, 0].reshape(n_points, n_points),
            z_sample[:, 1].reshape(n_points, n_points),
            score.reshape(n_points, n_points).detach().cpu(),
            levels=50,
            zorder=0,
        )
        plt.colorbar()
        plt.title(name)




    # fig = plt.figure()
    # plot_embeddings(alpha=0.5)
    # plt.contourf(
    #     z_sample[:, 0].reshape(n_points, n_points),
    #     z_sample[:, 1].reshape(n_points, n_points),
    #     x_var.reshape(n_points, n_points).detach().cpu(),
    #     levels=50,
    #     zorder=0,
    # )
    # plt.colorbar()

    # fig = plt.figure()
    # plot_embeddings(alpha=0.5)
    # plt.contourf(
    #     z_sample[:, 0].reshape(n_points, n_points),
    #     z_sample[:, 1].reshape(n_points, n_points),
    #     x_var_std.reshape(n_points, n_points).detach().cpu(),
    #     levels=50,
    #     zorder=0,
    # )

    # plt.colorbar()
    
    # fig = plt.figure()
    # plot_embeddings(alpha=0.5)
    # plt.contourf(
    #     z_sample[:, 0].reshape(n_points, n_points),
    #     z_sample[:, 1].reshape(n_points, n_points),
    #     disagreement_score.reshape(n_points, n_points).detach().cpu(),
    #     levels=50,
    #     zorder=0,
    # )
    # plt.title('Disagreement score')
    # plt.colorbar()

# with torch.inference_mode():
#     fig, ax = plt.subplots(1, 3)

#     for i, name in enumerate(['means', 'samples', 'ensemble']):
#         embeddings, labels = [ ], [ ]
#         for batch_idx, batch in tqdm(enumerate(train_dl)):
#             if batch_idx > 400:
#                 break
#             x, y = batch
#             if name == 'means':
#                 emb, _ = model.encode(x, 0)
#             elif name == 'samples':
#                 z_mu, z_std = model.encode(x, 0)
#                 emb = D.Independent(D.Normal(z_mu, z_std + 1e-4), 1).sample()
#             elif name == 'ensemble':
#                 z_mu, z_std = model.encode(x)
#                 emb = D.Independent(D.Normal(z_mu, z_std + 1e-4), 1).sample()
#             else:
#                 raise ValueError()
#             embeddings.append(emb)
#             labels.append(y)
            
#         embeddings = torch.cat(embeddings, dim=0)
#         labels = torch.cat(labels, dim=0)
        
#         classifer = KNeighborsClassifier()
#         classifer.fit(embeddings[::2], labels[::2])
#         out = classifer.predict(z_sample)
#         c = ax[i].contourf(
#             z_sample[:, 0].reshape(n_points, n_points),
#             z_sample[:, 1].reshape(n_points, n_points),
#             out.reshape(n_points, n_points)
#         )
#         preds = classifer.predict(embeddings[1::2])
#         true = labels[1::2].numpy()
#         print(f"{name} - Acc: {(preds==true).mean()}")
        
#     plt.colorbar(c)

# with torch.inference_mode():
#     from stochman.discretized_manifold import DiscretizedManifold

#     print("Constructing discretized manifold")
#     manifold = DiscretizedManifold()
#     manifold.fit(
#         model=model,
#         grid=linspaces,
#         batch_size=16
#     )

#      fig, ax = plt.subplots(2, 5)
#      for i in range(10):
#          plot_embeddings(ax[i%2, i%5], alpha=0.3)
#          idx1, idx2 = D.Categorical(embeddings.norm(dim=1)).sample((2,))
#          curve, dist = manifold.connecting_geodesic(
#              embeddings[idx1,:].unsqueeze(0),
#              embeddings[idx2,:].unsqueeze(0),
#          )
#          ax[i%2, i%5].plot(embeddings[idx1,0], embeddings[idx1,1], 'rx')
#          ax[i%2, i%5].plot(embeddings[idx2,0], embeddings[idx2,1], 'bx')
#          curve.plot(ax=ax[i%2,i%5])

# fig, ax = plt.subplots(1, 1)
# plot_embeddings(alpha=0.2)
# idx1, idx2 = D.Categorical(embeddings.norm(dim=1)).sample((500,)).chunk(2)
# for i, j in zip(idx1, idx2):
#     curve, _ = manifold.connecting_geodesic(
#         embeddings[i, :].unsqueeze(0),
#         embeddings[j, :].unsqueeze(0),
#     )
#     curve.plot(ax=ax, c='magenta', alpha=0.5)

# plt.show()
