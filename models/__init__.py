from .double_ensample import DEVAE
from .ensample import EVAE
from .mc_dropout import MCVAE
from .mixer_ensample import MEVAE
from .normal_vae import NVAE
from .vae import VAE


def get_model(name: str):
    return {"VAE": VAE, "NVAE": NVAE, "MCVAE": MCVAE, "EVAE": EVAE, "DEVAE": DEVAE, "MEVAE": MEVAE}[
        name.upper()
    ]
