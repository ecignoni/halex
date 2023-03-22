import os

import numpy as np
import scipy  # noqa

# non so perche devo importare cdist da scipy prima di halex e non lo voglio sapere
# se non lo faccio, su newtimes ****** non funziona
# in qualunque altro posto si
from scipy.spatial.distance import cdist  # noqa

from halex.utils import tensormap_as_torch, drop_target_samples
from halex.rotations import ClebschGordanReal
from halex.hamiltonian import (
    compute_ham_features,
    drop_unused_features,
)
from halex.decomposition import EquivariantPCA
from halex.dataset import SCFData, BatchedMemoryDataset
from halex.models import RidgeOnEnergiesAndLowdinByMO

import torch

torch.set_default_dtype(torch.float64)


SERVER = "cosmo-workstation"


def set_root_dir(server):
    if server == "cosmo-workstation":
        root_dir = "/local/scratch/cignoni/"
    elif server == "newtimes":
        root_dir = "/home/e.cignoni/"
    else:
        raise ValueError("server not known.")
    return root_dir


ROOT_DIR = set_root_dir(SERVER)


if __name__ == "__main__":
    cg = ClebschGordanReal(4)

    # ==================================================================
    # Load data in small basis: we predict a Fock of the same dimension
    # ==================================================================

    # butadiene STO-3G
    sto3g_dir = os.path.join(
        ROOT_DIR,
        "HamiltonianLearningEPFL/datasets/butadiene/b3lypg_sto3g/out_spherical/",
    )
    sto3g = SCFData(
        frames=os.path.join(
            ROOT_DIR,
            "HamiltonianLearningEPFL/datasets/butadiene/coords_butadiene_1000.xyz",
        ),
        focks=os.path.join(sto3g_dir, "focks.npy"),
        ovlps=os.path.join(sto3g_dir, "ovlps.npy"),
        orbs=os.path.join(sto3g_dir, "orbs.json"),
        cg=cg,
        max_frames=200,
    )

    # ==================================================================
    # Load data in big basis: we use its properties as target
    # ==================================================================

    # butadiene def2-SVP
    def2svp_dir = os.path.join(
        ROOT_DIR,
        "HamiltonianLearningEPFL/datasets/butadiene/b3lypg_def2svp/out_spherical/",
    )
    def2svp = SCFData(
        frames=os.path.join(
            ROOT_DIR,
            "HamiltonianLearningEPFL/datasets/butadiene/coords_butadiene_1000.xyz",
        ),
        focks=os.path.join(def2svp_dir, "focks.npy"),
        ovlps=os.path.join(def2svp_dir, "ovlps.npy"),
        orbs=os.path.join(def2svp_dir, "orbs.json"),
        cg=cg,
        max_frames=200,
    )

    # ==================================================================
    # Compute the Features
    # ==================================================================

    # compute features
    rascal_hypers = {
        "interaction_cutoff": 2.0,  # to never compute features between the two ends C
        "cutoff_smooth_width": 0.5,
        "max_radial": 6,
        "max_angular": 4,
        "gaussian_sigma_constant": 0.2,
        "gaussian_sigma_type": "Constant",
        "compute_gradients": False,
    }

    feats = tensormap_as_torch(
        compute_ham_features(rascal_hypers, frames=sto3g.frames, cg=cg, lcut=2)
    )
    feats = drop_unused_features(feats, targs_coupled=sto3g.focks_orth_tmap_coupled)

    # ==================================================================
    # Clean the features with PCA
    # ==================================================================

    # equivariant PCA
    epca = EquivariantPCA(n_components=200).fit(feats)
    feats = epca.transform(feats)

    # ==================================================================
    # Take what's needed
    # ==================================================================

    # frames
    frames = sto3g.frames

    # mo_energy
    mo_energy = def2svp.mo_energy[:, : sto3g.mo_energy.shape[-1]]

    # lowdin charges
    lowdin_charges = def2svp.lowdin_charges_byMO

    # orbitals per atom species
    orbs = sto3g.orbs

    # ao_labels
    ao_labels = sto3g.ao_labels

    # nelec dict
    nelec_dict = {"H": 1.0, "C": 6.0}

    # target coupled
    targ_coupled = sto3g.focks_orth_tmap_coupled

    # note: we have to remove those samples from the coupled target that include
    # pairs ij whose distance is bigger than the radial cutoff, as we do not have
    # features for those samples
    targ_coupled = drop_target_samples(feats, targ_coupled)

    # ==================================================================
    # Get everything you need inside a single dataset
    # ==================================================================

    dataset = BatchedMemoryDataset(
        len(frames),
        feats,
        frames,
        mo_energy,
        lowdin_charges,
        orbs=orbs,
        nelec_dict=nelec_dict,
        ao_labels=ao_labels,
        batch_size=16,
    )

    # ==================================================================
    # Instantiate model and get the analytical Ridge guess
    # ==================================================================

    model = RidgeOnEnergiesAndLowdinByMO(
        coupled_tmap=targ_coupled,
        features=feats,
        alpha=1e-16,
        dump_dir=os.path.join(
            ROOT_DIR, "HamiltonianLearningEPFL/training/butadiene/00/train_output"
        ),
        bias=False,
    )

    model.fit_ridge_analytical(feats, targ_coupled)

    # save the loss for the analytical guess
    with torch.no_grad():
        full_loss, loss_eigvals, loss_lowdinq, loss_regular = model.loss_fn(
            pred_blocks=model(feats),
            frames=frames,
            eigvals=mo_energy,
            lowdinq=lowdin_charges,
            orbs=orbs,
            ao_labels=ao_labels,
            nelec_dict=nelec_dict,
        )
        np.savez(
            os.path.join(
                ROOT_DIR,
                "HamiltonianLearningEPFL/training/butadiene/00/train_output/ridge_reference_losses.npz",
            ),
            **{
                "loss_full": full_loss.item(),
                "loss_eigvals": loss_eigvals.item(),
                "loss_lowdinq": loss_lowdinq.item(),
                "loss_regular": loss_regular.item(),
            },
        )

    # ==================================================================
    # Training loop
    # ==================================================================

    model.fit(
        dataset,
        epochs=20_000,
        optim_kwargs=dict(lr=1),
        verbose=10,
        dump=50,
    )

    # final dumping
    model.dump_state()
