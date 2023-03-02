import os

import numpy as np
import torch

from halex.utils import tensormap_as_torch
from halex.rotations import ClebschGordanReal
from halex.hamiltonian import (
    compute_ham_features,
    drop_unused_features,
)
from halex.model_selection import train_test_split
from halex.decomposition import EquivariantPCA
from halex.dataset import SCFData, BatchedMemoryDataset
from halex.models import RidgeOnEnergiesAndLowdin

torch.set_default_dtype(torch.float64)


if __name__ == "__main__":
    n_frames = 200

    cg = ClebschGordanReal(4)

    # small fock matrices, in a basis equivalent to what we want to learn
    # this is used to define the model and to provide a starting guess
    sto3g_dir = "/local/scratch/cignoni/HamiltonianLearningEPFL/datasets/ethene/b3lypg_sto3g/out_spherical/"
    sto3g_basis = SCFData(
        frames="/local/scratch/cignoni/HamiltonianLearningEPFL/datasets/ethene/coords_ethene_1000.xyz",
        focks=os.path.join(sto3g_dir, "focks.npy"),
        ovlps=os.path.join(sto3g_dir, "ovlps.npy"),
        orbs=os.path.join(sto3g_dir, "orbs.json"),
        cg=cg,
        max_frames=n_frames,
    )

    # fock matrices in bigger basis, we use lowdin charges and (selected) MO energies
    # of this basis as target
    def2svp_dir = "/local/scratch/cignoni/HamiltonianLearningEPFL/datasets/ethene/b3lypg_def2svp/out_spherical/"
    def2svp_basis = SCFData(
        frames="/local/scratch/cignoni/HamiltonianLearningEPFL/datasets/ethene/coords_ethene_1000.xyz",
        focks=os.path.join(def2svp_dir, "focks.npy"),
        ovlps=os.path.join(def2svp_dir, "ovlps.npy"),
        orbs=os.path.join(def2svp_dir, "orbs.json"),
        cg=cg,
        max_frames=n_frames,
    )

    # compute features
    rascal_hypers = {
        "interaction_cutoff": 3.5,
        "cutoff_smooth_width": 0.5,
        "max_radial": 6,
        "max_angular": 4,
        "gaussian_sigma_constant": 0.2,
        "gaussian_sigma_type": "Constant",
        "compute_gradients": False,
    }

    feats = compute_ham_features(rascal_hypers, sto3g_basis.frames, cg, lcut=2)
    feats = tensormap_as_torch(feats)
    # we retain only the features necessary to learn a STO-3G hamiltonian
    feats = drop_unused_features(feats, sto3g_basis.focks_orth_tmap_coupled)

    # train test split
    (
        sto3g_train_focks_coup,
        _,
        sto3g_train_focks,
        _,
        sto3g_train_eigvals,
        _,
        def2svp_train_focks,
        _,
        def2svp_train_eigvals,
        _,
        def2svp_train_lowdinq,
        _,
        train_feats,
        _,
        train_frames,
        _,
    ) = train_test_split(
        sto3g_basis.focks_orth_tmap_coupled,
        sto3g_basis.focks_orth,
        sto3g_basis.mo_energy,
        def2svp_basis.focks_orth,
        def2svp_basis.mo_energy,
        def2svp_basis.lowdin_charges,
        feats,
        sto3g_basis.frames,
        n_frames=sto3g_basis.max_frames,
        train_size=0.8,
    )

    # reducing the number of features with PCA
    epca = EquivariantPCA(n_components=160).fit(train_feats)
    train_feats = epca.transform(train_feats)

    # eigenvalues on big basis, select the first n where n is the dimension of the small basis
    nmo_sto3g = sto3g_basis.focks_orth[0].shape[0]
    def2svp_train_eigvals = def2svp_train_eigvals[:, :nmo_sto3g]

    train_dataset = BatchedMemoryDataset(
        len(train_frames),
        train_feats,
        train_frames,
        def2svp_train_eigvals,
        def2svp_basis.lowdin_charges,
        orbs=sto3g_basis.orbs,
        ao_labels=sto3g_basis.ao_labels,
        nelec_dict=sto3g_basis.nelec_dict,
        batch_size=16,
    )

    # not used anymore, try to save some space
    del def2svp_basis, sto3g_basis

    # model instantiated with small basis, so that it predicts a fock in a small basis
    model = RidgeOnEnergiesAndLowdin(
        coupled_tmap=sto3g_train_focks_coup,
        features=train_feats,
        alpha=1e-18,
        dump_dir="train_output",
        bias=False,
    )

    # get the train metrics on eigenvalues and eigenvectors for a linear ridge
    # solution (initial guess)
    model.fit_ridge_analytical(train_feats, sto3g_train_focks_coup)

    with torch.no_grad():
        full_loss, loss_eigvals, loss_lowdinq = model.loss_fn(
            pred_blocks=model(train_feats),
            frames=train_frames,
            eigvals=def2svp_train_eigvals,
            lowdinq=def2svp_train_lowdinq,
            orbs=train_dataset.orbs,
            ao_labels=train_dataset.ao_labels,
            nelec_dict=train_dataset.nelec_dict,
        )
        np.savez(
            "train_output/ridge_reference_losses.npz",
            **{
                "loss_full": full_loss.item(),
                "loss_eigvals": loss_eigvals.item(),
                "loss_lowdinq": loss_lowdinq.item(),
            },
        )

    # start training
    model.fit(
        train_dataset,
        epochs=50_000,
        optim_kwargs=dict(lr=1),
        verbose=10,
        dump=50,
    )

    # final dumping
    model.dump_state()
