import os

import numpy as np
import torch

from equistore import operations as eqop

from halex.utils import tensormap_as_torch, shift_structure_by_n, drop_target_samples
from halex.rotations import ClebschGordanReal
from halex.hamiltonian import (
    compute_ham_features,
    drop_unused_features,
)
from halex.decomposition import EquivariantPCA
from halex.dataset import SCFData, BatchedMemoryDataset
from halex.models import RidgeOnEnergiesAndLowdinMultipleMolecules

torch.set_default_dtype(torch.float64)

if __name__ == "__main__":
    cg = ClebschGordanReal(4)

    # ==================================================================
    # Load data in small basis: we predict a Fock of the same dimension
    # ==================================================================

    # Ethene STO-3G
    ethene_sto3g_dir = "/local/scratch/cignoni/HamiltonianLearningEPFL/datasets/ethene/b3lypg_sto3g/out_spherical/"
    ethene_sto3g = SCFData(
        frames="/local/scratch/cignoni/HamiltonianLearningEPFL/datasets/ethene/coords_ethene_1000.xyz",
        focks=os.path.join(ethene_sto3g_dir, "focks.npy"),
        ovlps=os.path.join(ethene_sto3g_dir, "ovlps.npy"),
        orbs=os.path.join(ethene_sto3g_dir, "orbs.json"),
        cg=cg,
        max_frames=150,
    )

    # Butadiene STO-3G
    butadiene_sto3g_dir = "/local/scratch/cignoni/HamiltonianLearningEPFL/datasets/butadiene/b3lypg_sto3g/out_spherical/"
    butadiene_sto3g = SCFData(
        frames="/local/scratch/cignoni/HamiltonianLearningEPFL/datasets/butadiene/coords_butadiene_1000.xyz",
        focks=os.path.join(butadiene_sto3g_dir, "focks.npy"),
        ovlps=os.path.join(butadiene_sto3g_dir, "ovlps.npy"),
        orbs=os.path.join(butadiene_sto3g_dir, "orbs.json"),
        cg=cg,
        max_frames=150,
    )

    # ==================================================================
    # Load data in big basis: we use its properties as target
    # ==================================================================

    # Ethene def2-SVP
    ethene_def2svp_dir = "/local/scratch/cignoni/HamiltonianLearningEPFL/datasets/ethene/b3lypg_def2svp/out_spherical/"
    ethene_def2svp = SCFData(
        frames="/local/scratch/cignoni/HamiltonianLearningEPFL/datasets/ethene/coords_ethene_1000.xyz",
        focks=os.path.join(ethene_def2svp_dir, "focks.npy"),
        ovlps=os.path.join(ethene_def2svp_dir, "ovlps.npy"),
        orbs=os.path.join(ethene_def2svp_dir, "orbs.json"),
        cg=cg,
        max_frames=150,
    )

    # Butadiene def2-SVP
    butadiene_def2svp_dir = "/local/scratch/cignoni/HamiltonianLearningEPFL/datasets/butadiene/b3lypg_def2svp/out_spherical/"
    butadiene_def2svp = SCFData(
        frames="/local/scratch/cignoni/HamiltonianLearningEPFL/datasets/butadiene/coords_butadiene_1000.xyz",
        focks=os.path.join(butadiene_def2svp_dir, "focks.npy"),
        ovlps=os.path.join(butadiene_def2svp_dir, "ovlps.npy"),
        orbs=os.path.join(butadiene_def2svp_dir, "orbs.json"),
        cg=cg,
        max_frames=150,
    )

    # ==================================================================
    # Compute the Features
    # ==================================================================

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

    ethene_feats = tensormap_as_torch(
        compute_ham_features(rascal_hypers, frames=ethene_sto3g.frames, cg=cg, lcut=2)
    )
    ethene_feats = drop_unused_features(
        ethene_feats, targs_coupled=ethene_sto3g.focks_orth_tmap_coupled
    )

    butadiene_feats = tensormap_as_torch(
        compute_ham_features(
            rascal_hypers, frames=butadiene_sto3g.frames, cg=cg, lcut=2
        )
    )
    butadiene_feats = drop_unused_features(
        butadiene_feats, targs_coupled=butadiene_sto3g.focks_orth_tmap_coupled
    )

    # shift the "structure" of the butadiene features, and join together the features
    # this avoids the creation of the "tensor" dimension in the "samples" axis when joining the tensormaps
    butadiene_feats = shift_structure_by_n(butadiene_feats, n=ethene_sto3g.n_frames)

    feats = eqop.join([ethene_feats, butadiene_feats], axis="samples")

    # ==================================================================
    # Clean the features with PCA
    # ==================================================================

    # equivariant PCA
    epca = EquivariantPCA(n_components=200).fit(feats)
    feats = epca.transform(feats)

    # ==================================================================
    # Concatenate Ethene and Butadiene data
    # ==================================================================

    # frames
    frames = ethene_sto3g.frames + butadiene_sto3g.frames

    # mo_energy
    mo_energy = [
        e for e in ethene_def2svp.mo_energy[:, : ethene_sto3g.mo_energy.shape[-1]]
    ]
    mo_energy += [
        e for e in butadiene_def2svp.mo_energy[:, : butadiene_sto3g.mo_energy.shape[-1]]
    ]

    # lowdin charges
    lowdin_charges = [q for q in ethene_def2svp.lowdin_charges]
    lowdin_charges += [q for q in butadiene_def2svp.lowdin_charges]

    # orbitals per atom species
    assert ethene_sto3g.orbs == butadiene_sto3g.orbs
    orbs = ethene_sto3g.orbs

    # ao_labels
    ao_labels = [ethene_sto3g.ao_labels for _ in range(ethene_sto3g.n_frames)]
    ao_labels += [butadiene_sto3g.ao_labels for _ in range(butadiene_sto3g.n_frames)]

    nelec_dict = {"H": 1.0, "C": 6.0}

    # target coupled
    targ_coupled = eqop.join(
        [
            ethene_sto3g.focks_orth_tmap_coupled,
            shift_structure_by_n(
                butadiene_sto3g.focks_orth_tmap_coupled, n=ethene_sto3g.n_frames
            ),
        ],
        axis="samples",
    )

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
        ao_labels,
        orbs=orbs,
        nelec_dict=nelec_dict,
        batch_size=16,
    )

    # ==================================================================
    # Instantiate model and get the analytical Ridge guess
    # ==================================================================

    model = RidgeOnEnergiesAndLowdinMultipleMolecules(
        coupled_tmap=targ_coupled,
        features=feats,
        alpha=1e-18,
        dump_dir="train_output",
        bias=False,
    )

    model.fit_ridge_analytical(feats, targ_coupled)

    # save the loss for the analytical guess
    with torch.no_grad():
        full_loss, loss_eigvals, loss_lowdinq = model.loss_fn(
            pred_blocks=model(feats),
            frames=frames,
            eigvals=mo_energy,
            lowdinq=lowdin_charges,
            orbs=orbs,
            ao_labels=ao_labels,
            nelec_dict=nelec_dict,
        )
        np.savez(
            "train_output/ridge_reference_losses.npz",
            **{
                "loss_full": full_loss.item(),
                "loss_eigvals": loss_eigvals.item(),
                "loss_lowdinq": loss_lowdinq.item(),
            },
        )

    # ==================================================================
    # Training loop
    # ==================================================================

    model.fit(
        dataset,
        epochs=50_000,
        optim_kwargs=dict(lr=1),
        verbose=10,
        dump=50,
    )

    # final dumping
    model.dump_state()
