from __future__ import annotations

import gc
import os

import metatensor
import numpy as np
import torch

from halex.decomposition import EquivariantPCA
from halex.models import RidgeOnEnergiesAndLowdinMultipleMolecules  # ByMO
from halex.rotations import ClebschGordanReal
from halex.train_utils import (
    compute_features,
    coupled_fock_matrix_from_multiple_molecules,
    load_batched_dataset,
    load_molecule_scf_datasets,
)
from halex.utils import drop_target_samples

torch.set_default_dtype(torch.float64)

ROOT_DIR = "/Users/divya/scratch/"


def load_molecule_datasets(mol: str, cg: ClebschGordanReal, indices: np.ndarray):
    """
    Load the SCFData objects storing data for a single molecule,
    in both a small basis and a big basis
    """
    coords_path = os.path.join(ROOT_DIR, f"CH-dataset/{mol}/coords_{mol}_1000.xyz")
    small_basis_path = os.path.join(ROOT_DIR, f"CH-dataset/{mol}/b3lyp_STO-3G/")
    big_basis_path = os.path.join(
        ROOT_DIR,
        f"CH-dataset/{mol}/b3lyp_def2tzvp/",
    )

    sb_data, bb_data = load_molecule_scf_datasets(
        coords_path=coords_path,
        small_basis_path=small_basis_path,
        big_basis_path=big_basis_path,
        cg=cg,
        train_indices=indices,
    )

    return sb_data, bb_data


if __name__ == "__main__":
    cg = ClebschGordanReal(4)

    molecules = [
        "ethane",
        "ethene",
        # "butadiene",
        # "hexane",
        # "hexatriene",
        # "isoprene",
        # "styrene",
    ]

    indices = np.arange(1000)
    np.random.shuffle(indices)
    valid_indices = indices[-200:]
    indices = indices[:500]
    np.save("train_output/train_indices.npy", indices)
    np.save("train_output/valid_indices.npy", valid_indices)

    # Get datasets in small and big basis
    datasets = {
        mol: load_molecule_datasets(mol, cg=cg, indices=indices) for mol in molecules
    }

    valid_datasets = {
        mol: load_molecule_datasets(mol, cg=cg, indices=valid_indices)
        for mol in molecules
    }

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

    feats = compute_features(datasets, rascal_hypers=rascal_hypers, cg=cg, lcut=2)
    gc.collect()

    epca = EquivariantPCA(n_components=200).fit(metatensor.join(feats, axis="samples"))

    feats = [epca.transform(feats_) for feats_ in feats]
    gc.collect()

    epca.save("train_output/epca.npz")

    valid_feats = compute_features(
        valid_datasets, rascal_hypers=rascal_hypers, cg=cg, lcut=2, epca=epca
    )
    gc.collect()

    # Batched Datasets
    nelec_dict = {"H": 1.0, "C": 6.0}

    multimol_datasets = [
        load_batched_dataset(
            scf_datasets=data,
            feats=feat,
            nelec_dict=nelec_dict,
            batch_size=100,
            lowdin_charges_by_MO=False,
            # lowdin_mo_indices=indices,
        )
        for data, feat in zip(datasets.values(), feats)
    ]

    valid_multimol_datasets = [
        load_batched_dataset(
            scf_datasets=data,
            feats=feat,
            nelec_dict=nelec_dict,
            batch_size=50,
            lowdin_charges_by_MO=False,
            # lowdin_mo_indices=indices,
        )
        for data, feat in zip(valid_datasets.values(), valid_feats)
    ]

    # ==================================================================
    # Convert the Fock matrix to coupled basis and drop samples
    # that are not present in the features
    # ==================================================================

    targ_coupled = coupled_fock_matrix_from_multiple_molecules(datasets.values())
    targ_coupled = drop_target_samples(
        metatensor.join(feats, axis="samples"), targ_coupled, verbose=True
    )

    # ==================================================================
    # Instantiate model and get the analytical Ridge guess
    # ==================================================================

    model = RidgeOnEnergiesAndLowdinMultipleMolecules(
        coupled_tmap=targ_coupled,
        features=metatensor.join(feats, axis="samples"),
        alpha=1e-14,
        dump_dir="train_output",
        bias=False,
    )

    model.fit_ridge_analytical(
        features=metatensor.join(feats, axis="samples"),
        targets=targ_coupled,
    )

    # ==================================================================
    # Training loop
    # ==================================================================

    model.fit(
        train_datasets=multimol_datasets,
        valid_datasets=valid_multimol_datasets,
        epochs=20_000,
        optim_kwargs=dict(lr=1),
        verbose=10,
        dump=50,
    )

    model.dump_state()
