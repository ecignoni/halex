"""
Used to compare the prediction of ML orbitals (in a small basis)
with the true orbitals (in a big basis).
"""

from ase.io import read
import numpy as np
from pyscf import gto
from pyscf.tools import cubegen

SMALL_BASIS = "sto3g"
BIG_BASIS = "def2svp"
FRAMES = "/local/scratch/cignoni/HamiltonianLearningEPFL/datasets/ethene/coords_ethene_1000.xyz"
INDICES = "indices_higherr.npy"
MO_PRED = "C_pred_higherr.npy"
MO_TRUE = "C_true_higherr.npy"


def get_geom(positions, numbers):
    atom = ""
    for n, p in zip(numbers, positions):
        atom += f"{n:d}   {p[0]:.6f}   {p[1]:.6f}   {p[2]:.6f}\n"
    atom = atom[:-1]
    return atom


def create_mol(positions, numbers, basis):
    mol = gto.M(
        atom=get_geom(positions, numbers),
        basis=basis,
        symmetry=False,
        charge=0,
        spin=0,
        verbose=0,
        cart=False,
    )
    mol.build()
    return mol


def cube_orbitals(
    mol, MO, prefix, imin=None, imax=None, norb=None, orb_lst=None, verbose=False
):
    """
    From Mattia's DMRG_SCF repo
    https://molimen1.dcci.unipi.it/mattia/dmrg_scf/-/blob/lorenzo-mods/utils.py
    """
    # Generate cube files from orbitals
    tot_ele = mol.nelectron
    HOMO = tot_ele // 2 - 1
    LUMO = HOMO + 1

    if orb_lst is None:
        if imin is None:
            if norb is not None:
                imin = HOMO - norb
            else:
                imin = 0

        if imax is None:
            if norb is not None:
                imax = LUMO + norb
            else:
                imax = MO.shape[1]
        orb_lst = range(imin, imax)

    for i in orb_lst:
        if verbose:
            print("Cubing {}".format(i))
        cubegen.orbital(mol, "{}{}.cube".format(prefix, i), MO[:, i], margin=5.0)


if __name__ == "__main__":
    frames = read(FRAMES, ":")
    indices = np.load(INDICES)
    frames = [frames[i] for i in indices]

    mols_small = [create_mol(f.positions, f.numbers, SMALL_BASIS) for f in frames]
    mols_big = [create_mol(f.positions, f.numbers, BIG_BASIS) for f in frames]

    mo_pred = np.load(MO_PRED)
    mo_true = np.load(MO_TRUE)

    [
        cube_orbitals(mol, mo, prefix=f"pred_{i}_", norb=2, verbose=True)
        for mol, mo, i in zip(mols_small, mo_pred, indices)
    ]
    [
        cube_orbitals(mol, mo, prefix=f"true_{i}_", norb=2, verbose=True)
        for mol, mo, i in zip(mols_big, mo_true, indices)
    ]
