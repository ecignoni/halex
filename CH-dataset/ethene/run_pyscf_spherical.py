import numpy as np
from pyscf import gto, dft
from ase.io import read

MOL = 'ethene'
FOLDER = 'b3lyp_STO-3G'
NFRAMES = 1000


def get_geom(positions, numbers):
    atom = ''
    for n, p in zip(numbers, positions):
        atom += f'{n:d}   {p[0]:.6f}   {p[1]:.6f}   {p[2]:.6f}\n'
    atom = atom[:-1]
    return atom


def get_output(mf):
    fock = mf.get_fock()
    ovlp = mf.get_ovlp()
    dm = mf.make_rdm1()
    pop, chg = mf.mulliken_pop()
    ao_labels = mf.mol.ao_labels()
    dip_moment = mf.dip_moment()
    energy_elec = mf.energy_elec()
    energy_tot = mf.energy_tot()
    converged = mf.converged

    out = dict(
        fock=fock,
        ovlp=ovlp,
        dm=dm,
        pop=pop,
        chg=chg,
        ao_labels=ao_labels,
        dip_moment=dip_moment,
        energy_elec=energy_elec,
        energy_tot=energy_tot,
        converged=converged,
    )

    return out


def single_point(positions, numbers, out_idx, dm=None):
    mol = gto.M(
        atom=get_geom(positions, numbers),
        basis='STO-3G',
        symmetry=False,
        charge=0,
        spin=0,
        verbose=5,
        cart=False,
        output=f'/scratch/suman/CH-dataset/{MOL}/{FOLDER}/ethene_out.pylog',
    )
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'b3lypg'
    mf.conv_tol = 1e-12
    # mf.max_cycle = 400
    # mf.conv_tol_grad = 1e-10
    # mf.diis_space = 12

    grids = dft.Grids(mol)
    grids.atom_grid = (99, 590)
    grids.build()
    mf.grids = grids

    if dm is None:
        mf.kernel()
    else:
        mf.kernel(dm)

    print('Converged:', mf.converged)

    out = get_output(mf)

    np.savez(f'/scratch/suman/CH-dataset/{MOL}/{FOLDER}/out_spherical/{out_idx:05d}.npz', **out)

    # return mf.make_rdm1()


if __name__ == '__main__':

    structures = read(f'/scratch/suman/CH-dataset/{MOL}/coords_{MOL}_{NFRAMES}.xyz', ':')

    for idx, structure in enumerate(structures):
        single_point(structure.positions, structure.numbers, idx, dm=None)
