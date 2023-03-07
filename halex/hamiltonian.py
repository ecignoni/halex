from .builder import TensorBuilder
from .rotations import ClebschGordanReal
from .rascal_wrapper import RascalSphericalExpansion, RascalPairExpansion
from .acdc_mini import acdc_standardize_keys, cg_increment

import numpy as np
from equistore import Labels, TensorBlock, TensorMap
from equistore.io import save as equisave
import torch


def _components_idx(l):  # noqa
    """just a mini-utility function to get the m=-l..l indices"""
    return np.arange(-l, l + 1, dtype=np.int32).reshape(2 * l + 1, 1)


def _components_idx_2d(li, lj):
    """indexing the entries in a 2d (l_i, l_j) block of the hamiltonian
    in the uncoupled basis"""
    return np.array(
        np.meshgrid(_components_idx(li), _components_idx(lj)), dtype=np.int32
    ).T.reshape(-1, 2)


def _orbs_offsets(orbs):
    """offsets for the orbital subblocks within an atom block of the Hamiltonian matrix"""
    orbs_tot = {}
    orbs_offset = {}
    for k in orbs:
        ko = 0
        for n, l, m in orbs[k]:
            if m != -l:
                continue
            orbs_offset[(k, n, l)] = ko
            ko += 2 * l + 1
        orbs_tot[k] = ko
    return orbs_tot, orbs_offset


def _atom_blocks_idx(frames, orbs_tot):
    """position of the hamiltonian subblocks for each atom in each frame"""
    atom_blocks_idx = {}
    for A, f in enumerate(frames):
        ki = 0
        for i, ai in enumerate(f.numbers):
            kj = 0
            for j, aj in enumerate(f.numbers):
                atom_blocks_idx[(A, i, j)] = (ki, kj)
                kj += orbs_tot[aj]
            ki += orbs_tot[ai]
    return atom_blocks_idx


def dense_to_blocks(dense, frames, orbs):  # noqa: C901
    """
    Converts a list of dense matrices `dense` corresponding to the single-particle Hamiltonians for the structures
    described by `frames`, and using the orbitals described in the dictionary `orbs` into a TensorMap storage format.

    The label convention is as follows:

    The keys that label the blocks are ["block_type", "a_i", "n_i", "l_i", "a_j", "n_j", "l_j"].
    block_type: 0 -> diagonal blocks, atom i=j
                2 -> different species block, stores only when n_i,l_i and n_j,l_j are lexicographically sorted
                1,-1 -> same specie, off-diagonal. store separately symmetric (1) and anti-symmetric (-1) term
    a_{i,j}: chemical species (atomic number) of the two atoms
    n_{i,j}: radial channel
    l_{i,j}: angular momentum
    """

    block_builder = TensorBuilder(
        ["block_type", "a_i", "n_i", "l_i", "a_j", "n_j", "l_j"],
        ["structure", "center", "neighbor"],
        [["m1"], ["m2"]],
        ["value"],
    )
    orbs_tot, _ = _orbs_offsets(orbs)
    for A in range(len(frames)):
        frame = frames[A]
        ham = dense[A]
        ki_base = 0
        for i, ai in enumerate(frame.numbers):
            kj_base = 0
            for j, aj in enumerate(frame.numbers):
                if i == j:
                    block_type = 0  # diagonal
                elif ai == aj:
                    if i > j:
                        kj_base += orbs_tot[aj]
                        continue
                    block_type = 1  # same-species
                else:
                    if ai > aj:  # only sorted element types
                        kj_base += orbs_tot[aj]
                        continue
                    block_type = 2  # different species
                if isinstance(ham, np.ndarray):
                    block_data = torch.from_numpy(
                        ham[
                            ki_base : ki_base + orbs_tot[ai],  # noqa: E203
                            kj_base : kj_base + orbs_tot[aj],  # noqa: E203
                        ]
                    )
                elif isinstance(ham, torch.Tensor):
                    block_data = ham[
                        ki_base : ki_base + orbs_tot[ai],  # noqa: E203
                        kj_base : kj_base + orbs_tot[aj],  # noqa: E203
                    ]
                else:
                    raise ValueError

                # print(block_data, block_data.shape)
                if block_type == 1:
                    block_data_plus = (block_data + block_data.T) * 1 / np.sqrt(2)
                    block_data_minus = (block_data - block_data.T) * 1 / np.sqrt(2)
                ki_offset = 0
                for ni, li, mi in orbs[ai]:
                    if (
                        mi != -li
                    ):  # picks the beginning of each (n,l) block and skips the other orbitals
                        continue
                    kj_offset = 0
                    for nj, lj, mj in orbs[aj]:
                        if (
                            mj != -lj
                        ):  # picks the beginning of each (n,l) block and skips the other orbitals
                            continue
                        if ai == aj and (ni > nj or (ni == nj and li > lj)):
                            kj_offset += 2 * lj + 1
                            continue
                        block_idx = (block_type, ai, ni, li, aj, nj, lj)
                        if block_idx not in block_builder.blocks:
                            block = block_builder.add_block(
                                keys=block_idx,
                                properties=np.asarray([[0]], dtype=np.int32),
                                components=[_components_idx(li), _components_idx(lj)],
                            )

                            if block_type == 1:
                                block_asym = block_builder.add_block(
                                    keys=(-1,) + block_idx[1:],
                                    properties=np.asarray([[0]], dtype=np.int32),
                                    components=[
                                        _components_idx(li),
                                        _components_idx(lj),
                                    ],
                                )
                        else:
                            block = block_builder.blocks[block_idx]
                            if block_type == 1:
                                block_asym = block_builder.blocks[(-1,) + block_idx[1:]]

                        islice = slice(ki_offset, ki_offset + 2 * li + 1)
                        jslice = slice(kj_offset, kj_offset + 2 * lj + 1)

                        if block_type == 1:
                            block.add_samples(
                                labels=[(A, i, j)],
                                data=block_data_plus[islice, jslice].reshape(
                                    (1, 2 * li + 1, 2 * lj + 1, 1)
                                ),
                            )
                            block_asym.add_samples(
                                labels=[(A, i, j)],
                                data=block_data_minus[islice, jslice].reshape(
                                    (1, 2 * li + 1, 2 * lj + 1, 1)
                                ),
                            )

                        else:
                            block.add_samples(
                                labels=[(A, i, j)],
                                data=block_data[islice, jslice].reshape(
                                    (1, 2 * li + 1, 2 * lj + 1, 1)
                                ),
                            )

                        kj_offset += 2 * lj + 1
                    ki_offset += 2 * li + 1
                kj_base += orbs_tot[aj]

            ki_base += orbs_tot[ai]
    # print(block_builder.build())
    return block_builder.build()


def blocks_to_dense(blocks, frames, orbs):  # noqa: C901
    """
    Converts a TensorMap containing matrix blocks in the uncoupled basis, `blocks` into dense matrices.
    Needs `frames` and `orbs` to reconstruct matrices in the correct order. See `dense_to_blocks` to understant
    the different types of blocks.
    """

    orbs_tot, orbs_offset = _orbs_offsets(orbs)
    atom_blocks_idx = _atom_blocks_idx(frames, orbs_tot)

    # init storage for the dense hamiltonians
    # ensure they live on GPU if tensormap values live on GPU
    device = blocks.block(0).values.device
    dense = []
    for f in frames:
        norbs = 0
        for ai in f.numbers:
            norbs += orbs_tot[ai]
        ham = torch.zeros(norbs, norbs, device=device)  # , dtype=np.float64)
        dense.append(ham)

    assign_fns = {
        0: _assign_same_atom,
        2: _assign_different_species,
        1: _assign_same_species_symm,
        -1: _assign_same_species_antisymm,
    }

    # loops over block types
    for idx, block in blocks:
        # I can't loop over the frames directly, so I'll keep track
        # of the frame with these two variables
        dense_idx = -1
        cur_A = -1

        block_type, ai, ni, li, aj, nj, lj = tuple(idx)

        # choose the right function (that assigns block values to fock blocks)
        assign_fn = assign_fns[block_type]

        # offset of the orbital block within the pair block in the matrix
        ki_offset = orbs_offset[(ai, ni, li)]
        kj_offset = orbs_offset[(aj, nj, lj)]
        same_koff = ki_offset == kj_offset

        # loops over samples (structure, i, j)
        for (A, i, j), block_data in zip(block.samples, block.values):
            # check if we have to update the frame and index
            if A != cur_A:
                cur_A = A
                dense_idx += 1

            ham = dense[dense_idx]

            # coordinates of the atom block in the matrix
            ki_base, kj_base = atom_blocks_idx[(dense_idx, i, j)]
            # values to assign
            values = block_data[:, :, 0].reshape(2 * li + 1, 2 * lj + 1)
            # assign values
            assign_fn(
                ham, values, ki_base, kj_base, ki_offset, kj_offset, same_koff, li, lj
            )

    return dense


def _assign_same_atom(
    ham, values, ki_base, kj_base, ki_offset, kj_offset, same_koff, li, lj
):
    islice = slice(ki_base + ki_offset, ki_base + ki_offset + 2 * li + 1)
    jslice = slice(kj_base + kj_offset, kj_base + kj_offset + 2 * lj + 1)
    ham[islice, jslice] = values
    if not same_koff:
        ham[jslice, islice] = values.T


def _assign_different_species(
    ham, values, ki_base, kj_base, ki_offset, kj_offset, same_koff, li, lj
):
    islice = slice(ki_base + ki_offset, ki_base + ki_offset + 2 * li + 1)
    jslice = slice(kj_base + kj_offset, kj_base + kj_offset + 2 * lj + 1)
    ham[islice, jslice] = values
    ham[jslice, islice] = values.T


def _assign_same_species_symm(
    ham, values, ki_base, kj_base, ki_offset, kj_offset, same_koff, li, lj
):
    islice = slice(ki_base + ki_offset, ki_base + ki_offset + 2 * li + 1)
    jslice = slice(kj_base + kj_offset, kj_base + kj_offset + 2 * lj + 1)
    values_2norm = values / (2 ** (0.5))
    ham[islice, jslice] += values_2norm
    ham[jslice, islice] += values_2norm.T
    if not same_koff:
        islice = slice(ki_base + kj_offset, ki_base + kj_offset + 2 * lj + 1)
        jslice = slice(kj_base + ki_offset, kj_base + ki_offset + 2 * li + 1)
        ham[islice, jslice] += values_2norm.T
        ham[jslice, islice] += values_2norm


def _assign_same_species_antisymm(
    ham, values, ki_base, kj_base, ki_offset, kj_offset, same_koff, li, lj
):
    islice = slice(ki_base + ki_offset, ki_base + ki_offset + 2 * li + 1)
    jslice = slice(kj_base + kj_offset, kj_base + kj_offset + 2 * lj + 1)
    values_2norm = values / (2 ** (0.5))
    ham[islice, jslice] += values_2norm
    ham[jslice, islice] += values_2norm.T
    if not same_koff:
        islice = slice(ki_base + kj_offset, ki_base + kj_offset + 2 * lj + 1)
        jslice = slice(kj_base + ki_offset, kj_base + ki_offset + 2 * li + 1)
        ham[islice, jslice] -= values_2norm.T
        ham[jslice, islice] -= values_2norm


def couple_blocks(blocks, cg=None):
    if cg is None:
        lmax = max(blocks.keys["li"] + blocks.keys["lj"])
        cg = ClebschGordanReal(lmax)

    block_builder = TensorBuilder(
        ["block_type", "a_i", "n_i", "l_i", "a_j", "n_j", "l_j", "L"],
        ["structure", "center", "neighbor"],
        [["M"]],
        ["value"],
    )
    for idx, block in blocks:
        block_type, ai, ni, li, aj, nj, lj = tuple(idx)
        decoupled = torch.moveaxis(block.values, -1, -2).reshape(
            (len(block.samples), len(block.properties), 2 * li + 1, 2 * lj + 1)
        )
        coupled = cg.couple(decoupled)[(li, lj)]
        for L in coupled:
            block_idx = tuple(idx) + (L,)
            # skip blocks that are zero because of symmetry
            if ai == aj and ni == nj and li == lj:
                parity = (-1) ** (li + lj + L)
                if (parity == -1 and block_type in (0, 1)) or (
                    parity == 1 and block_type == -1
                ):
                    continue
            new_block = block_builder.add_block(
                keys=block_idx,
                properties=np.asarray([[0]], dtype=np.int32),
                components=[_components_idx(L).reshape(-1, 1)],
            )
            new_block.add_samples(
                labels=block.samples.view(dtype=np.int32).reshape(
                    block.samples.shape[0], -1
                ),
                data=torch.moveaxis(coupled[L], -1, -2),
            )

    return block_builder.build()


def decouple_blocks(blocks, cg=None):
    if cg is None:
        lmax = max(blocks.keys["L"])
        cg = ClebschGordanReal(lmax)

    block_builder = TensorBuilder(
        ["block_type", "a_i", "n_i", "l_i", "a_j", "n_j", "l_j"],
        ["structure", "center", "neighbor"],
        [["m1"], ["m2"]],
        ["value"],
    )
    for idx, block in blocks:
        block_type, ai, ni, li, aj, nj, lj, L = tuple(idx)
        block_idx = (block_type, ai, ni, li, aj, nj, lj)
        if block_idx in block_builder.blocks:
            continue
        coupled = {}
        for L in range(np.abs(li - lj), li + lj + 1):
            bidx = blocks.keys.position(block_idx + (L,))
            if bidx is not None:
                coupled[L] = torch.moveaxis(blocks.block(bidx).values, -1, -2)
        decoupled = cg.decouple({(li, lj): coupled})

        new_block = block_builder.add_block(
            keys=block_idx,
            properties=np.asarray([[0]], dtype=np.int32),
            components=[_components_idx(li), _components_idx(lj)],
        )
        new_block.add_samples(
            labels=block.samples.view(dtype=np.int32).reshape(
                block.samples.shape[0], -1
            ),
            data=torch.moveaxis(decoupled, 1, -1),
        )
    return block_builder.build()


def hamiltonian_features(centers, pairs):
    """Builds Hermitian, HAM-learning adapted features starting
    from generic center |rho_i^nu> and pair |rho_ij^nu> features.
    The sample and property labels must match."""
    keys = []
    blocks = []
    # central blocks
    for k, b in centers:
        keys.append(
            tuple(k)
            + (  # noqa: W503
                k["species_center"],
                0,
            )
        )
        samples_array = np.vstack(b.samples.tolist())
        blocks.append(
            TensorBlock(
                samples=Labels(
                    names=b.samples.names + ("neighbor",),
                    values=np.asarray(
                        np.hstack([samples_array, samples_array[:, -1:]]),
                        dtype=np.int32,
                    ),
                ),
                components=b.components,
                properties=b.properties,
                values=b.values,
            )
        )

    for k, b in pairs:
        if k["species_center"] == k["species_neighbor"]:
            # off-site, same species
            idx_up = np.where(b.samples["center"] < b.samples["neighbor"])[0]
            if len(idx_up) == 0:
                continue
            idx_lo = np.where(b.samples["center"] > b.samples["neighbor"])[0]

            # we need to find the "ji" position that matches each "ij" sample.
            # we exploit the fact that the samples are sorted by structure to do a "local" rearrangement
            smp_up, smp_lo = 0, 0
            for smp_up in range(len(idx_up)):
                ij = b.samples[idx_up[smp_up]][["center", "neighbor"]]
                for smp_lo in range(smp_up, len(idx_lo)):
                    ij_lo = b.samples[idx_lo[smp_lo]][["neighbor", "center"]]
                    if (
                        b.samples[idx_up[smp_up]]["structure"]
                        != b.samples[idx_lo[smp_lo]]["structure"]  # noqa: W503
                    ):
                        raise ValueError(
                            f"Could not find matching ji term for sample {b.samples[idx_up[smp_up]]}"
                        )
                    if tuple(ij) == tuple(ij_lo):
                        idx_lo[smp_up], idx_lo[smp_lo] = idx_lo[smp_lo], idx_lo[smp_up]
                        break

            keys.append(tuple(k) + (1,))
            keys.append(tuple(k) + (-1,))
            blocks.append(
                TensorBlock(
                    samples=Labels(
                        names=b.samples.names,
                        values=np.asarray(b.samples[idx_up].tolist(), dtype=np.int32),
                    ),
                    components=b.components,
                    properties=b.properties,
                    values=(b.values[idx_up] + b.values[idx_lo]) / np.sqrt(2),
                )
            )
            blocks.append(
                TensorBlock(
                    samples=Labels(
                        names=b.samples.names,
                        values=np.asarray(b.samples[idx_up].tolist(), dtype=np.int32),
                    ),
                    components=b.components,
                    properties=b.properties,
                    values=(b.values[idx_up] - b.values[idx_lo]) / np.sqrt(2),
                )
            )
        elif k["species_center"] < k["species_neighbor"]:
            # off-site, different species
            keys.append(tuple(k) + (2,))
            blocks.append(
                TensorBlock(
                    samples=b.samples,
                    components=b.components,
                    properties=b.properties,
                    values=b.values.copy(),
                )
            )

    return TensorMap(
        keys=Labels(
            names=pairs.keys.names + ("block_type",),
            values=np.asarray(keys, dtype=np.int32),
        ),
        blocks=blocks,
    )


def compute_ham_features(rascal_hypers, frames, cg, lcut, saveto=None, verbose=False):
    if verbose:
        print("Computing |rho_i>")
    spex = RascalSphericalExpansion(rascal_hypers)
    rhoi = spex.compute(frames)

    if verbose:
        print("Computing |g_ij>")
    pairs = RascalPairExpansion(rascal_hypers)
    gij = pairs.compute(frames)

    # make them compatible for cg increments...
    rho1i = acdc_standardize_keys(rhoi)
    rho1i = rho1i.keys_to_properties(["species_neighbor"])
    gij = acdc_standardize_keys(gij)

    if verbose:
        print("Computing |rho^2_i>")
    rho2i = cg_increment(
        rho1i, rho1i, lcut=lcut, other_keys_match=["species_center"], clebsch_gordan=cg
    )

    if verbose:
        print("Computing |rho^1_ij>")
    rho1ij = cg_increment(
        rho1i, gij, lcut=lcut, other_keys_match=["species_center"], clebsch_gordan=cg
    )

    if verbose:
        print("Getting Hamiltonian features")
    ham_feats = hamiltonian_features(rho2i, rho1ij)

    if saveto is not None:
        if verbose:
            print(f"Saving to {saveto}")
        equisave(saveto, ham_feats)

    return ham_feats


def drop_unused_features(feats, targs_coupled):
    retained_keys = []
    retained_blocks = []

    for targ_key, _ in targs_coupled:
        block_type = targ_key["block_type"]
        ai = targ_key["a_i"]
        li = targ_key["l_i"]
        aj = targ_key["a_j"]
        lj = targ_key["l_j"]
        L = targ_key["L"]
        inversion_sigma = (-1) ** (li + lj + L)

        for feat_key, feat_block in feats:
            cond_block = feat_key["block_type"] == block_type
            cond_l = feat_key["spherical_harmonics_l"] == L
            cond_sigma = feat_key["inversion_sigma"] == inversion_sigma
            cond_cent = feat_key["species_center"] == ai
            cond_neig = feat_key["species_neighbor"] == aj
            cond = cond_block and cond_l and cond_sigma and cond_cent and cond_neig

            if cond:
                retained_keys.append(np.array(tuple(feat_key)))
                retained_blocks.append(feat_block.copy())

    # remove duplicates
    _, idx = np.unique(retained_keys, axis=0, return_index=True)
    retained_keys = np.array([retained_keys[i] for i in idx])
    retained_blocks = [retained_blocks[i] for i in idx]

    retained_keys = Labels(names=feats.keys.names, values=np.array(retained_keys))
    trim_feats = TensorMap(keys=retained_keys, blocks=retained_blocks)

    return trim_feats


# if __name__ == "__main__":
#     from utils.rotations import rotation_matrix, wigner_d_real
#     from tqdm import tqdm
#     from torch_utils import load_frames, load_orbs, load_hamiltonians, compute_ham_features
#
#     def test_rotation_equivariance_features():
#         '''
#         Equivariance test: f(ŜA) = Ŝ(f(A))
#         Here the operation is a rotation in 3D space: Ŝ = R
#         '''
#         # rotation angles, ZYZ
#         alpha = np.pi / 3
#         beta = np.pi / 3
#         gamma = np.pi / 4
#
#         n_frames = 1
#         frames = load_frames(
#             "../data/hamiltonian/water-hamiltonian/water_coords_1000.xyz",
#             n_frames=n_frames,
#         )
#         orbs = load_orbs("../data/hamiltonian/water-hamiltonian/water_orbs.json")
#         hams = load_hamiltonians(
#             "../data/hamiltonian/water-hamiltonian/water_saph_orthogonal.npy",
#             n_frames=n_frames,
#         )
#
#         R = rotation_matrix(alpha, beta, gamma).T
#
#         cg = ClebschGordanReal(4)
#
#         rascal_hypers = {
#             "interaction_cutoff": 3.5,
#             "cutoff_smooth_width": 0.5,
#             "max_radial": 1,
#             "max_angular": 3,
#             "gaussian_sigma_constant": 0.2,
#             "gaussian_sigma_type": "Constant",
#             "compute_gradients": False,
#         }
#
#         for frame in frames:
#             A = frame.copy()
#             RA = frame.copy()
#             RA.positions = RA.positions @ R
#             RA.cell = RA.cell @ R
#
#             f_A = compute_ham_features(rascal_hypers, [A], cg)
#             f_RA = compute_ham_features(rascal_hypers, [RA], cg)
#
#             for (key, block), (_, rotated_block) in zip(f_A, f_RA):
#                 _, _, l, _, _, _ = key
#                 D = wigner_d_real(int(l), alpha, beta, gamma)
#                 rotated_values = np.einsum("nm,smp->snp", D, block.values)
#                 diff = np.sum(abs(rotated_values) - abs(rotated_block.values))
#
#                 assert (
#                     diff < 1e-17
#                 ), f"mismatch for key={key}, sum of abs. differences: {diff}"
#
#     test_rotation_equivariance_features()
