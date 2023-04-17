from __future__ import annotations
from typing import List, Dict, Tuple, Any, Union

from ..builder import TensorBuilder

import numpy as np
import torch

from equistore import Labels, TensorMap, TensorBlock

Atoms = Any


def _components_idx(l):  # noqa
    """just a mini-utility function to get the m=-l..l indices"""
    return np.arange(-l, l + 1, dtype=np.int32).reshape(2 * l + 1, 1)


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


# ============================================================================
# TensorMap -> Dense -> TensorMap transformations
# ============================================================================


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


def blocks_to_dense(
    blocks: TensorMap,
    frames: List[Atoms],
    orbs: Dict[int, List[Tuple[int, int, int]]],
    vectorized: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    if vectorized:
        return _vectorized_blocks_to_dense(blocks=blocks, frames=frames, orbs=orbs)
    else:
        return _blocks_to_dense(blocks=blocks, frames=frames, orbs=orbs)


def _blocks_to_dense(
    blocks: TensorMap, frames: List[Atoms], orbs: Dict[int, List[Tuple[int, int, int]]]
) -> Union[np.ndarray, torch.Tensor]:
    """from tensormap to dense representation

    Converts a TensorMap containing matrix blocks in the uncoupled basis,
    `blocks` into dense matrices.
    Needs `frames` and `orbs` to reconstruct matrices in the correct order.
    See `dense_to_blocks` to understant the different types of blocks.
    """

    # total number of orbitals per atom, orbital offset per atom
    orbs_tot, orbs_offset = _orbs_offsets(orbs)

    # indices of the block for each atom
    atom_blocks_idx = _atom_blocks_idx(frames, orbs_tot)

    # init storage for the dense hamiltonians
    # ensure they live on GPU if tensormap values live on GPU
    device = blocks.block(0).values.device
    dense = []
    for f in frames:
        norbs = 0
        for ai in f.numbers:
            norbs += orbs_tot[ai]
        ham = torch.zeros(norbs, norbs, device=device)
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
    ham: Union[np.ndarray, torch.Tensor],
    values: Union[np.ndarray, torch.Tensor],
    ki_base: int,
    kj_base: int,
    ki_offset: int,
    kj_offset: int,
    same_koff: bool,
    li: int,
    lj: int,
) -> None:
    islice = slice(ki_base + ki_offset, ki_base + ki_offset + 2 * li + 1)
    jslice = slice(kj_base + kj_offset, kj_base + kj_offset + 2 * lj + 1)
    ham[islice, jslice] = values
    if not same_koff:
        ham[jslice, islice] = values.T


def _assign_different_species(
    ham: Union[np.ndarray, torch.Tensor],
    values: Union[np.ndarray, torch.Tensor],
    ki_base: int,
    kj_base: int,
    ki_offset: int,
    kj_offset: int,
    same_koff: bool,
    li: int,
    lj: int,
) -> None:
    islice = slice(ki_base + ki_offset, ki_base + ki_offset + 2 * li + 1)
    jslice = slice(kj_base + kj_offset, kj_base + kj_offset + 2 * lj + 1)
    ham[islice, jslice] = values
    ham[jslice, islice] = values.T


def _assign_same_species_symm(
    ham: Union[np.ndarray, torch.Tensor],
    values: Union[np.ndarray, torch.Tensor],
    ki_base: int,
    kj_base: int,
    ki_offset: int,
    kj_offset: int,
    same_koff: bool,
    li: int,
    lj: int,
) -> None:
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
    ham: Union[np.ndarray, torch.Tensor],
    values: Union[np.ndarray, torch.Tensor],
    ki_base: int,
    kj_base: int,
    ki_offset: int,
    kj_offset: int,
    same_koff: bool,
    li: int,
    lj: int,
) -> None:
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


def _vectorized_blocks_to_dense(
    blocks: TensorMap, frames: List[Atoms], orbs: Dict[int, List[List[int, int, int]]]
) -> torch.Tensor:
    # total number of orbitals per atom, orbital offset per atom
    orbs_tot, orbs_offset = _orbs_offsets(orbs)

    # indices of the block for each atom
    atom_blocks_idx = _atom_blocks_idx(frames, orbs_tot)

    # init storage for the dense hamiltonians
    # note: this vectorized function relies on storing `dense`
    #     as a torch tensor, i.e., all frames correspond to a
    #     single molecule. we can then use the first frame to
    #     set the size of the dense hamiltonians
    norbs = sum(orbs_tot[ai] for ai in frames[0].numbers)
    device = blocks.block(0).values.device
    dense = torch.zeros((len(frames), norbs, norbs), device=device)

    # instantiating the slices of each block in order to assign
    # the values to the dense matrices is the slow part of this function
    # here we use a simple caching mechanism to avoid computing the same
    # slice if we have already computed it previously
    SLICES_CACHE = {}

    # loops over block types
    for idx, block in blocks:
        # I can't loop over the frames directly, so I'll keep track
        # of the frame with these two variables
        dense_idx = -1
        cur_A = -1

        block_type, ai, ni, li, aj, nj, lj = tuple(idx)

        # offset of the orbital block within the pair block in the matrix
        ki_offset = orbs_offset[(ai, ni, li)]
        kj_offset = orbs_offset[(aj, nj, lj)]
        same_koff = ki_offset == kj_offset

        # get the slices of the dense hamiltonian in order to assign
        # the values from the tensormap
        # this is the computationally expensive part of the function
        # we use caching to speed up the process a bit
        fslices, islices, jslices, islices2, jslices2 = _get_slices(
            block=block,
            cur_A=cur_A,
            dense_idx=dense_idx,
            atom_blocks_idx=atom_blocks_idx,
            ki_offset=ki_offset,
            kj_offset=kj_offset,
            li=li,
            lj=lj,
            idx=idx,
            slices_cache=SLICES_CACHE,
        )

        if block_type == 0:
            values = torch.transpose(block.values, 1, 2).reshape(-1)
            dense[fslices, islices, jslices] = values
            if not same_koff:
                dense[fslices, jslices, islices] = values

        elif block_type == 2:
            values = torch.transpose(block.values, 1, 2).reshape(-1)
            dense[fslices, islices, jslices] = values
            dense[fslices, jslices, islices] = values

        elif block_type == 1:
            values = torch.transpose(block.values, 1, 2).reshape(-1) / (2**0.5)
            valuesT = block.values.reshape(-1) / (2**0.5)
            dense[fslices, islices, jslices] += values
            dense[fslices, jslices, islices] += values
            if not same_koff:
                dense[fslices, islices2, jslices2] += valuesT
                dense[fslices, jslices2, islices2] += valuesT

        elif block_type == -1:
            values = torch.transpose(block.values, 1, 2).reshape(-1) / (2**0.5)
            valuesT = block.values.reshape(-1) / (2**0.5)
            dense[fslices, islices, jslices] += values
            dense[fslices, jslices, islices] += values
            if not same_koff:
                dense[fslices, islices2, jslices2] -= valuesT
                dense[fslices, jslices2, islices2] -= valuesT

    return dense


def _get_slices(
    block: TensorBlock,
    cur_A: int,
    dense_idx: int,
    atom_blocks_idx: Dict,
    ki_offset: int,
    kj_offset: int,
    li: int,
    lj: int,
    idx: Labels,
    slices_cache: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    islices = []
    jslices = []
    fslices = []
    islices2 = []
    jslices2 = []
    # loops over samples (structure, i, j)
    for A, i, j in block.samples:
        # check if we have to update the frame and index
        if A != cur_A:
            cur_A = A
            dense_idx += 1

        # coordinates of the atom block in the matrix
        ki_base, kj_base = atom_blocks_idx[(dense_idx, i, j)]

        fslice, islice, jslice, islice2, jslice2 = _get_slices_cached(
            ki_base=ki_base,
            kj_base=kj_base,
            ki_offset=ki_offset,
            kj_offset=kj_offset,
            li=li,
            lj=lj,
            dense_idx=dense_idx,
            block_idx=idx,
            slices_cache=slices_cache,
        )

        islices.append(islice)
        jslices.append(jslice)
        fslices.append(fslice)
        islices2.append(islice2)
        jslices2.append(jslice2)

    fslices = np.concatenate(fslices)
    islices = np.concatenate(islices)
    islices2 = np.concatenate(islices2)
    jslices = np.concatenate(jslices)
    jslices2 = np.concatenate(jslices2)

    return fslices, islices, jslices, islices2, jslices2


def _get_slices_cached(
    ki_base: int,
    kj_base: int,
    ki_offset: int,
    kj_offset: int,
    li: int,
    lj: int,
    dense_idx: int,
    block_idx: Labels,
    slices_cache: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    key = (ki_base, kj_base, ki_offset, kj_offset, li, lj, block_idx)
    # try to get the cached slices
    slices = slices_cache.get(key, None)

    # if slices are not cached, compute them
    if slices is None:
        islice = np.arange(ki_base + ki_offset, ki_base + ki_offset + 2 * li + 1)
        islice = np.tile(islice, 2 * lj + 1)

        jslice = np.arange(
            kj_base + kj_offset, kj_base + kj_offset + 2 * lj + 1
        ).repeat(2 * li + 1)

        fslice = np.array([dense_idx]).repeat((2 * li + 1) * (2 * lj + 1))

        islice2 = np.arange(ki_base + kj_offset, ki_base + kj_offset + 2 * lj + 1)
        islice2 = np.tile(islice2, 2 * li + 1)

        jslice2 = np.arange(
            kj_base + ki_offset, kj_base + ki_offset + 2 * li + 1
        ).repeat(2 * lj + 1)

        # cache the slices
        slices_cache[key] = (islice, jslice, islice2, jslice2)

    # use the cached slices
    else:
        islice, jslice, islice2, jslice2 = slices
        fslice = np.array([dense_idx]).repeat((2 * li + 1) * (2 * lj + 1))

    return fslice, islice, jslice, islice2, jslice2
