from __future__ import annotations

import pickle
import ase
import json
import numpy as np
import ase.io
import torch

from equistore import Labels, TensorBlock, TensorMap


# very exhaustive list
symbol2atomic_number = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
}
atomic_number2symbol = {v: k for (k, v) in symbol2atomic_number.items()}


def get_ao_labels(orbs, atomic_numbers):
    """
    Params
    orbs: dictionary with key = z (int) and value = list of [n, l, m]
          for each AO
    atomic_numbers: list of atomic numbers for each atom
    Returns
    ao_labels: list of (index (int), symbol (str), label)
    """
    ao_labels = []
    for i, a in enumerate(atomic_numbers):
        symbol = atomic_number2symbol[a]
        for ao in orbs[a]:
            label = (i, symbol, ao)
            ao_labels.append(label)
    return ao_labels


def fix_pyscf_l1(dense, frame, orbs, return_index=False):
    """converts from the pyscf ordering for l=1, (1, -1, 0), to
    the canonical ordering (-1, 0, 1)"""
    idx = []
    iorb = 0
    atoms = list(frame.numbers)
    for atype in atoms:
        cur = ()
        for ia, a in enumerate(orbs[atype]):
            n, l, m = a
            if (n, l) != cur:
                if l == 1:
                    idx += [iorb + 1, iorb + 2, iorb]
                else:
                    idx += range(iorb, iorb + 2 * l + 1)
                iorb += 2 * l + 1
                cur = (n, l)
    if return_index:
        return idx
    else:
        return dense[idx][:, idx]


def recover_pyscf_l1(dense, frame, orbs, return_index=False):
    """converts from the canonical ordering (-1, 0, 1) to the pyscf
    ordering (1, -1, 0) for l=1"""
    idx = []
    iorb = 0
    atoms = list(frame.numbers)
    for atype in atoms:
        cur = ()
        for ia, a in enumerate(orbs[atype]):
            n, l, m = a
            if (n, l) != cur:
                if l == 1:
                    idx += [iorb + 2, iorb, iorb + 1]
                else:
                    idx += range(iorb, iorb + 2 * l + 1)
                iorb += 2 * l + 1
                cur = (n, l)
    if return_index:
        return np.array(idx)
    else:
        return dense[idx][:, idx]


def fix_pyscf_l1_orbs(orbs):
    orbs = orbs.copy()
    for key in orbs:
        new_orbs = []
        i = 0
        while True:
            try:
                n, l, m = orbs[key][i]
            except IndexError:
                break
            i += 2 * l + 1
            for m in range(-l, l + 1, 1):
                new_orbs.append([n, l, m])
        orbs[key] = new_orbs
    return orbs


def load_frames(path, n_frames=None):
    frames = ase.io.read(
        path,
        ":%d" % n_frames if n_frames is not None else ":",
    )
    # why this?
    for f in frames:
        f.cell = [100, 100, 100]
        f.positions += 50
    return frames


def load_orbs(path):
    with open(path, "r") as handle:
        jorbs = json.load(handle)
        # try:
        #     jorbs = json.loads(json.load(handle))
        # except Exception:
        #     jorbs = json.load(handle)
    orbs = {}
    zdic = {"O": 8, "C": 6, "H": 1}
    for k in jorbs:
        orbs[zdic[k]] = jorbs[k]
    return orbs


def dump_dict(path, d):
    with open(path, "wb") as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dict(path):
    with open(path, "rb") as handle:
        d = pickle.load(handle)
    return d


def _convert_tensormap_dtype(tmap, convert_dtype_fn):
    blocks = []
    for _, block in tmap:
        values = convert_dtype_fn(block.values)
        block = TensorBlock(
            values=values,
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )
        blocks.append(block)
    return TensorMap(tmap.keys, blocks)


def tensormap_as_torch(tmap):
    def convert_dtype_fn(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).type(torch.float64)
        return x

    return _convert_tensormap_dtype(tmap, convert_dtype_fn)


def tensormap_as_numpy(tmap):
    def convert_dtype_fn(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy().copy()
        return x

    return _convert_tensormap_dtype(tmap, convert_dtype_fn)


def shift_structure_by_n(tmap: TensorMap, n: int) -> TensorMap:
    """shifts the "structure" value by an integer n

    Given a TensorMap with samples (structure, center, neighbor),
    returns the a new TensorMap object with the values of
    "structure" shifted by an integer n.
    Useful if you want to join two TensorMap objects avoiding
    the creation of a new "tensor" dimension.
    """
    if tmap.sample_names != ("structure", "center", "neighbor"):
        raise ValueError(f"input TensorMap has wrong samples: {tmap.sample_names}")

    blocks = []

    for _, block in tmap:
        # shift the structure values
        samples = np.array(block.samples.asarray())
        samples[:, 0] += n
        samples = Labels(block.samples.names, values=samples)

        # check if numpy or torch, otherwise complain
        if isinstance(block.values, np.ndarray):
            values = block.values.copy()
        elif isinstance(block.values, torch.Tensor):
            values = block.values.clone()
        else:
            raise ValueError("TensorMap values neighter np.ndarray nor torch.Tensor")

        # new block with update structure values
        block = TensorBlock(
            values=values,
            samples=samples,
            components=block.components,
            properties=block.properties,
        )

        blocks.append(block)

    return TensorMap(tmap.keys, blocks)


def get_feature_block(features: TensorMap, key: Labels) -> TensorBlock:
    block_type, ai, ni, li, aj, nj, lj, L = key
    inversion_sigma = (-1) ** (li + lj + L)
    block = features.block(
        block_type=block_type,
        spherical_harmonics_l=L,
        inversion_sigma=inversion_sigma,
        species_center=ai,
        species_neighbor=aj,
    )
    return block


def drop_target_samples(
    feats: TensorMap, targ_coupled: TensorMap, verbose: bool = False
) -> TensorMap:
    blocks = []
    all_keys_matched = True

    for key, block in targ_coupled:
        feat_samples = get_feature_block(feats, key).samples.asarray().copy()
        targ_samples = block.samples.asarray().copy()
        # find samples in common between the two TensorMaps
        idx_common = (targ_samples[:, None] == feat_samples).all(-1).any(-1)

        if not np.all(idx_common):
            all_keys_matched = False
            if verbose:
                keystr = (
                    "("
                    + "".join(
                        ["{:s}={}, ".format(n, v) for v, n in zip(key, key.dtype.names)]
                    )[:-1]
                    + ")"
                )
                print("mismatch for key:", keystr)

        targ_samples = targ_samples[idx_common]
        targ_samples = Labels(block.samples.names, values=targ_samples)

        if isinstance(block.values, np.ndarray):
            targ_values = block.values.copy()
        elif isinstance(block.values, torch.Tensor):
            targ_values = block.values.clone()
        else:
            raise ValueError("TensorMap values neither np.ndarray nor torch.Tensor")

        targ_values = targ_values[idx_common]

        block = TensorBlock(
            values=targ_values,
            samples=targ_samples,
            components=block.components,
            properties=block.properties,
        )

        blocks.append(block)

    if all_keys_matched and verbose:
        print("all keys matched successfully")

    return TensorMap(targ_coupled.keys, blocks)


def fix_pyscf_l1_crossoverlap(cross_ovlp, frame, orbs_sb, orbs_bb):
    indeces_sb = fix_pyscf_l1(None, frame, orbs_sb, return_index=True)
    indeces_bb = fix_pyscf_l1(None, frame, orbs_bb, return_index=True)
    cross_ovlp = cross_ovlp[indeces_sb][:, indeces_bb]
    return cross_ovlp


def load_cross_ovlps(path, frames, orbs_sb, orbs_bb, indices):
    """
    Loads the cross overlap matrices from path, and reorders
    the l1 terms of both AO bases.
    """
    cross_ovlps = torch.from_numpy(np.load(path)[indices])
    # reorder the AO basis on both sides
    cross_ovlps = torch.stack(
        [
            fix_pyscf_l1_crossoverlap(cross_s, frame, orbs_sb, orbs_bb)
            for cross_s, frame in zip(cross_ovlps, frames)
        ]
    )
    return cross_ovlps
