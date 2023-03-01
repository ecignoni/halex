import pickle
import ase
import json
import numpy as np
import ase.io
import torch

import equistore
from equistore import TensorBlock, TensorMap
import equistore.operations as eqop


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


def fix_pyscf_l1(dense, frame, orbs):
    """pyscf stores l=1 terms in a xyz order, corresponding to (m=0, 1, -1).
    this converts into a canonical form where m is sorted as (-1, 0,1)"""
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


def train_test_split(*elements, n_frames, train_size=0.8):
    n_train = int(n_frames * train_size)
    res = []
    for elem in elements:
        if isinstance(elem, equistore.TensorMap):
            # train
            res.append(
                eqop.slice(
                    elem,
                    samples=equistore.Labels(
                        names=["structure"],
                        values=np.asarray(range(n_train), dtype=np.int32).reshape(
                            -1, 1
                        ),
                    ),
                )
            )
            # test
            res.append(
                eqop.slice(
                    elem,
                    samples=equistore.Labels(
                        names=["structure"],
                        values=np.asarray(
                            range(n_train, n_frames), dtype=np.int32
                        ).reshape(-1, 1),
                    ),
                )
            )
        else:
            # train
            res.append(elem[:n_train])
            # test
            res.append(elem[n_train:n_frames])
    return tuple(res)


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
