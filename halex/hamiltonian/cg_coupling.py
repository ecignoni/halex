from .tensormap_dense import _components_idx
from ..builder import TensorBuilder
from ..rotations import ClebschGordanReal

import numpy as np
import torch


# ===================================================================
# Clebsch Gordan Coupling / Decoupling
# ===================================================================


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

        # Moves the components at the end as cg.couple assumes so
        decoupled = torch.moveaxis(block.values, -1, -2).reshape(
            (len(block.samples), len(block.properties), 2 * li + 1, 2 * lj + 1)
        )
        # selects the (only) key in the coupled dictionary (l1 and l2
        # that gave birth to the coupled terms L, with L going from
        # |l1 - l2| up to |l1 + l2|
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
        # last key name is L, we remove it here
        blocks.keys.names[:-1],
        # sample_names from the blocks
        # this is because, e.g. for multiple molecules, we
        # may have an additional sample name indexing the
        # molecule id
        blocks.sample_names,
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
