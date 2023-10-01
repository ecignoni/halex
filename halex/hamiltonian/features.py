from ..rascal_wrapper import RascalSphericalExpansion, RascalPairExpansion
from ..acdc_mini import acdc_standardize_keys, cg_increment

import equistore
from equistore import Labels, TensorBlock, TensorMap
from equistore.io import save as equisave
import numpy as np


# ============================================================================
# Features
# ============================================================================


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
        # ===
        # `Try to handle the case of no computed features
        if len(b.samples.tolist()) == 0:
            samples_array = b.samples
        else:
            samples_array = np.vstack(b.samples.tolist())
            samples_array = np.hstack([samples_array, samples_array[:, -1:]]).astype(
                np.int32
            )
        # ===
        # samples_array = np.vstack(b.samples.tolist())
        blocks.append(
            TensorBlock(
                samples=Labels(
                    names=b.samples.names + ("neighbor",),
                    values=samples_array,
                    # values=np.asarray(
                    #     np.hstack([samples_array, samples_array[:, -1:]]),
                    #     dtype=np.int32,
                    # ),
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


def compute_ham_features(
    rascal_hypers, frames, cg, lcut, saveto=None, verbose=False, global_species=None
):
    if verbose:
        print("Computing |rho_i>")
    spex = RascalSphericalExpansion(rascal_hypers)
    rhoi = spex.compute(frames, global_species=global_species)

    if verbose:
        print("Computing |g_ij>")
    pairs = RascalPairExpansion(rascal_hypers)
    gij = pairs.compute(frames, global_species=global_species)

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


def is_core_feature(key):
    """
    Given a key of a TensorMap of features, tells if it
    can be used to learn a core element of the Fock matrix.
    Only considers 1s AOs as being part of the core.
    """
    c0 = key["spherical_harmonics_l"] == 0
    c1 = key["species_center"] != 1
    c2 = key["species_neighbor"] != 1
    c3 = key["block_type"] == 0
    is_core = c0 and c1 and c2 and c3
    return is_core


def drop_noncore_features(feats: TensorMap) -> TensorMap:
    "drop every block of features that cannot be used to learn a core element"
    keys_to_drop = []
    for key in feats.keys:
        if not is_core_feature(key):
            keys_to_drop.append(list(key))
    keys_to_drop = Labels(feats.keys.names, values=np.array(keys_to_drop))
    return equistore.drop_blocks(feats, keys=keys_to_drop)
