import numpy as np
import wigners
import torch


class ClebschGordanReal:
    def __init__(self, l_max):
        self._l_max = l_max
        self._cg = {}

        # real-to-complex and complex-to-real transformations as matrices
        r2c = {}
        c2r = {}
        for L in range(0, self._l_max + 1):
            r2c[L] = _real2complex(L)
            c2r[L] = np.conjugate(r2c[L]).T

        for l1 in range(self._l_max + 1):
            for l2 in range(self._l_max + 1):
                for L in range(
                    max(l1, l2) - min(l1, l2), min(self._l_max, (l1 + l2)) + 1
                ):
                    complex_cg = _complex_clebsch_gordan_matrix(l1, l2, L)

                    real_cg = (r2c[l1].T @ complex_cg.reshape(2 * l1 + 1, -1)).reshape(
                        complex_cg.shape
                    )

                    real_cg = real_cg.swapaxes(0, 1)
                    real_cg = (r2c[l2].T @ real_cg.reshape(2 * l2 + 1, -1)).reshape(
                        real_cg.shape
                    )
                    real_cg = real_cg.swapaxes(0, 1)

                    real_cg = real_cg @ c2r[L].T

                    if (l1 + l2 + L) % 2 == 0:
                        rcg = np.real(real_cg)
                    else:
                        rcg = np.imag(real_cg)

                    new_cg = []
                    for M in range(2 * L + 1):
                        cg_nonzero = np.where(np.abs(rcg[:, :, M]) > 1e-15)
                        cg_M = np.zeros(
                            len(cg_nonzero[0]),
                            dtype=[("m1", ">i4"), ("m2", ">i4"), ("cg", ">f8")],
                        )
                        cg_M["m1"] = cg_nonzero[0]
                        cg_M["m2"] = cg_nonzero[1]
                        cg_M["cg"] = rcg[cg_nonzero[0], cg_nonzero[1], M]
                        new_cg.append(cg_M)

                    self._cg[(l1, l2, L)] = new_cg

    def combine(self, rho1, rho2, L):
        # automatically infer l1 and l2 from the size of the coefficients vectors
        l1 = (rho1.shape[1] - 1) // 2
        l2 = (rho2.shape[1] - 1) // 2
        if L > self._l_max or l1 > self._l_max or l2 > self._l_max:
            raise ValueError("Requested CG entry has not been precomputed")

        n_items = rho1.shape[0]
        n_features = rho1.shape[2]
        if rho1.shape[0] != rho2.shape[0] or rho1.shape[2] != rho2.shape[2]:
            raise IndexError("Cannot combine differently-shaped feature blocks")

        rho = np.zeros((n_items, 2 * L + 1, n_features))
        if (l1, l2, L) in self._cg:
            for M in range(2 * L + 1):
                for m1, m2, cg in self._cg[(l1, l2, L)][M]:
                    rho[:, M] += rho1[:, m1, :] * rho2[:, m2, :] * cg

        return rho

    def combine_einsum(self, rho1, rho2, L, combination_string):
        # automatically infer l1 and l2 from the size of the coefficients vectors
        l1 = (rho1.shape[1] - 1) // 2
        l2 = (rho2.shape[1] - 1) // 2
        if L > self._l_max or l1 > self._l_max or l2 > self._l_max:
            raise ValueError(
                "Requested CG entry ", (l1, l2, L), " has not been precomputed"
            )

        n_items = rho1.shape[0]
        if rho1.shape[0] != rho2.shape[0]:
            raise IndexError(
                "Cannot combine feature blocks with different number of items"
            )

        # infers the shape of the output using the einsum internals
        features = np.einsum(combination_string, rho1[:, 0, ...], rho2[:, 0, ...]).shape
        rho = np.zeros((n_items, 2 * L + 1) + features[1:])

        if (l1, l2, L) in self._cg:
            for M in range(2 * L + 1):
                for m1, m2, cg in self._cg[(l1, l2, L)][M]:
                    rho[:, M, ...] += np.einsum(
                        combination_string, rho1[:, m1, ...], rho2[:, m2, ...] * cg
                    )

        return rho

    def couple(self, decoupled, iterate=0):
        """
        Goes from an uncoupled product basis to a coupled basis.
        A (2l1+1)x(2l2+1) matrix transforming like the outer product of
        Y^m1_l1 Y^m2_l2 can be rewritten as a list of coupled vectors,
        each transforming like a Y^L irrep.

        The process can be iterated: a D dimensional array that is the product
        of D Y^m_l can be turned into a set of multiple terms transforming as
        a single Y^M_L.

        decoupled: array or dict
            (...)x(2l1+1)x(2l2+1) array containing coefficients that
            transform like products of Y^l1 and Y^l2 harmonics. can also
            be called on a array of higher dimensionality, in which case
            the result will contain matrices of entries.
            If the further index also correspond to spherical harmonics,
            the process can be iterated, and couple() can be called onto
            its output, in which case the decoupling is applied to each
            entry.

        iterate: int
            calls couple iteratively the given number of times. equivalent to
            couple(couple(... couple(decoupled)))

        Returns:
        --------
        A dictionary tracking the nature of the coupled objects. When called one
        time, it returns a dictionary containing (l1, l2) [the coefficients of the
        parent Ylm] which in turns is a dictionary of coupled terms, in the form
        L:(...)x(2L+1)x(...) array. When called multiple times, it applies the
        coupling to each term, and keeps track of the additional l terms, so that
        e.g. when called with iterate=1 the return dictionary contains terms of
        the form
        (l3,l4,l1,l2) : { L: array }


        Note that this coupling scheme is different from the NICE-coupling where
        angular momenta are coupled from left to right as (((l1 l2) l3) l4)... )
        Thus results may differ when combining more than two angular channels.
        """

        coupled = {}

        # when called on a matrix, turns it into a dict form to which we can
        # apply the generic algorithm
        if not isinstance(decoupled, dict):
            l2 = (decoupled.shape[-1] - 1) // 2
            decoupled = {(): {l2: decoupled}}

        # runs over the tuple of (partly) decoupled terms
        for ltuple, lcomponents in decoupled.items():
            # each is a list of L terms
            for lc in lcomponents.keys():
                # this is the actual matrix-valued coupled term,
                # of shape (..., 2l1+1, 2l2+1), transforming as Y^m1_l1 Y^m2_l2
                dec_term = lcomponents[lc]
                l1 = (dec_term.shape[-2] - 1) // 2
                l2 = (dec_term.shape[-1] - 1) // 2

                # there is a certain redundance: the L value is also the last entry
                # in ltuple
                if lc != l2:
                    raise ValueError(
                        "Inconsistent shape for coupled angular momentum block."
                    )

                # in the new coupled term, prepend (l1,l2) to the existing label
                coupled[(l1, l2) + ltuple] = {}
                for L in range(
                    max(l1, l2) - min(l1, l2), min(self._l_max, (l1 + l2)) + 1
                ):
                    # ensure that Lterm is created on the same device as the dec_term
                    device = dec_term.device
                    Lterm = torch.zeros(
                        size=dec_term.shape[:-2] + (2 * L + 1,), device=device
                    )
                    for M in range(2 * L + 1):
                        for m1, m2, cg in self._cg[(l1, l2, L)][M]:
                            Lterm[..., M] += dec_term[..., m1, m2] * cg
                    coupled[(l1, l2) + ltuple][L] = Lterm

        # repeat if required
        if iterate > 0:
            coupled = self.couple(coupled, iterate - 1)
        return coupled

    def decouple(self, coupled, iterate=0):
        """
        Undoes the transformation enacted by couple.
        """

        decoupled = {}
        # applies the decoupling to each entry in the dictionary
        for ltuple, lcomponents in coupled.items():
            # the initial pair in the key indicates the decoupled terms that generated
            # the L entries
            l1, l2 = ltuple[:2]

            # shape of the coupled matrix (last entry is the 2L+1 M terms)
            shape = next(iter(lcomponents.values())).shape[:-1]

            device = next(iter(lcomponents.values())).device
            dec_term = torch.zeros(
                shape
                + (  # noqa
                    2 * l1 + 1,
                    2 * l2 + 1,
                ),
                device=device,
            )
            for L in range(max(l1, l2) - min(l1, l2), min(self._l_max, (l1 + l2)) + 1):
                # supports missing L components, e.g. if they are zero because of symmetry
                if L not in lcomponents:
                    continue
                for M in range(2 * L + 1):
                    for m1, m2, cg in self._cg[(l1, l2, L)][M]:
                        dec_term[..., m1, m2] += cg * lcomponents[L][..., M]
            # stores the result with a key that drops the l's we have just decoupled
            if not ltuple[2:] in decoupled:
                decoupled[ltuple[2:]] = {}
            decoupled[ltuple[2:]][l2] = dec_term

        # rinse, repeat
        if iterate > 0:
            decoupled = self.decouple(decoupled, iterate - 1)

        # if we got a fully decoupled state, just return an array
        if ltuple[2:] == ():
            decoupled = next(iter(decoupled[()].values()))
        return decoupled


def _real2complex(L):
    """transformation matrix between spherical harmonics

    Computes the transformation matrix that goes from a set
    of real spherical harmonics, ordered as:

        (l, -l), (l, -l + 1), ..., (l, l - 1), (l, l)

    to a set of complex spherical harmonics (in the same order).
    The transformation matrix can be found in several places,
    e.g.:
    https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form

    Taking the conjugate transpose of the matrix gives the
    transformation from complex to real spherical harmonics.
    As an example, using scipy:

    >>> from scipy.special import sph_harm
    >>> U = _real2complex(1)
    >>> # complex spherical harmonics, ordered from m=-l to m=l
    >>> comp_sph = np.array([
    ... sph_harm(-1, 1, 0.2, 0.2),
    ... sph_harm(0, 1, 0.2, 0.2),
    ... sph_harm(1, 1, 0.2, 0.2)
    ... ])
    >>> real_sph = np.conjugate(U).T @ comp_sph
    >>> assert np.max(abs(comp_sph - U @ real_sph)) < 1e-15
    """
    mult = 2 * L + 1
    mat = np.zeros((mult, mult), dtype=np.complex128)
    # m = 0
    mat[L, L] = 1.0

    if L == 0:
        return mat

    isqrt2 = 1.0 / 2**0.5
    for m in range(1, L + 1):
        # m > 0
        mat[L + m, L + m] = isqrt2 * (-1) ** m
        mat[L + m, L - m] = isqrt2 * 1j * (-1) ** m

        # m < 0
        mat[L - m, L + m] = isqrt2
        mat[L - m, L - m] = -isqrt2 * 1j

    return mat


def _complex_clebsch_gordan_matrix(l1, l2, L):
    r"""clebsch-gordan matrix

    Computes the Clebsch-Gordan (CG) matrix for
    transforming complex-valued spherical harmonics.
    The CG matrix is computed as a 3D array of elements

        < l1 m1 l2 m2 | L M >

    where the first axis loops over m1, the second loops over m2,
    and the third one loops over M.

    For example, using the relation:

        | l1 l2 L M > = \sum_{m1, m2} <l1 m1 l2 m2 | L M > | l1 m1 > | l2 m2 >

    one can obtain the spherical harmonics L from two sets of
    spherical harmonics with l1 and l2 (up to a normalization factor).
    E.g.:

    >>> from scipy.special import sph_harm
    >>> C_112 = _complex_clebsch_gordan_matrix(1, 1, 2)
    >>> comp_sph_1 = np.array([
    ... sph_harm(m, 1, 0.2, 0.2) for m in range(-1, 1+1)
    ... ])
    >>> # obtain the (unnormalized) spherical harmonics with
    >>> l = 2 by contraction over m1 and m2
    >>> comp_sph_2_u = np.einsum("ijk,i,j->k", C_112, comp_sph_1, comp_sph_2)
    >>> # we can check that they differ from the spherical harmonics by a
    >>> # constant factor
    >>> comp_sph_2 = np.array([sph_harm(m, 2, 0.2, 0.2) for m in range(-2, 2+1)])
    >>> print(comp_sph2 / comp_sph_2_u)
    ... [3.23604319-1.69568664e-16j 3.23604319+7.31506235e-17j
    ...  3.23604319+0.00000000e+00j 3.23604319-7.31506235e-17j
    ...  3.23604319+1.69568664e-16j]
    """
    if np.abs(l1 - l2) > L or np.abs(l1 + l2) < L:
        return np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * L + 1), dtype=np.double)
    else:
        return wigners.clebsch_gordan_array(l1, l2, L)
