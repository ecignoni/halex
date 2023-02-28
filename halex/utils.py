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
