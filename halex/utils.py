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
