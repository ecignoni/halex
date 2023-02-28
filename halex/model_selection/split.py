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
