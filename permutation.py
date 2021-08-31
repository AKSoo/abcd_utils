import numpy as np
from joblib import Parallel, delayed


def _permutations(rng, x, n):
    for _ in range(n):
        yield rng.permutation(x)

def permute_func(func, X, Y, n_perm=99,
                 random_state=None, n_procs=1):
    """
    Permute Y, apply func to generate a distribution.

    Params:
        func: takes X, Y as params and returns a number
        X: ndarray
        Y: ndarray, to permute along first axis
        n_perm: int number of permutations
        random_state: int random seed
        n_procs: int number of processes for parallelization

    Returns:
        distrib: ndarray (n_perm,)
    """
    rng = np.random.default_rng(random_state)
    perms = _permutations(rng, len(Y), n_perm)

    distrib = Parallel(n_jobs=n_procs)(delayed(func)(X, Y[perm])
                                       for perm in perms)
    return np.array(distrib)
