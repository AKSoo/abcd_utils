"""
Python version of R ComBat yoinked from
https://rdrr.io/bioc/sva/src/R/ComBat.R
and ported for connectivity data
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import warnings
from sklearn.exceptions import ConvergenceWarning


def _lambda_invgamma(mean, var):
    return mean**2 / var + 2

def _theta_invgamma(mean, var):
    return mean**3 / var + mean

def _postmean(n, g_hat, g_bar, t_bar2, d_star):
    return (n*t_bar2*g_hat + d_star*g_bar) / (n*t_bar2 + d_star)

def _postvar(n, sum_sq, l_bar, th_bar):
    return (0.5*sum_sq + th_bar) / (0.5*n + l_bar - 1)

def _em_fit(z, g_hat, g_bar, t_bar2,
            d_hat, l_bar, th_bar,
            tol, max_iter):
    n = z.shape[0]
    g_old, d_old = g_hat, d_hat

    for n_iter in range(1, max_iter+1):
        g_new = _postmean(n, g_hat, g_bar, t_bar2, d_old)
        d_new = _postvar(n, ((z - g_new)**2).sum(axis=0), l_bar, th_bar)

        change = max((np.abs(g_new - g_old) / g_old).max(),
                     (np.abs(d_new - d_old) / d_old).max())
        if change < tol:
            converged = True
            break
        g_old = g_new
        d_old = d_new

    if not converged:
        warnings.warn('Batch did not converge!', ConvergenceWarning)
    return {'gamma': g_new, 'delta': d_new, 'n_iter': n_iter}


def combat(data, batch, covars=None, tol=0.0001, max_iter=100,
           n_procs=1, verbose=False):
    """Adjust for batch effects using an empirical Bayes method.
    Does NOT handle missing data.

    Parameters
    ----------
    data : pandas.DataFrame
        (n, features) data to adjust
    batch : pandas.Series
        Batch covariate
    covars : pandas.DataFrame, optional
        Other covariates. Non-numeric columns will be dummy encoded.
    tol : float
        Convergence criterion for model fitting
    max_iter : int
        Maximum number of iterations for model fitting
    n_procs : int
        Number of processes for parallelization
    verbose : bool, optional
        If True, print status messages.

    Returns
    -------
    adjusted : pandas.DataFrame
        (n, features) data adjusted for batch effects
    """
    if data.isna().any().any():
        raise ValueError('Cannot handle missing data.')

    design = pd.get_dummies(batch.loc[data.index], prefix='_batch',
                            dtype=float)
    n_array, n_batch = design.shape
    if covars is not None:
        design = design.join(pd.get_dummies(covars, drop_first=True,
                                            dtype=float))
    if design.isna().any().any():
        raise ValueError('Missing data in covariates.')

    if verbose:
        print(f'Found {n_batch} batches')
    Y = data.to_numpy()
    X = design.to_numpy()
    X_batch, X_cov = X[:, :n_batch], X[:, n_batch:]
    n_batches = X_batch.sum(axis=0)
    batches = [b.nonzero() for b in X_batch.T]

    if verbose:
        print('Standardizing data')
    B_hat = np.linalg.solve(X.T @ X, X.T @ Y)
    var_pooled = np.full(n_array, 1/n_array) @ (Y - X @ B_hat)**2
    stand_mean = ((n_batches / n_array) @ B_hat[:n_batch, :]
                  + X_cov @ B_hat[n_batch:, :])
    Z = (Y - stand_mean) / np.sqrt(var_pooled)

    if verbose:
        print('Fitting location/scale model and finding priors')
    gamma_hat = np.linalg.solve(X_batch.T @ X_batch, X_batch.T @ Z)
    gamma_bar, tau_bar2 = gamma_hat.mean(axis=1), gamma_hat.var(axis=1, ddof=1)

    delta_hat = np.vstack([Z[batch].var(axis=0, ddof=1) for batch in batches])
    V, S2 = delta_hat.mean(axis=1), delta_hat.var(axis=1, ddof=1)
    lambda_bar, theta_bar = _lambda_invgamma(V, S2), _theta_invgamma(V, S2)

    if verbose:
        print('Finding parametric adjustments')
    batch_fits = Parallel(n_jobs=n_procs)(delayed(_em_fit)(
            Z[batches[i]], gamma_hat[i], gamma_bar[i], tau_bar2[i],
            delta_hat[i], lambda_bar[i], theta_bar[i],
            tol, max_iter
    ) for i in range(n_batch))
    gamma_star = np.array([result['gamma'] for result in batch_fits])
    delta_star = np.array([result['delta'] for result in batch_fits])

    if verbose:
        print('Adjusting data')
    adjusted = (np.sqrt(var_pooled / (X_batch @ delta_star))
                * (Z - X_batch @ gamma_star)) + stand_mean

    return pd.DataFrame(adjusted, index=data.index, columns=data.columns)
