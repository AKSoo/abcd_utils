import pandas as pd
from statsmodels.tools.tools import add_constant


def regress(y, X, model, output=None,
            model_kws={}, fit_kws={}):
    """
    Regress X on y with a statsmodels model.
    Missing data are skipped.

    Params:
        y: Series
        X: DataFrame
        model: statsmodels model
        output: str. What statsmodels result attribute to return.
            If None, return the whole RegressionResults.
        model_kws: kwargs for model
        fit_kws: kwargs for model.fit

    Returns:
        result: Certain results like 'resid' are reindexed to match y.
    """
    #NOTE missing='drop' is broken for MixedLM
    notna = y.notna()
    endog = y.loc[notna]
    exog = add_constant(pd.get_dummies(X.loc[y.index].loc[notna],
                                       drop_first=True, dtype=float))
    for k, v in model_kws.items():
        if k in ['groups', 'exog_re']:
            model_kws[k] = v.loc[y.index].loc[notna]

    # fit
    result = model(endog, exog, **model_kws).fit(**fit_kws)

    if output is None:
        return result

    result = getattr(result, output)
    if output in ['resid']:
        result = result.reindex(y.index)
        result.name = y.name
    return result


def mass_regress(Y, X, model, output=None,
                 n_procs=1, progressbar=False, **kwargs):
    """
    Regress X on each target in Y with a statsmodels model.
    Only matching indices are regressed. Missing data are skipped.

    Params:
        Y: (n, targets) DataFrame or Series
        X: DataFrame or Series
        model: statsmodels model
        output: str. What statsmodels result attribute to return.
            If None, return Series of RegressionResults.
        n_procs: int number of processes for parallelization
        progressbar: show progress when parallel
        **kwargs: kwargs for regress

    Returns:
        results: (n, targets) DataFrame or (targets,) Series
            depending on output
    """
    sample = Y.index.intersection(X.index)
    # convert Series to DataFrame
    targets = pd.DataFrame(Y.loc[sample])
    features = pd.DataFrame(X.loc[sample])

    if n_procs > 1:
        import mapply
        mapply.init(n_workers=n_procs, chunk_size=1, progressbar=progressbar)
        apply = targets.mapply
    else:
        apply = targets.apply

    return apply(regress, X=features, model=model, output=output, **kwargs)
