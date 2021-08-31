import pandas as pd
import mapply
from statsmodels.tools import tools as sm_tools


def residuals(feature, model=None, regressors=None,
              groups=None, exog_re=None,
              return_result=False, **kwargs):
    """
    Regress out variables from a feature with a statsmodels model.

    Params:
        feature: Series
        model: statsmodels model (OLS, MixedLM)
        regressors: DataFrame
        groups: [MixedLM] Series of independent groups
        exog_re: [MixedLM] DataFrame of random effect covariates
        return_result: return result instead of residuals
        **kwargs: passed to model.fit()

    Returns:
        resid: Series, same index as feature
        OR
        result: statsmodels results
    """
    if model is None or regressors is None:
        raise ValueError('Model or regressors not specified.')

    # MixedLM(missing='drop') doesn't work
    na_filter = feature.notna()
    endog = feature.loc[na_filter]
    exog = sm_tools.add_constant(pd.get_dummies(
        regressors.loc[feature.index].loc[na_filter],
        drop_first=True
    ))
    if groups is not None:
        groups = groups.loc[feature.index].loc[na_filter]
    if exog_re is not None:
        exog_re = exog_re.loc[feature.index].loc[na_filter]

    # fit
    result = model(endog, exog, groups=groups, exog_re=exog_re).fit(**kwargs)
    if return_result:
        return result

    resid = result.resid.reindex(feature.index)
    resid.name = feature.name
    return resid


def residualize(data, model, confounds, return_results=False,
                n_procs=1, progressbar=False, **kwargs):
    """
    Regress out confounds from data with a statsmodels model.
    Only the subset of data with matching confounds are processed.

    Params:
        data: (n, features) DataFrame
        model: statsmodels model (OLS, MixedLM)
        confounds: DataFrame
        return_results: return results instead of residuals
        n_procs: int number of processes for parallelization
        progressbar: show progress
        **kwargs: passed to residuals()

    Returns:
        resids: (n, features) DataFrame
        OR
        results: (features,) Series of statsmodels results
    """
    sample = data.index.intersection(confounds.index)
    # convert Series to DataFrame
    data = pd.DataFrame(data.loc[sample])
    confounds = pd.DataFrame(confounds.loc[sample])
    
    if n_procs > 1:
        mapply.init(n_workers=n_procs, chunk_size=1, progressbar=progressbar)
        data_apply = data.mapply
    else:
        data_apply = data.apply

    r = data_apply(residuals, model=model, regressors=confounds,
                   return_result=return_results, **kwargs)
    return r
