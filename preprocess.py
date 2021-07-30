import pandas as pd

from statsmodels.tools import tools as sm_tools


def residualize(feature, model=None, regressors=None, groups=None,
                return_result=False, **kwargs):
    """
    Regress out variables from a feature with a statsmodels model.

    Params:
        feature: Series
        model: statsmodels model (OLS, MixedLM)
        regressors: DataFrame
        groups: Series, for mixed effects model
        return_result: also return model fit result?
        **kwargs: passed to model.fit()

    Returns:
        resid: Series, same index as feature
        (result: from statsmodels fit)
    """
    if model is None or regressors is None:
        raise ValueError('Model or regressors not specified.')

    # MixedLM(missing='drop') doesn't work
    na_filter = feature.notna()
    endog = feature.loc[na_filter]
    exog = sm_tools.add_constant(pd.get_dummies(regressors, drop_first=True)).loc[na_filter]
    if groups is not None:
        groups = groups.loc[na_filter]

    result = model(endog, exog, groups=groups).fit(**kwargs)

    resid = result.resid.reindex(feature.index)
    resid.name = feature.name

    if return_result:
        return resid, result
    return resid
