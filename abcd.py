from pathlib import Path
import pandas as pd
idx = pd.IndexSlice

PATH = Path('inputs/ABCD')
INPUTS = {
    'fcon': 'abcd_betnet02.txt',
    'scon': 'abcd_dti_p101.txt',
    'sconfull': 'abcd_dmdtifp101.txt',
    'imgincl': 'abcd_imgincl01.txt',
    'mri': 'abcd_mri01.txt',
    'covar': 'outputs/abcd_covariates.csv'
}
INDEX = ['src_subject_id', 'eventname']
EVENTS = ['baseline_year_1_arm_1', '2_year_follow_up_y_arm_1']

FCON = pd.Series({
    'ad': 'auditory',
    'cgc': 'cingulo-opercular',
    'ca': 'cingulo-parietal',
    'dt': 'default',
    'dla': 'dorsal attention',
    'fo': 'fronto-parietal',
    'n': None,
    'rspltp': 'retrosplenial temporal',
    'smh': 'sensorimotor hand',
    'smm': 'sensorimotor mouth',
    'sa': 'salience',
    'vta': 'ventral attention',
    'vs': 'visual'
})

def fcon_colname(a, b):
    return f"rsfmri_c_ngd_{a}_ngd_{b}"

def scon_colname(a, full=False, metric='fa'):
    return f"dmri_dti{'full' if full else ''}{metric}_fiberat_{a}"

SCAN_INFO = ['mri_info_manufacturer', 'mri_info_manufacturersmn',
             'mri_info_deviceserialnumber', 'mri_info_softwareversion']

_LOAD_COLS = {
    'fcon': {'imgincl': 'imgincl_rsfmri_include',
             'meanmotion': 'rsfmri_c_ngd_meanmotion'},
    'scon': {'imgincl': 'imgincl_dmri_include',
             'meanmotion': 'dmri_dti_meanmotion'}
}


def load_fcon(path=None, include_rec=True, dropna=True,
              exclude_n=True):
    """
    Load a longitudinally ordered ABCD functional connectivity dataset.
    
    Params:
        path: Path. If given, load data from a specific file.
        include_rec: filter by recommended inclusion?
        dropna: drop subject with missing data?
        exclude_n: ignore None "network"?

    Returns:
        dataset: DataFrame indexed and sorted by (subject, event)
        extra: DataFrame of relevant supplementary data
            * mean motion
            * scan info
    """
    if path is not None:
        data = pd.read_table(path, sep=None, engine='python', index_col=INDEX)
    else:
        data = pd.read_table(PATH / INPUTS['fcon'],
                             skiprows=[1], index_col=INDEX)

    # columns
    fcon_codes = FCON.index
    if exclude_n:
        fcon_codes = fcon_codes.drop('n')

    columns = []
    for i in range(len(fcon_codes)):
        for j in range(i+1):
            columns.append(fcon_colname(fcon_codes[i], fcon_codes[j]))

    data_cols = data.loc[:, columns]

    # rows
    if include_rec:
        data_cols = _rec_inclusion(data_cols, 'fcon')
    if dropna:
        data_cols = data_cols.dropna()

    dataset = _longitudinal(data_cols)
    extra = _extra_data(dataset, 'fcon')
    return dataset, extra


def load_scon(path=None, include_rec=True, dropna=True,
              metrics=['fa'], full=False):
    """
    Load a longitudinally ordered ABCD structural connectivity dataset.
    
    Params:
        path: Path. If given, load data from a specific file.
        include_rec: filter by recommended inclusion?
        dropna: drop subject with missing data?
        metrics: list of DTI metrics (fa, md)
        full: full shell DTI data?

    Returns:
        dataset: DataFrame indexed and sorted by (subject, event)
        extra: DataFrame of relevant supplementary data
            * mean motion
            * scan info
    """
    if path is not None:
        data = pd.read_table(path, sep=None, engine='python', index_col=INDEX)
    else:
        if full:
            path = PATH / INPUTS['sconfull']
        else:
            path = PATH / INPUTS['scon']
        data = pd.read_table(path, skiprows=[1], index_col=INDEX)

    # columns
    if full:
        data = data.rename(columns=_nums_sconfull())

    columns = False
    for m in metrics:
        columns |= data.columns.str.startswith(scon_colname('', full, m))

    data_cols = data.loc[:, columns]

    # rows
    if include_rec:
        data_cols = _rec_inclusion(data_cols, 'scon')
    if dropna:
        data_cols = data_cols.dropna()

    dataset = _longitudinal(data_cols)
    extra = _extra_data(dataset, 'scon')
    return dataset, extra


def _rec_inclusion(data, data_type):
    """recommended inclusion on index"""
    imgincl = pd.read_table(PATH / INPUTS['imgincl'],
                            skiprows=[1], index_col=INDEX)
    include = imgincl.loc[imgincl[_LOAD_COLS[data_type]['imgincl']] == 1].index
    return data.loc[data.index.intersection(include), :]


def _longitudinal(data):
    """longitudinal only and ordered"""
    subs_counts = data.groupby(level=0).size()
    subs_long = subs_counts.index[subs_counts == len(EVENTS)]
    return data.loc[idx[subs_long, EVENTS], :]


def _extra_data(data, data_type):
    """extra info for index"""
    mri = pd.read_table(PATH / INPUTS['mri'],
                        skiprows=[1], index_col=INDEX)
    extra = pd.read_table(PATH / INPUTS[data_type],
                          skiprows=[1], index_col=INDEX)

    columns = {_LOAD_COLS[data_type]['meanmotion']: 'meanmotion'}
    extra = extra.loc[data.index, columns.keys()].rename(columns=columns)
    return extra.join(mri[SCAN_INFO])


def _nums_sconfull():
    """nums to names for sconfull columns (match by description)"""
    nums_descs = pd.read_table(PATH / INPUTS['sconfull'], nrows=1).iloc[0]
    scon_descs = pd.read_table(PATH / INPUTS['scon'], nrows=1).iloc[0]
    descs_scon = pd.Series(scon_descs.index, index=scon_descs)
    nums_scon = nums_descs.map(descs_scon).dropna()

    nums_sconfull = nums_scon.str.replace('dmri_dti', 'dmri_dtifull')
    return nums_sconfull


def get_scon_descriptions():
    """
    Get DTI atlas tract descriptions.

    Returns:
        scon_descs: Series of (code, description) for each tract
    """
    descs = pd.read_table(PATH / INPUTS['scon'], nrows=1).iloc[0]
    col_start = scon_colname('')
    desc_starts = ('Average fractional anisotropy within ', 'DTI atlas tract ')

    scon_descs = descs.loc[descs.index.str.startswith(col_start)]
    scon_descs = scon_descs.rename(lambda s: s.replace(col_start, ''))
    scon_descs = (scon_descs.str.replace(desc_starts[0], '')
                  .str.replace(desc_starts[1], ''))

    return scon_descs


def load_covariates(covars=None, simple_race=False):
    """
    Load ABCD covariates.

    Params:
        covars: list of columns to load. If None, all.
        simple_race: Include simple 'race' covariate with [White, Black,
            Asian, Other, Mixed]. Other includes missing.

    Returns:
        covariates: DataFrame indexed by (subject, event)
    """
    covariates = pd.read_csv(PATH / INPUTS['covar'], index_col=INDEX)
    if covars is not None:
        covars = covariates[covars].copy()
    else:
        covars = covariates.copy()

    if simple_race:
        covars['race'] = (covariates['race.6level']
                          .replace('AIAN/NHPI', 'Other').fillna('Other'))

    return covars


def filter_siblings(data, random_state=None):
    """
    Sample 1 subject per family.

    Params:
        data: DataFrame indexed by (subject, event)
        random_state: int random seed

    returns:
        filtered: DataFrame indexed by (subject, event)
    """
    family = load_covariates(covars=['rel_family_id'])
    subs = (data.join(family).groupby('rel_family_id')
            .sample(1, random_state=random_state).index.get_level_values(0))

    filtered = data.loc[subs]
    return filtered
