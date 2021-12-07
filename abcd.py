from pathlib import Path
from itertools import chain, product
import pandas as pd
idx = pd.IndexSlice

PATH = Path('inputs/ABCD')
INPUTS = {
    'fcon': 'betnet0',
    'scon': 'dti_p1',
    'sconfull': 'dmdtifp1',
    'imgincl': 'imgincl0',
    'mri': 'mri0',
    'covar': Path('outputs/abcd_covariates.csv')
}

def _inputpath(name):
    if name in INPUTS:
        name = INPUTS[name]
    if isinstance(name, Path):
        return name

    path = list(PATH.glob(f"*{name}*.txt"))
    if len(path) < 1:
        raise ValueError('No matches: ' + name)
    if len(path) > 1:
        raise ValueError('Multiple matches: '
                         + ', '.join([p.stem for p in path]))
    return path[0]

INDEX = {
    'subject': ['src_subject_id', 'subjectkey'],
    'event': ['eventname', 'visit']
}

def _setindex(table):
    index_cols = None
    for tryindex in product(*INDEX.values()):
        if set(tryindex).issubset(table.columns):
            index_cols = list(tryindex)
            break
    if index_cols is None:
        raise ValueError('No valid ABCD index!')

    # set index and remove other index cols
    table = table.set_index(index_cols).rename_axis(INDEX.keys())
    table = table.drop(columns=sum(INDEX.values(), []), errors='ignore')
    return table

def _eventcode(table):
    table = table.reset_index()

    # code events by year
    df = table['event'].str.split('_', n=2, expand=True)
    def datecode(row):
        n = int(row[0]) if row[0].isdigit() else 0
        return n / (12 if row[1] == 'month' else 1)
    table['event'] = df.apply(datecode, axis=1)

    table = table.set_index(list(INDEX.keys()))
    return table

def load(name, descriptions=False):
    """
    Load a table of ABCD data indexed by (subject, event).

    Params:
        name: str. If not an INPUTS key, searches PATH.
        descriptions: Load column descriptions instead of data

    Returns:
        table: DataFrame
    """
    path = _inputpath(name)
    if descriptions:
        table = pd.read_table(path, nrows=1)
    else:
        table = pd.read_table(path, skiprows=[1])

    # drop useless cols
    ignore = ['collection_id', 'dataset_id', 'collection_title']
    ignore.append(f"{path.stem}_id")
    table = table.drop(columns=ignore, errors='ignore')

    if descriptions:
        table = _setindex(table).reset_index(drop=True)
    else:
        table = _eventcode(_setindex(table))
    return table

def longitudinal(table, interval, aggregate=False, dropna=False):
    """
    Convert table into a longitudinally ordered ABCD dataset.

    Params:
        table: DataFrame
        interval: int. Years between data points.
        aggregate: bool, str, or func
            * False: Do not aggregate across interval
            * True: Take first value of each interval
            * str: Apply a GroupBy function to each interval
            * func: Apply func to each interval
        dropna: Drop subjects with any missing data?

    Returns:
        dataset: DataFrame indexed and sorted by (subject, event)
    """
    # events IntervalIndex
    table = table.sort_index().reset_index(level='event')
    intervals = pd.interval_range(
        table['event'].min(), table['event'].max() + interval,
        freq=interval, closed='left'
    )
    table['event'] = pd.cut(table['event'], intervals)
    table = table.set_index('event', append=True, drop=True)

    if aggregate:
        table_g = table.groupby(table.index.names)
        if callable(aggregate):
            table = table_g.apply(aggregate)
        elif isinstance(aggregate, str):
            table = getattr(table_g, aggregate)()
        else:
            table = table_g.first()
    else:
        table = table.loc[~table.index.duplicated(keep='first')]

    if dropna:
        table = table.dropna(how='any')
    else:
        table = table.dropna(how='all')

    dataset = _longitudinal(table)
    return dataset

def _longitudinal(data):
    num_events = data.index.get_level_values('event').nunique()
    sub_counts = data.groupby(level=0).size()
    subs_long = sub_counts.index[sub_counts == num_events]
    return data.loc[subs_long]


## PHENOTYPES
PHENOS = {
    'ASD': {'screen0': ['scrn_asd'], 'mhp0': ['ssrs_p_ss_sum']},
    'ADHD': {'cbcls0': ['cbcl_scr_syn_attention_r']},
    'BIP': {'mhp0': ['pgbi_p_ss_score'], 'mhy0': ['sup_y_ss_sum']},
    'ANX': {'cbcls0': ['cbcl_scr_syn_anxdep_r', 'cbcl_scr_dsm5_anxdisord_r']},
    'MDD': {'cbcls0': ['cbcl_scr_syn_anxdep_r', 'cbcl_scr_syn_withdep_r',
                       'cbcl_scr_dsm5_depress_r']},
    'SCZ': {'mhy0': ['pps_y_ss_number']},
    'ALZ': {}
}
KSADS = {
    'ASD': ['ksads_18_903'],
    'ADHD': [f"ksads_14_{i}" for i in range(853, 857)],
    'BIP': [f"ksads_2_{i}" for i in range(830, 840)],
    'ANX': ['ksads_5_857', 'ksads_5_858', 'ksads_6_859', 'ksads_6_860',
            'ksads_7_861', 'ksads_7_862', 'ksads_8_863', 'ksads_8_864',
            'ksads_9_867', 'ksads_9_868', 'ksads_10_869', 'ksads_10_870'],
    'MDD': [f"ksads_1_{i}" for i in range(840, 848)],
    'SCZ': [f"ksads_4_{i}" for i in chain(range(826, 830), range(849, 853))]
}

def load_pheno(pheno, descriptions=False, interval=None, **kwargs):
    """
    Load an ABCD phenotype dataset.

    Params:
        pheno: str. A PHENOS key
        descriptions: Load column descriptions instead of data
        interval: int or falsy. If int, longitudinal interval.
        **kwargs: passed to longitudinal

    Returns:
        dataset: DataFrame indexed and sorted by (subject, event)
    """
    if pheno not in PHENOS:
        raise ValueError(f"Invalid phenotype: {pheno}")
    table_cols = PHENOS[pheno].copy()

    if pheno in KSADS:
        table_cols['ksad0'] = [col + '_p' for col in KSADS[pheno]]
        table_cols['ksad5'] = [col + '_t' for col in KSADS[pheno]]

    table = pd.concat([load(table, descriptions=descriptions)[cols]
                       for table, cols in table_cols.items()], axis=1)

    # NA: 555, 777, 888, 999
    table = table.where(table < 555)

    if descriptions or not interval:
        dataset = table.sort_index()
    else:
        dataset = longitudinal(table, interval, **kwargs)

    return dataset


#TODO
## IMAGING

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
        dropna: drop subjects with missing data?
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
        dropna: drop subjects with missing data?
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


## COVARIATES

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
