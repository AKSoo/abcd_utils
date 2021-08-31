from pathlib import Path
import pandas as pd
idx = pd.IndexSlice

PATH = Path('inputs/ABCD')
OUT_PATH = Path('outputs')
INPUTS = {
    'fcon': 'abcd_betnet02.tsv',
    'scon': 'abcd_dti_p101.tsv',
    'imgincl': 'abcd_imgincl01.tsv',
    'mri': 'abcd_mri01.tsv'
}
INDEX = ['src_subject_id', 'eventname']
EVENTS = ['baseline_year_1_arm_1', '2_year_follow_up_y_arm_1']

FCON_TEMPLATE = 'rsfmri_c_ngd_{0}_ngd_{1}'
FCON = {
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
}

SCON_TEMPLATE = 'dmri_dtifa_fiberat_{0}'

SCAN_INFO = ['mri_info_manufacturer', 'mri_info_manufacturersmn',
             'mri_info_deviceserialnumber', 'mri_info_softwareversion']


def load_mri_data(data_type, path=None, include_rec=True, dropna=False,
                  exclude_n=True):
    """
    Load a longitudinally ordered ABCD MRI dataset.
    * fcon: Gordon network z-transformed correlations
    * scon: DTI atlas tract fractional anisotropy averages

    Params:
        data_type: type of dataset to load
        path: specific ABCD MRI dataset file Path
        include_rec: filter by recommended inclusion?
        dropna: drop subject if any dataset column is NA
        exclude_n: [fcon] ignore None "network"?

    Returns:
        dataset: DataFrame indexed and sorted by (subject, event)
        extra: DataFrame of relevant supplementary data
            * fcon, scon: mean motion
    """
    # read tabulated data
    if data_type not in INPUTS:
        raise ValueError('Unknown dataset type ' + data_type)

    raw_data = pd.read_csv(PATH / INPUTS[data_type], sep='\t',
                           skiprows=[1], index_col=INDEX)
    if path is not None:
        data = pd.read_csv(path, sep=None, engine='python',
                           index_col=INDEX)
    else:
        data = raw_data

    # columns
    if data_type == 'fcon':
        fcon_codes = list(FCON.keys())
        if exclude_n:
            fcon_codes.remove('n')

        columns = []
        for i in range(len(fcon_codes)):
            for j in range(i+1):
                columns.append(FCON_TEMPLATE.format(fcon_codes[i], fcon_codes[j]))

        extra_columns = {'rsfmri_c_ngd_meanmotion': 'meanmotion'}
    elif data_type == 'scon':
        columns = data.columns.str.startswith(SCON_TEMPLATE.format(''))

        extra_columns = {'dmri_dti_meanmotion': 'meanmotion'}
    else:
        raise ValueError('Cannot load ' + data_type)

    data_cols = data.loc[:, columns]

    # rows
    if include_rec:
        imgincl = pd.read_csv(PATH / INPUTS['imgincl'], sep='\t',
                              skiprows=[1], index_col=INDEX)
        imgincl = imgincl.dropna(subset=['visit'])
        # NOTE has identical duplicate rows for whatever reason
        imgincl = imgincl.loc[~imgincl.index.duplicated(keep='last')]

        if data_type == 'fcon':
            inclusion = imgincl.loc[imgincl['imgincl_rsfmri_include'] == 1].index
        elif data_type == 'scon':
            inclusion = imgincl.loc[imgincl['imgincl_dmri_include'] == 1].index
        else:
            raise ValueError('Cannot load ' + data_type)

        included = data_cols.loc[data_cols.index.intersection(inclusion), :]
    else:
        included = data_cols

    # longitudinal only dataset
    if dropna:
        included = included.dropna()

    subs_included = included.groupby(level=0).size()
    subs_long = subs_included.index[subs_included == len(EVENTS)]
    dataset = data.loc[idx[subs_long, EVENTS], columns]

    # extra
    mri = pd.read_csv(PATH / INPUTS['mri'], sep='\t',
                      skiprows=[1], index_col=INDEX)
    # NOTE has empty duplicate rows for whatever reason
    mri = mri.dropna(how='all', subset=SCAN_INFO)

    extra = (raw_data.loc[dataset.index, extra_columns.keys()]
             .rename(columns=extra_columns)
             .join(mri[SCAN_INFO]))

    return dataset, extra


def get_scon_dict():
    """
    Builds and returns a dict of DTI atlas tract descriptions.

    Returns:
        scon_dict: dict with (code, description) for each tract
    """
    dti_labels = pd.read_csv(PATH / INPUTS['scon'], sep='\t', nrows=1)
    code_start = SCON_TEMPLATE.format('')
    description_starts = ('Average fractional anisotropy within ', 'DTI atlas tract ')

    scon_labels = dti_labels.loc[0, dti_labels.columns.str.startswith(code_start)]
    codes = scon_labels.index.str.replace(code_start, '')
    descriptions = (scon_labels.str.replace(description_starts[0], '')
                    .str.replace(description_starts[1], '').values)

    scon_dict = dict(zip(codes, descriptions))
    return scon_dict


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
    covariates = pd.read_csv(OUT_PATH / 'abcd_covariates.csv', index_col=INDEX)
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
