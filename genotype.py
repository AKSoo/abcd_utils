from pathlib import Path
import pandas as pd

PRS_DIR = Path('outputs/prs')

PHENOTYPES = {
    'ASD': {'n': 46350}
}

def subs_index(index):
    return index.str.split('_', n=1).str[1]


def load_prscores(pheno):
    """
    Load polygenic risk scores.

    Params:
        pheno: A key in PHENOTYPES

    Returns:
        scores: DataFrame indexed by subject
    """
    scores_dir = PRS_DIR / pheno / 'scores'
    cols_dict = {'#IID': None, 'SCORE1_SUM': pheno}

    scores = 0
    for score_path in scores_dir.glob('*.sscore'):
        scores += pd.read_table(score_path, index_col='#IID',
                                usecols=cols_dict.keys())

    scores.index = subs_index(scores.index).rename(cols_dict['#IID'])
    scores = scores.rename(columns=cols_dict)
    return scores
