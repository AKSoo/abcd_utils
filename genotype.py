from pathlib import Path
import pandas as pd

GWAS_DIR = Path('inputs/GWAS')
PRS_DIR = Path('outputs/prs')

BIM_COLS = ['CHR', 'SNP', 'POS', 'BP', 'A1', 'A2']


def reference_ids(genes, reference, swap=False):
    """
    Get SNP IDs from a reference for given genes. No matches are None.
    Matches
        CHR: chromosome (no X), BP: base pair
        A1: minor allele, A2: major allele

    Params:
        genes: pd.DataFrame with columns CHR, BP, A1, A2, SNP
        reference: pd.DataFrame with columns CHR, BP, A1, A2, SNP
        swap: bool, try swapping A1 and A2 for match.
            rs IDs for swapped matches are reversed ('...sr').

    Returns:
        ids: pd.Series of str
    """
    reference = (reference.dropna().astype({'CHR': int, 'BP': int})
                 .set_index(['CHR', 'BP']).sort_index())

    def query_id(row):
        if (row['CHR'], row['BP']) in reference.index:
            match = reference.loc[(row['CHR'], row['BP'])]
            if match['A1'] == row['A1'] and match['A2'] == row['A2']:
                return match['SNP']
            if swap and match['A1'] == row['A2'] and match['A2'] == row['A1']:
                return match['SNP'][::-1]
        return None

    ids = genes.apply(query_id, axis=1)
    return ids


def subs_index(index):
    return index.str.split('_', n=1).str[1]

def load_prscores(pheno):
    """
    Load polygenic risk scores.

    Params:
        pheno: str, phenotype ID

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
