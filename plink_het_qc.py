#!/usr/bin/env python
# coding: utf-8
"""
Read PLINK het file, remove outlier individuals, and save .fam
"""
import argparse
from pathlib import Path
import pandas as pd
from scipy import stats

p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
p.add_argument('het', type=Path,
               help='PLINK het file')
p.add_argument('fam', type=Path,
               help='output fam')
p.add_argument('--sd', type=float, default=3,
               help='outlier standard deviations')
args = p.parse_args()


def save_fam(sample, path):
    fam = sample.reset_index()[['FID', 'IID']]
    fam.to_csv(path, sep='\t', index=False, header=False)


het = pd.read_table(args.het, delim_whitespace=True)
het['Fz'] = stats.zscore(het['F'])
valid = het.loc[het['Fz'].abs() <= args.sd]
save_fam(valid, args.fam)
