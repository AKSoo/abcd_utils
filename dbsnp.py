from pathlib import Path
from pysam import VariantFile

PATH = Path('inputs/dbSNP')
BUILDS = {
    'GRCh37': 'GRCh37p13/00-All.vcf.gz',
    'hg19': 'GRCh37p13/00-All.vcf.gz',
    'GRCh38': 'GRCh38p7/00-All.vcf.gz',
    'hg38': 'GRCh38p7/00-All.vcf.gz'
}


def load_snps(build='hg38'):
    """
    Load a dbSNP variation set.
    
    Params:
        build: a human genome reference build name

    Returns:
        snps: pysam.VariantFile
    """
    if build not in BUILDS:
        raise ValueError(f'{build} is not a valid human reference genome build.')

    snps = VariantFile(PATH / BUILDS[build], 'r')
    return snps


def _allele_match(ref, alts, a1, a2, max_allele=None):
    if alts is None:
        alts = ()

    if max_allele is not None:
        ref = ref[:max_allele]
        alts = [a[:max_allele] for a in alts]
        a1 = a1[:max_allele]
        a2 = a2[:max_allele]

    return a1 == ref and a2 in alts

def find_rs(ch, bp, a1, a2, snps=None,
            max_allele=None, topmed=False):
    """
    Queries a dbSNP variation set for SNPs by chromosome location and alleles.
    Tries best to return exactly one.

    Params:
        ch: int or str, chromosome
        bp: int, base pair coordinate
        a1: str, allele 1 (major)
        a2: str, allele 2 (minor)
        snps: pysam.VariantFile; If None, load the default.
        topmed: Favor SNPs in TOPMED reference if multiple matches.
        max_allele: Number of allele bp's to match.
            Falls back to exact match if multiple matches.

    Returns:
        rs_ids: list of matched SNPs' rs IDs
    """
    if snps is None:
        snps = load_snps()

    matches = []

    # add rough matches
    for rec in snps.fetch(str(ch), bp-1, bp):
        if _allele_match(rec.ref, rec.alts, a1, a2, max_allele):
            matches.append(rec)

    # if multiple matches, try more exact
    if len(matches) > 1 and topmed:
        rematch = [rec for rec in matches
                   if 'TOPMED' in rec.info.keys()]
        if len(rematch) > 0:
            matches = rematch

    if len(matches) > 1 and max_allele is not None:
        rematch = [rec for rec in matches
                   if _allele_match(rec.ref, rec.alts, a1, a2)]
        if len(rematch) > 0:
            matches = rematch

    # report bad matches if verbose
    rs_ids = [rec.id for rec in matches]

    return rs_ids
