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


def _allele_match(ref, alts, a1, a2, len_match=None):
    if alts is None:
        alts = ()

    if len_match is None:
        return a1 == ref and a2 in alts

    a1, a2 = a1[:len_match], a2[:len_match]
    if ref.startswith(a1):
        for alt in alts:
            if alt.startswith(a2):
                return True
    return False

def find_rs(ch, bp, a1, a2, snps=None,
            swap=False, tags=None, len_match=None):
    """
    Queries a dbSNP variation set for SNPs by chromosome location and alleles.
    Tries best to return exactly one.

    Params:
        ch: int or str, chromosome
        bp: int, base pair coordinate
        a1: str, allele 1 (major)
        a2: str, allele 2 (minor)
        snps: pysam.VariantFile; If None, load the default.
        swap: bool, try swapping a1 and a2 if no match.
            rs IDs for swapped matches are reversed ('...sr').
        tags: str's, multiple matches filter tags.
            If set, tags must be a subset.
            If list, each tag is filtered in order.
        len_match: int, length of long alleles. Long alleles are compared by
            len_match. If multiple matches, fall back to exact match.

    Returns:
        rs_ids: list of matched SNPs' rs IDs
    """
    if snps is None:
        snps = load_snps()
    if len(a1) < len_match and len(a2) < len_match:
        len_match = None

    matches = []

    # add rough matches
    for rec in snps.fetch(str(ch), bp-1, bp):
        if _allele_match(rec.ref, rec.alts, a1, a2, len_match):
            matches.append(rec)

    # if no match, try swapped
    swapped = False
    if len(matches) == 0 and swap:
        swapped = True
        for rec in snps.fetch(str(ch), bp-1, bp):
            if _allele_match(rec.ref, rec.alts, a2, a1, len_match):
                matches.append(rec)

    # if multiple matches, try exact and filter
    if len(matches) > 1 and len_match is not None:
        rematch = [rec for rec in matches
                   if _allele_match(rec.ref, rec.alts, a1, a2)]
        if len(rematch) > 0:
            matches = rematch

    if len(matches) > 1 and tags is not None:
        if isinstance(tags, set):
            rematch = [rec for rec in matches
                       if tags <= set(rec.info.keys())]
            if len(rematch) > 0:
                matches = rematch
        elif isinstance(tags, list):
            for tag in tags:
                rematch = [rec for rec in matches
                           if tag in rec.info.keys()]
                if len(rematch) > 0:
                    matches = rematch
                if len(matches) == 1:
                    break
        else:
            raise ValueError('tags not list or set:', type(tags))

    # reversed IDs if swapped
    if swapped:
        rs_ids = [rec.id[::-1] for rec in matches]
    else:
        rs_ids = [rec.id for rec in matches]
    return rs_ids
