"""
Author: Nicolas Raymond

Description: Selection of constants repeatedly used in the project,
             mainly in the preprocessing scripts.
"""


class Maps():
    """
    Stores maps (i.e., dictionaries) reused multiple
    times in the project.
    """
    # Nucleotide to one-hot encoding
    NUC2ONEHOT: dict[str, list[int]] = {"A": [1, 0, 0, 0, 0],
                                        "C": [0, 1, 0, 0, 0],
                                        "G": [0, 0, 1, 0, 0],
                                        "T": [0, 0, 0, 1, 0],
                                        "N": [0, 0, 0, 0, 1],
                                        "X": [0, 0, 0, 0, 1]}

    # Nucleotide to probability encoding (i.e., soft encoding)
    NUC2PROBA: dict[str, list[int]] = {"A": [1, 0, 0, 0],
                                       "C": [0, 1, 0, 0],
                                       "G": [0, 0, 1, 0],
                                       "T": [0, 0, 0, 1],
                                       "N": [.25, .25, .25, .25],
                                       "X": [.25, .25, .25, .25]}

    # Argmax of one-hot encoding to nucleotide
    ONEHOT2NUC: dict[int, str] = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'NX'}

    # Maps the position of a 1 in a soft-encoding to a nucleotide
    PROBA2NUC: dict[int, str] = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    ALIGN2IDX: dict[int, str] = {pair: i for i, pair in enumerate(['AA', 'A-', '-A',
                                                                   'CC', 'C-', '-C',
                                                                   'GG', 'G-', '-G',
                                                                   'TT', 'T-', '-T'])}


    # Pairs of species to id
    PAIR2ID: dict[tuple[str, str], int] = {('grasspea', 'faba'): 0,
                                           ('grasspea', 'medicago'): 10,
                                           ('grasspea', 'pea'): 20,
                                           ('medicago', 'faba'): 30,
                                           ('medicago', 'pea'): 40,
                                           ('pea', 'faba'): 50}

    # Shortcut of species name to full name
    SHORTCUT2SPECIE: dict[str, str] = {'grasspea': 'Lathyrus_sativus',
                                       'medicago': 'Medicago_truncatula',
                                       'pea': 'Pisum_sativum',
                                       'faba': 'Vicia_faba'}

class Lists():
    """
    Stores constants that takes the form of a lists.
    """
    QUANTILES: list[float] = [0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875]
    SPECIES: str = ['pea', 'faba', 'medicago', 'grasspea']


class Keys:
    """
    Stores strings that are used as column names of dictionary keys.
    """
    EXP_DIFF: str = 'exp_diff'
    EXP_DIFF_CAT: str = 'exp_diff_cat'
    GENE: str = 'gene'
    GENES: str = 'genes'
    GENE_IDS: str = 'gene_ids'
    MEAN: str = 'mean'
    MEDIAN: str = 'median'
    DESEQ: str = 'deseq'
    PAIR_COUNT: str = 'pair_count'
    PRO_MEAN: str = 'pro_mean'
    PRO_MEDIAN: str = 'pro_median'
    PRO_STD: str = 'pro_std'
    PRO_RANK: str = 'pro_rank'
    RANK: str = 'rank'
    RNA_MEAN: str = 'rna_mean'
    RNA_MEDIAN: str = 'rna_median'
    RNA_STD: str = 'rna_std'
    RNA_RANK: str = 'rna_rank'
    SEQ: str = 'seq'
    SINGLE_COUNT: str = 'single_count'
    SPECIES: str = 'species'
    STD: str = 'std'
    TRANSCRIPT_DIFF: str = 'transcript_diff'
