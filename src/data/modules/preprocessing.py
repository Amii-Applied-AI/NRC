"""
Authors: Nicolas Raymond
         Fatima Davelouis

Description: Stores all the functions related to data preprocessing.
"""
import sys

from numpy import array, median
from os.path import abspath, join, pardir
from pandas import DataFrame ,merge, read_csv
from torch import Tensor

# Project imports
sys.path.append(abspath(join(__file__, *[pardir]*4)))
from settings.paths import RAW_DATA
from src.data.modules.constants import Maps, Lists
from src.data.modules.constants import Keys as k


def get_reverse_complement(seq: str) -> str:
    """
    Reverses the order of the characters and switches A <-> T and C <-> G.
    Other characters are left untouched.
    
    Args:
        seq (str): sequence of nucleotides

    Returns:
        str: reverse complement of the sequence
    """
    return seq[::-1].translate(str.maketrans('ACGT', 'TGCA'))

def nuc_to_onehot(seq: str) -> Tensor:
    """
    Convert each nucleotide of a sequence to a one-hot representation.
    
    Args:
        seq (str): sequence of nucleotides

    Returns:
        Tensor: sequence represented using one-hot encodings (SEQ LENGTH, 5)
    """
    return Tensor([Maps.NUC2ONEHOT.get(nucleotide, [0, 0, 0, 0, 1]) for nucleotide in seq])


def nuc_to_proba(seq: str) -> Tensor:
    """
    Convert each nucleotide of a sequence to a soft encoded representation.
    
    Args:
        seq (str): sequence of nucleotides

    Returns:
        Tensor: sequence represented using soft encodings (SEQ LENGTH, 4)
    """
    return Tensor([Maps.NUC2PROBA.get(nucleotide, [.25, .25, .25, .25]) for nucleotide in seq])


def filter_seq(seq: str) -> str:
    """
    Removes the 'N' and the 'X' from a sequence of nucleotides.
    
    Args:
        seq (str): sequence of nucleotides.

    Returns:
        str: same sequence without any 'N' and 'X'.
    """
    return seq.translate(str.maketrans('', '', 'NX'))


def seq_to_kmer(seq: str,
                k: int = 6,
                stride: int = 1) -> list[str]:
    """
    Takes a sequence of nucleotides and generate a list of k-mers.
    To avoid the inclusion of an incomplete k-mer, the iterative creation process stops
    before stride*i + k > len(seq).
    
    Args:
        seq (str): sequence of nucleotides.
        k (int, optional): length of k-mers.
                           Default to 6.
        stride (int, optional): step size in between each k-mers. 
                                If stride < k, than two contiguous k-mers will share
                                (k - stride) nucleotides. 
                                Default to 1.
    Returns:
        list[str]: list of k-mers of length floor((len(seq) - k)/stride + 1)
    """
    return [seq[i:i+k] for i in range(0, len(seq) - k + 1, stride)]


def split_kmers(kmers: list[str],
                max_length: int = 512,
                overlap: int = 256) -> list[list[int]]:
    """
    Extract subsequences of kmers from a list of kmers.
    IMPORTANT DETAIL: kmers[i:i+max_length] == kmers[i:min(len(kmers), i+max_length)]
    
    Args:
        kmers (list[str]): list of kmers.
        max_length (int, optional): maximum number of kmers included within each subsequence.
                                    Defaults to 512.
        overlap (int, optional): number of kmers shared by two contiguous subsequence. 
                                 Defaults to 256.

    Returns:
        list[list[int]]: list of list of kmers
    """
    return [kmers[i:i+max_length] for i in range(0, len(kmers), max_length - overlap)]


def add_padding(subsequences: list[list[int]],
                padding_token_id: int,
                max_length: int = 512,
                padding_option: str = 'right') -> list[list[int]]:
    """
    Adds padding to subsequences of tokens of length < max_length.
    
    Args:
        tokens_subseqs (list[list[int]]): list containing list of tokens.
        padding_token_id (int): id associated to the padding token
        max_length (int, optional): maximum number of tokens included within each subsequence.
                                    Default to 512.
        padding_option (str, optional): 'right', 'left' or 'both'.
                                        Default to 'right'.

    Returns:
        list[list[int]]: list containing list of tokens.
    """
    for i, subseq in enumerate(subsequences):

        # If padding needs to be added
        subseq_length = len(subseq)

        if subseq_length < max_length:

            # Check validity of padding option
            if padding_option not in ['right', 'left', 'both']:
                return ValueError("padding_option must be either 'right', 'left' of 'both'")

            elif padding_option == 'both':

                # Generate the padding
                padding = [padding_token_id]*((max_length - subseq_length)//2)

                # If the current the length of the current subseq is even
                if subseq_length % 2 == 0:

                    # Add equal amount of padding on left and right
                    subsequences[i] = padding + subseq + padding

                else:

                    # Add more padding on the right than the left
                    subsequences[i] = padding + subseq + padding + [padding_token_id]

            else:

                # Generate the padding
                padding = [padding_token_id]*(max_length - subseq_length)

                if padding_option == 'left':

                    # Add padding on left
                    subsequences[i] = padding + subseq

                else:

                    # Add padding on right
                    subsequences[i] = subseq + padding

    return subsequences


def get_unique_genes(rna_path: str,
                     pro_path: str) -> dict[str, dict]:
    """
    Extract rna-seq and pro-seq data for each unique gene.
    
    Args:
        rna_path (str): path to folder with rna-seq data
        pro_path (str): path to folder with pro-seq data
        
    Returns:
        dict: dictionary with unique genes from each species
    """
    # Initialize a dictionary that will contain genes of each species
    unique = {s: {} for s in Maps.SHORTCUT2SPECIE}

    for pair in Maps.PAIR2ID:

        # Load the tsv file containing the rna-seq data
        rna_df = read_csv(join(rna_path, f'{pair[0]}_{pair[1]}.std.tsv'), sep='\t')
        rna_df.rename(columns={f'{k.RANK}_1': f'{k.RNA_RANK}_1',
                               f'{k.RANK}_2': f'{k.RNA_RANK}_2',
                               f'{k.MEAN}_1': f'{k.RNA_MEAN}_1',
                               f'{k.MEAN}_2': f'{k.RNA_MEAN}_2'},
                      inplace=True)

        # Load the tsv file containing the pro-seq data
        pro_df = read_csv(join(pro_path, f'{pair[0]}_{pair[1]}.std.tsv'), sep='\t')
        pro_df.rename(columns={f'{k.RANK}_1': f'{k.PRO_RANK}_1',
                               f'{k.RANK}_2': f'{k.PRO_RANK}_2',
                               f'{k.MEAN}_1': f'{k.PRO_MEAN}_1',
                               f'{k.MEAN}_2': f'{k.PRO_MEAN}_2'},
                      inplace=True)

        # Merge the dataframes
        all_df = merge(rna_df, pro_df, how='inner', on=[f'{k.GENE}_1', f'{k.GENE}_2'])
        del rna_df
        del pro_df

        for j in range(2):

            # Extract info
            genes = all_df[f'{k.GENE}_{j+1}'].tolist()
            rna_ranks = all_df[f'{k.RNA_RANK}_{j+1}'].tolist()
            rna_means = all_df[f'{k.RNA_MEAN}_{j+1}'].tolist()
            pro_ranks = all_df[f'{k.PRO_RANK}_{j+1}'].tolist()
            pro_means = all_df[f'{k.PRO_MEAN}_{j+1}'].tolist()

            # Add info of gene that were not already encountered
            for i, gene in enumerate(genes):
                if gene not in unique[pair[j]]:
                    unique[pair[j]][gene] = {k.RNA_MEAN: rna_means[i],
                                             k.RNA_RANK: rna_ranks[i],
                                             k.PRO_MEAN: pro_means[i],
                                             k.PRO_RANK: pro_ranks[i]}
    return unique


def normalize_df(df: DataFrame,
                 rna: bool,
                 ranks: bool,
                 pair_of_species: tuple[str, str],
                 unique_genes_dict: dict[str, dict] = None) -> DataFrame:
    """
    Divide the columns of the dataframe containing read means (or rank values).
    
    If rank is True, the values are divided by the number of rows.
    Otherwise, if we are considering read means, we divide by the 
    median and apply log2 scaling.
    The median is either:
    - The median of the non-zero means;
    - The median of ratios (DESeq2).

    Args:
        df (DataFrame): dataframe with read means or ranks.
        rna (bool): if True, indicates that the dataframe contains rna-seq data.
        ranks (bool): if True, indicates that the values are ranks.
        pair_of_species (tuple[str, str]): ordered pair of species contained in the dataframe.
        unique_genes_dict (optional, dict[str, dict]): dictionary with information on
                                                       unique genes from each species.
                                                       Default to None.
    Returns:
        DataFrame: same dataframe, but with normalized column
    """
    if ranks:

        # Divide by the number of rows
        n = len(df)
        df[f'{k.RANK}_1'] = df[f'{k.RANK}_1']/n
        df[f'{k.RANK}_2'] = df[f'{k.RANK}_2']/n

    else:

        if unique_genes_dict is None:
            raise ValueError('unique_genes_dict must be provided if ranks is False')

        # Set the sequencing type
        sequencing_type = 'rna' if rna else 'pro'

        for i , s in enumerate(pair_of_species):

            # # Divide by median of specie
            # means = df[f'{k.MEAN}_{i+1}'].to_numpy()
            # means /= unique_genes_dict[s][f'{sequencing_type}_{k.MEDIAN}']
            #
            # # Add 2 and apply log2 scaling
            # df[f'{k.MEAN}_{i+1}'] = log2(means + 2).tolist()
            pass

    return df


def categorize(x: float,
               quantiles: array) -> int:
    """
    Assign a category to an expression difference
    according to its comparison to given quantiles.
    
    Args:
        x (float): rank difference
        quantiles (list[float]): list of quantiles, ORDERED FROM HIGHEST TO LOWEST.

    Returns:
        int: category
    """
    for i, q in enumerate(quantiles):
        if x >= q:
            return i

    return len(quantiles)


def get_median_of_ratios(unique_genes_dict: dict[str, dict],
                         rna: bool,
                         epsilon: float = 5e-4) -> dict[str, dict[str, float]]:
    """
    Calculate median of means ratio across specie.

    Args:
        unique_genes_dict (dict[str, dict]): dictionary with information on
                                             unique genes from each species.
        rna (bool): if True, indicates that ratios are calculated
                    using rna-seq data.
        epsilon (float): small value added to read means to avoid
                         division by zero.

    Returns:
        dict[str, dict[str, float]]: dictionary with rna-seq median and pro-seq
                                     median for each specie.
    """

    # Load csv with intersecting orthologs
    df = read_csv(join(RAW_DATA, '_'.join(Lists.SPECIES), 'complete_orthogroups.csv'))

    # Save appropriate key
    key = k.RNA_MEAN if rna else k.PRO_MEAN

    # Save the numebr of species
    nb_species = len(Lists.SPECIES)

    # Match each genes to their mean read value
    means = []
    for i in range(len(df)):
        row = []
        for c in df.columns:
            gene_info = unique_genes_dict[c].get(df.iloc[i][c])
            if gene_info is not None:
                row.append(gene_info[key])
            else:
                break
        if len(row) == nb_species:
            means.append(row)

    means = array(means)

    # Add a small value to avoid any division by 0
    means += epsilon

    # Divide all columns by the first column
    ratios = means / means[:, 0].reshape(-1, 1)
    del means

    # Calculate median of each column
    medians = median(ratios, axis=0).tolist()
    del ratios

    # Save median in dictionary
    return {c: medians[i] for i, c in enumerate(df.columns)}
