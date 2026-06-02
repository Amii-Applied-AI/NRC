"""
Authors: Nicolas Raymond
         
         
Description: Creates training, validation and test sets to proceed to a 
             5-fold cross validation with a contrast model. Splits are
             created based on orthogroups to avoid any gene showing up in
             different sets of a split at the same time.
             
"""
import sys
import numpy as np
from argparse import ArgumentParser
from Bio import SeqIO
from matplotlib import pyplot as plt
from os import makedirs, mkdir
from os.path import abspath, join, pardir
from pandas import concat, DataFrame, merge, read_csv
from pickle import dump
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LinearRegression
from torch import stack, Tensor

# Project imports
sys.path.append(abspath(join(__file__, *[pardir]*4)))
from settings.paths import INTERIM_DATA, PFMG_DATA, RAW_DATA
from src.data.modules.dataset import SeqDataset
from src.data.modules.constants import Lists, Maps
from src.data.modules.constants import Keys as k
from src.data.modules.preprocessing import get_unique_genes, nuc_to_proba
from src.utils.reproducibility import SEED
from src.utils.nucleotide import align_sequences
from tqdm import tqdm


# Definition of the arguments parsing function
def get_args():
    """
    Retrieves the settings given in the terminal to run the script.
    """
    parser = ArgumentParser(usage='python process_by_orthogroups.py',
                            description='Proceed to pea-faba-medicago-grasspea data processing.')

    # Plot
    parser.add_argument('-plot', '--plot', default=False, action='store_true',
                        help='If provided, pie charts are saved to visualize ' +
                        'the stratification of the splits. Default to False.')

    # Normalization method
    parser.add_argument('-norm', '--normalization', type=str, default=None,
                        choices=[k.MEDIAN, k.DESEQ, None],
                        help='Choice of normalization method. Default to None.')

    # Synteny level
    parser.add_argument('-s', '--synteny', type=int, default=0, choices=[0, 5],
                        help='Synteny level. Default to 0.')

    # Pre-align the sequences to speed up training
    parser.add_argument('-pre_align', '--pre_align', default=False, action='store_true',
                        help='Pre-align the sequences to speed up training')

    arguments, _ = parser.parse_known_args()

    return arguments


# Definition of a function to compute ajusted effect size
def adjusted_effect_size(mean1: float,
                         mean2: float,
                         std1: float,
                         std2: float) -> float:
    """
    Computes the ajusted effect size betweem two distributions
    using their mean and standard deviations.

    Args:
        mean1 (float): mean of 1st distribution.
        mean2 (float): mean of 2nd distribution.
        std1 (float): standard deviation of 1st distribution.
        std2 (float): standard deviation of 2nd distribution.

    Returns:
        float: _description_
    """
    if std1 > 0 and std2 > 0:
        return (mean1 - mean2) / (np.sqrt((np.square(std1) + np.square(std2)) / 2))

    if std1 > 0 and std2 == 0:
        return (mean1 - mean2) / std1

    if std2 > 0 and std1 == 0:
        return (mean1 - mean2) / std2

    return np.nan


# Execution of the script
if __name__ == '__main__':

    # Retrieve the arguments that dictate the script procedure
    args = get_args()

    # Save the path leading to the future processed data
    postfix = f'synteny_{args.synteny}'
    if args.normalization is not None:
        postfix = f'{postfix}_{args.normalization}'

    PROCESSED_DATA: str = join(PFMG_DATA, postfix)

    # Create new folders to store the data and the figures
    makedirs(PFMG_DATA, exist_ok=True)
    mkdir(PROCESSED_DATA)
    if args.plot:
        fig_folder = join(PROCESSED_DATA, 'figures')
        mkdir(fig_folder)

    # Save variables with paths to raw data folders
    SPECIES = '_'.join(Lists.SPECIES)
    seq_folder = join(RAW_DATA, SPECIES, 'seq_file', f'synteny_{args.synteny}')
    rna_folder = join(RAW_DATA, SPECIES, 'quantseq')
    pro_folder = join(RAW_DATA, SPECIES, 'proseq')

    # Extract information of unique genes per species
    print('Extracting genes information..')
    unique_genes = get_unique_genes(rna_path=rna_folder, pro_path=pro_folder)

    # Load ortholog pairs
    print('Loading ortholog pairs..')
    df = []
    for (s0, s1), pair_id in Maps.PAIR2ID.items():

        # Save a string that will be used to load files
        file_prefix = f'{s0}_{s1}'

        # Load the fasta file
        filename = f'{Maps.SHORTCUT2SPECIE[s0]}_to_{Maps.SHORTCUT2SPECIE[s1]}.modelseq.fasta'
        ortho_data = SeqIO.parse(join(seq_folder, filename), 'fasta')

        # Extract the important data from the fasta file
        print(f'\n{file_prefix}')
        temp_df = DataFrame([(o.id, o.seq) for o in ortho_data], columns=[k.GENE_IDS, k.SEQ])
        print(f'Nb pairs in fasta = {len(temp_df)}')

        # Load the tsv file containing the expression levels (Quantseq)
        rna_df = read_csv(join(rna_folder, f'{file_prefix}.std.tsv'), sep='\t')
        print(f'Nb pairs in quantset file = {len(rna_df)}')

        # QQ
        # Fix estimated standard deviations
        lm = LinearRegression().fit(rna_df['mean_1'].to_numpy().reshape(-1, 1), rna_df['std_1'].to_numpy())
        assert lm.coef_ > 0.
        assert lm.intercept_ > 0.
        rna_df['expected_std_1'] = (rna_df['mean_1'] * lm.coef_[0]) + lm.intercept_

        lm = LinearRegression().fit(rna_df['mean_2'].to_numpy().reshape(-1, 1), rna_df['std_2'].to_numpy())
        assert lm.coef_ > 0.
        assert lm.intercept_ > 0.
        rna_df['expected_std_2'] = (rna_df['mean_2'] * lm.coef_[0]) + lm.intercept_

        # Create a new column with identifiers to match with the data from the fasta
        rna_df[k.GENE_IDS] = rna_df[f'{k.GENE}_1'] + '___' + rna_df[f'{k.GENE}_2']

        # Create new columns to remember species in which genes belong
        rna_df['S0'] = s0
        rna_df['S1'] = s1
        rna_df[k.SPECIES] = pair_id

        # Rename some of the columns
        rna_df.rename(columns={f'{k.MEAN}_1': f'{k.RNA_MEAN}_0',
                               f'{k.MEAN}_2': f'{k.RNA_MEAN}_1',
                               f'{k.STD}_1': f'{k.RNA_STD}_0',
                               f'{k.STD}_2': f'{k.RNA_STD}_1'}, inplace=True)

        # Remove rows which both genes' standard deviation is equal to 0 for QuantSeq data
        rna_df = rna_df[(rna_df[f'rna_{k.STD}_0'] != 0) & (rna_df[f'rna_{k.STD}_1'] != 0)]

        # Load the tsv file containing the transcription rates (PRO-seq)
        pro_df = read_csv(join(pro_folder, f'{file_prefix}.std.tsv'), sep='\t')
        print(f'Nb pairs in proseq file = {len(pro_df)}')

        # QQ
        # Fix estimated standard deviations
        lm = LinearRegression().fit(pro_df['mean_1'].to_numpy().reshape(-1, 1), pro_df['std_1'].to_numpy())
        assert lm.coef_ > 0.
        assert lm.intercept_ > 0.
        pro_df['expected_std_1'] = (pro_df['mean_1'] * lm.coef_[0]) + lm.intercept_

        lm = LinearRegression().fit(pro_df['mean_2'].to_numpy().reshape(-1, 1), pro_df['std_2'].to_numpy())
        assert lm.coef_ > 0.
        assert lm.intercept_ > 0.
        pro_df['expected_std_2'] = (pro_df['mean_2'] * lm.coef_[0]) + lm.intercept_

        # Create a new column with identifiers to match with the data from the fasta
        pro_df[k.GENE_IDS] = pro_df[f'{k.GENE}_1'] + '___' + pro_df[f'{k.GENE}_2']

        # Rename some of the columns
        pro_df.rename(columns={f'{k.MEAN}_1': f'{k.PRO_MEAN}_0',
                               f'{k.MEAN}_2': f'{k.PRO_MEAN}_1',
                               f'{k.STD}_1': f'{k.PRO_STD}_0',
                               f'{k.STD}_2': f'{k.PRO_STD}_1'}, inplace=True)

        # Remove rows which both genes' standard deviation is equal to 0 for ProSeq data
        pro_df = pro_df[(pro_df[f'pro_{k.STD}_0'] != 0) & (pro_df[f'pro_{k.STD}_1'] != 0)]

        # Concatenate both dataframes
        temp_df = merge(rna_df, temp_df, how='inner', on=k.GENE_IDS)
        temp_df = merge(temp_df, pro_df, how='inner', on=k.GENE_IDS)
        print(f'Nb match = {len(temp_df)}\n')

        # Filter columns
        temp_df = temp_df[[k.GENE_IDS,
                           'S0', 'S1',
                           f'{k.RNA_MEAN}_0', f'{k.RNA_MEAN}_1',
                           f'{k.RNA_STD}_0', f'{k.RNA_STD}_1',
                           f'{k.PRO_MEAN}_0', f'{k.PRO_MEAN}_1',
                           f'{k.PRO_STD}_0', f'{k.PRO_STD}_1',
                           k.SPECIES,
                           k.SEQ]]

        # Add the dataframe to the list
        df.append(temp_df)

    # Concatenate the dataframes
    df = concat(df)

    # Load orthogroups and remove duplicated rows
    ortho_df = read_csv(join(RAW_DATA, SPECIES, 'orthogroups.csv'), index_col=0)
    nb_before_filtering = len(ortho_df)
    ortho_df = ortho_df[~ortho_df.index.duplicated(keep='first')]
    print(f'Number of duplicated orthogroups removed: {nb_before_filtering - len(ortho_df)}')
    orthogroups = ortho_df.index.to_list()
    group_labels = ortho_df['composition_label'].to_numpy()

    # Save the name of the genes in the pairs
    pairs = df[k.GENE_IDS].to_list()

    # Save species
    species = Tensor(df[k.SPECIES].values)

    # Generate input tensors for the contrast models
    print('\nEncoding nucleotide sequences...')
    inputs = stack([nuc_to_proba(seq) for seq in df[k.SEQ].to_list()]).permute(0, 2, 1)

    # Initialization of the stratified k-fold object (5 splits => 20% test)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Generate splits for k-fold cross validation based on orthogroups
    print('Proceeding to data splits creation...')
    for i, (remaining_idx, test_group_idx) in enumerate(skf.split(orthogroups, group_labels)):

        # Separate train and valid orthogroups
        train_group_idx, valid_group_idx = train_test_split(remaining_idx,
                                                            test_size=0.15,
                                                            stratify=group_labels[remaining_idx],
                                                            random_state=SEED)

        # Create a list of boolean indicating if a pair
        # of orthologs was already assigned to a split
        selected = [False]*len(pairs)

        # Initialize a dictionary to store normalization constants
        # for each specie and each type of data
        if args.normalization is not None:
            quant_norm_cst = {s: [] for s in Lists.SPECIES}
            pro_norm_cst = {s: [] for s in Lists.SPECIES}

        # Find the indexes of the ortholog pairs associated
        # with the groups contained in each split
        for split, name in [(train_group_idx, 'train'),
                            (valid_group_idx, 'valid'),
                            (test_group_idx, 'test')]:

            # Initialize a list to store the indexes of the
            # ortholog pairs with a gene in any of the groups
            # associated to a split
            pair_idx = []

            # Extract the genes of all the orthogroups in a split
            genes = set() # Using a set reduces execution time
            for group_idx in split:

                # Extract the name of the group
                group_name = orthogroups[group_idx]

                # Extract the genes in this group
                genes.update([g for g in ortho_df.loc[group_name].to_list()[:-2]
                              if isinstance(g, str)])

            # Identify pairs with a gene among the list of genes just created
            for j, pair in enumerate(pairs):

                # Check if the pair was already assigned
                if not selected[j]:

                    # Extract the genes
                    gene_a, gene_b = tuple(pair.split('___'))

                    # Check if any gene is part of the orthogroup
                    # and add its index if it's the case. Note that if
                    # a gene is part of an orthogroup, the second is necessarily is.
                    if gene_a in genes:
                        pair_idx.append(j)

            # Extract the subet of ortholog pairs linked to the orthogroups
            subset = df.iloc[pair_idx].copy()

            if args.normalization is not None:

                if name == 'train':

                    if args.normalization == k.MEDIAN:

                        # Extract unique read counts and transcript rates
                        # of each gene in the training set
                        # and determine the median for each species
                        for g in genes:

                            # Determine the species of the gene
                            if 'LATSA' in g:
                                s = 'grasspea'
                            elif 'Vfaba' in g:
                                s = 'faba'
                            elif 'Psat' in g:
                                s = 'pea'
                            elif 'Mtrun' in g:
                                s = 'medicago'
                            else:
                                raise ValueError(f'Unkown species: {s}')

                            # Extract the read count (quantseq) and transcript rate (proseq)
                            quant_norm_cst[s].append(unique_genes[s][g][f'{k.RNA_MEAN}'])
                            pro_norm_cst[s].append(unique_genes[s][g][f'{k.PRO_MEAN}'])

                        # Compute medians from non null read counts
                        for s in Lists.SPECIES:

                            # Median on quantseq
                            counts = np.array(quant_norm_cst[s])
                            quant_norm_cst[s] = np.median(counts[counts != 0]).item()

                            # Median on proseq
                            counts = np.array(pro_norm_cst[s])
                            pro_norm_cst[s] = np.median(counts[counts != 0]).item()

                    if args.normalization == k.DESEQ:

                        # Set faba species as a reference
                        # since it is second in all the pairs
                        pro_norm_cst['faba'] = 1
                        quant_norm_cst['faba'] = 1

                        # Create a smaller subset of the dataframe
                        # with pairs of non null Faba counts
                        row_filter = (subset[f'{k.RNA_MEAN}_1'] > 0) & (subset['S1'] == 'faba')
                        faba_pairs = subset[row_filter]

                        # Compute median of ratios for QuantSeq
                        for s in Lists.SPECIES:
                            
                            if s != 'faba':

                                # Extract rows of the small subset associated
                                # to the current species and the reference one (faba)
                                row_filter = (faba_pairs[f'{k.RNA_MEAN}_0'] > 0) & (faba_pairs['S0'] == s)
                                small_df = faba_pairs[row_filter]

                                # Compute median of ratios for QuantSeq
                                ratios = small_df[f'{k.RNA_MEAN}_0'].values/small_df[f'{k.RNA_MEAN}_1'].values
                                quant_norm_cst[s] = np.median(ratios).item()

                        # Create a smaller subset of the dataframe
                        # with pairs of non null Faba transcript rate
                        row_filter = (subset[f'{k.PRO_MEAN}_1'] > 0) & (subset['S1'] == 'faba')
                        faba_pairs = subset[row_filter]

                        # Compute median of ratios for PRO-seq
                        for s in Lists.SPECIES:

                            if s != 'faba':
                                
                                # Extract rows of the small subset associated
                                # to the current species and the reference one (faba)
                                row_filter = (faba_pairs[f'{k.PRO_MEAN}_0'] > 0) & (faba_pairs['S0'] == s)
                                small_df = faba_pairs[row_filter]

                                # Compute median of ratios for PRO-seq
                                ratios = small_df[f'{k.PRO_MEAN}_0'].values/small_df[f'{k.PRO_MEAN}_1'].values
                                pro_norm_cst[s] = np.median(ratios).item()

                # Save normalization constants
                subset['S0_QNORM_CST'] = subset['S0'].map(quant_norm_cst)
                subset['S1_QNORM_CST'] = subset['S1'].map(quant_norm_cst)
                subset['S0_PNORM_CST'] = subset['S0'].map(pro_norm_cst)
                subset['S1_PNORM_CST'] = subset['S1'].map(pro_norm_cst)

                # Proceed to normalization on the subset
                for key in [k.MEAN, k.STD]:
                    subset.loc[:, f'rna_{key}_0'] = subset[f'rna_{key}_0']/subset['S0_QNORM_CST']
                    subset.loc[:, f'rna_{key}_1'] = subset[f'rna_{key}_1']/subset['S1_QNORM_CST']
                    subset.loc[:, f'pro_{key}_0'] = subset[f'pro_{key}_0']/subset['S0_PNORM_CST']
                    subset.loc[:, f'pro_{key}_1'] = subset[f'pro_{key}_1']/subset['S1_PNORM_CST']

            # Compute Cohen's D with quantseq
            statistics = [subset[col].tolist() for col in [f'{k.RNA_MEAN}_0', f'{k.RNA_MEAN}_1',
                                                           f'{k.RNA_STD}_0', f'{k.RNA_STD}_1']]
            statistics = zip(*statistics)
            q_y = Tensor([adjusted_effect_size(m1, m2, std1, std2) for m1, m2, std1, std2 in statistics])

            # Compute Cohen's D with proseq
            statistics = [subset[col].tolist() for col in [f'{k.PRO_MEAN}_0', f'{k.PRO_MEAN}_1',
                                                           f'{k.PRO_STD}_0', f'{k.PRO_STD}_1']]
            statistics = zip(*statistics)
            p_y = Tensor([adjusted_effect_size(m1, m2, std1, std2) for m1, m2, std1, std2 in statistics])

            # Balance the dataset associated to the split
            # The dataset is balanced according to quantseq cohen's D
            x, y, flip_mask = SeqDataset.balance_data(inputs[pair_idx], q_y)

            # If the sequences are to be pre-aligned, we need to do that to x before saving
            if args.pre_align:
                seqs_a = x[:, :, :3000].float().numpy()
                seqs_b = x[:, :, 3030:].float().numpy()

                x = []

                for j in tqdm(range(seqs_a.shape[0])):
                    x.append(Tensor(align_sequences(seqs_a[j, :, :], seqs_b[j, :, :])))

            if args.plot:

                # Save a pie chart illustrating the stratification of the tags
                fig, ax = plt.subplots()
                slice_ids, sizes =  [], []
                for label in set(group_labels.tolist()):
                    sizes.append((group_labels[split] == label).sum().item())
                    slice_id = str(label).translate(str.maketrans('1234', 'GFMP'))
                    slice_ids.append(slice_id.replace('0', ''))
                ax.pie(sizes, labels=slice_ids, autopct='%1.1f%%')
                fig.suptitle(f'{name.capitalize()} ({i}) - Total: {sum(sizes)}')
                fig.savefig(join(fig_folder, f'Fold_{i}_{name}_species.pdf'))
                plt.close()

                # Save a pie chart illustrating the stratification of the class labels
                fig, ax = plt.subplots()
                nb_ones = (y >= 0).sum().item()
                sizes = [y.shape[0] - nb_ones, nb_ones]
                slice_ids = ['0', '1']
                ax.pie(sizes, labels=slice_ids, autopct='%1.1f%%')
                fig.suptitle(f'{name.capitalize()} - Total: {sum(sizes)}')
                fig.savefig(join(fig_folder, f'Fold_{i}_{name}_labels.pdf'))
                plt.close()

            # Save dataset
            with open(join(PROCESSED_DATA, f'Fold_{i}_{name}.pkl'), 'wb') as file:
                dump([Tensor(pair_idx),
                      x,
                      y,
                      p_y,
                      species[pair_idx],
                      flip_mask],
                     file)
