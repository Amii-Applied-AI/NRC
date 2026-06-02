"""
Authors: Nicolas Raymond

Description: Procedure used to generate DeepLift figure.
             In this case, we test a PlanTT-CNN-MLP model 
             trained with data at synteny 0.
"""

import sys
from argparse import ArgumentParser
from captum.attr import DeepLift
from matplotlib import pyplot as plt
from numpy import array
from os.path import abspath, join, pardir
from torch import cat, device, load, ones, zeros
from torch.utils.data import DataLoader

sys.path.append(abspath(join(__file__, *[pardir]*2)))
from settings.paths import PFMG_DATA
from src.data.modules.dataset import SeqDataset
from src.models.cnn import WCNN

DEVICE = device('cuda')

def get_settings():
    """
    Retrieves the settings given in the terminal to run the script.
    """
    parser = ArgumentParser(usage='python deeplift.py',
                            description='Analysis of the average attribution scores ' +
                            'of PlanTT-CNN-MLP with DeepLift.')

    parser.add_argument('-weights_path', '--weights_path', type=str, required=True,
                        help='Path to trained model weights (i.e. records file with saved models per fold)')

    parser.add_argument('-proseq', '--proseq', action='store_true', help='Use the proseq data')

    # DeepLift reference
    parser.add_argument('-ref', '--reference', type=str, default='zeros',
                        choices=['zeros', 'uniform'],
                        help='Type of reference used to calculate DeepLift attribution scores.')

    # Moving average window size
    parser.add_argument('-ws', '--window_size', type=int, default=1,
                        help='Window size used to calculate moving average on attribution scores.')

    return parser.parse_args()

# Execution of the script
if __name__ == '__main__':

    # Retrieve environment settings
    settings = get_settings()

    # 1. Calculate the attribution scores on the five test sets
    # Initialize a list that will contain tensors with attribution scores
    attr_scores = []

    if settings.reference == 'uniform':
        # Set a baseline sequence (a sequence full of 'X' characters)
        baseline = (ones(1, 4, 3000, requires_grad=True)*0.25).to(DEVICE)
    else:
        # Set a baseline sequence full of zeros
        baseline = zeros(1, 4, 3000, requires_grad=True).to(DEVICE)

    for i in range(5):
        model = WCNN(regression=True, encoding_size=4)
        model.load_state_dict(load(join(settings.weights_path, f'fold_{i}.pt')))

        model.eval()

        # Load the data and create a dataloader
        test_set = SeqDataset.from_path(path=join(PFMG_DATA, 'synteny_0', f'Fold_{i}_test.pkl'),
                                        regression=True,
                                        include_flip=False,
                                        include_reverse_complement=False,
                                        proseq=settings.proseq)

        dl = DataLoader(test_set, batch_size=128, shuffle=False)

        # Create the DeepLift object
        deep_lift = DeepLift(model=model.to(DEVICE))

        # Get the attribution scores of each batch
        batch_attr_scores = []
        for seq_a, seq_b, _, _ in dl:
            seq_a, seq_b = seq_a.to(DEVICE), seq_b.to(DEVICE)
            seq_a.requires_grad = True
            seq_b.requires_grad = True
            attr_1, attr_2 = deep_lift.attribute(inputs=(seq_a, seq_b),
                                                 baselines=(baseline, baseline))
            batch_attr_scores.append(attr_1.squeeze().detach().to('cpu'))
            batch_attr_scores.append(attr_2.squeeze().detach().to('cpu'))

        batch_attr_scores = cat(batch_attr_scores)
        attr_scores.append(batch_attr_scores)

    attr_scores = cat(attr_scores)

    # OPTIONAL: display TFModisco plots
    # from modisco.visualization import viz_sequence
    #
    # for i in range(10):
    #     viz_sequence.plot_weights(attr_scores[i], subticks_frequency=200)
    #     plt.show()

    # Generate a plot

    # Calculate the mean of the attribution score at each coordinate
    coordinate_scores = attr_scores.sum(dim=1).abs().squeeze()

    # Set a variable for moving average window size
    k = settings.window_size
    coord = range(3000 - k + 1)
    xticks = array(coord) + int(k/2)

    # Calculate moving averages
    moving_average = []

    for i in coord:
        kmer_mean = coordinate_scores[:, i:(i+k)].mean(dim=-1)
        moving_average.append(kmer_mean.mean(dim=0).item())

    moving_average = array(moving_average)

    # Set parameters of figure
    plt.rcParams["figure.figsize"] = (8, 2.5)
    fontdict = {'font': 'serif'}
    fig, ax = plt.subplots()

    # Find argmax and argmin in promoter and terminator
    split_idx = 1500 - int(k/2)
    min_promoter = moving_average[:split_idx].min()
    max_promoter = moving_average[:split_idx].max()
    min_terminator = moving_average[split_idx:].min()
    max_terminator = moving_average[split_idx:].max()

    # Plot moving average bar plot
    ax.bar(xticks, moving_average, color='darkcyan')
    ax.set_xticks([xticks[0], 500, 1000, 1500, 2000, 2500, xticks[-1]],
                  labels=[xticks[0], 500, 'TSS', 1500, 'TTS', 2500, xticks[-1]],
                  fontdict=fontdict)

    # Set axis labels
    ax.set_xlabel('Nucleotide index', fontdict=fontdict)
    ax.set_ylabel('Absolute attribution score', fontdict=fontdict)

    # Format and save figure
    fig.tight_layout()
    plt.savefig(f'deeplift_{settings.reference}.pdf')
