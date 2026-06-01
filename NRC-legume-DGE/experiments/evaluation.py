"""
Authors: Nicolas Raymond

Description: Evaluates model with the execution of a cross
             validation on the Pea-Faba-Medicago-Grasspea dataset. 
"""

import sys

from argparse import ArgumentParser
from datetime import datetime
from json import dump, load
from os import mkdir
from os.path import abspath, join, pardir
from shutil import move
from time import time
from torch import device, Tensor
from torch import load as th_load
from torch.backends import cudnn
from torch.cuda import is_available, set_per_process_memory_fraction
from torch.utils.data import DataLoader

# Project imports
sys.path.append(abspath(join(__file__, *[pardir]*2)))
from settings.paths import PFMG_DATA, RECORDS
from src.data.modules.constants import Maps
from src.data.modules.constants import Keys as k
from src.data.modules.dataset import SeqDataset
from src.models.cnn import WCNN
from src.optimization.training import CMTrainer
from src.utils import metrics as m
from src.utils.analysis import save_comparison_figure, save_progress_figure, save_table
from src.utils.loss import BinaryCrossEntropy, L1, MSE, MultitaskLoss, SmoothL1
from src.utils.reproducibility import SEED, set_seed


# Definition of the arguments parsing function
def get_settings():
    """
    Retrieves the settings given in the terminal to run the script.
    """
    parser = ArgumentParser(usage='python evaluation.py',
                            description='Evaluates with the execution of a cross ' +
                            'validation on the Pea-Faba-Medicago-Grasspea (PFMG) dataset. ')

    parser.add_argument('-pro', '--proseq', default=False, action='store_true',
                        help='If provided, pro-seq data will be use instead of QuantSeq.')

    # Normalization method
    parser.add_argument('-norm', '--normalization', type=str, default=None,
                        choices=[k.MEDIAN, k.DESEQ, None],
                        help='Choice of normalization method. Default to None.')

    # Remove zeros
    parser.add_argument('-rz', '--remove_zeros', default=False, action='store_true',
                        help='If provided, pairs of orthologs with a target of 0 are removed.')

    # Reverse complements
    parser.add_argument('-rc', '--reverse_complements', default=False, action='store_true',
                        help='If provided, reverse complements of ortholog pairs ' +
                        'are included in the training set.')

    # Include flipped orthologs (train)
    parser.add_argument('-flip_tr', '--include_flipped_train', default=False, action='store_true',
                        help='If provided, flipped versions of the pairs of orthologs will ' +
                        'also be included in the training set. This option is only ' +
                        'relevant for wcnn model.')

    # Include flipped orthologs (test)
    parser.add_argument('-flip_test', '--include_flipped_test', default=False, action='store_true',
                        help='If provided, flipped versions of the pairs of orthologs will ' +
                        'also be included in the test set.')

    # Loss function
    parser.add_argument('-loss', '--loss', default='l2', type=str,
                        choices=['l2', 'l1', 'sl1', 'mtl'],
                        help='Choice of loss function. ' +
                              'L2 is for the mean squared error loss. ' +
                              'L1 is for the mean absolute error loss. ' +
                              'sL1 is for the SmoothL1 loss. ' +
                              'MTL is for the multitask loss. ' +
                              'Default to l2.')

    # Beta
    parser.add_argument('-beta', '--beta', type=float, default=0.5,
                        help='Beta parameter of the SmoothL1 or MTL loss. ' +
                             'For the MTL loss, beta must be in the range [0, 1]. ' +
                             'Default to 0.5.')

    # Eval metric
    parser.add_argument('-em', '--eval_metric', default='spearmanr', type=str,
                        choices=['rmse', 'mae', 'spearmanr'],
                        help='Choice of evaluation metric used for early stopping. ' +
                        'Default to spearmanr.')

    # Synteny level
    parser.add_argument('-s', '--synteny', type=int, default=0, choices=[0, 5],
                        help='Synteny level. Default to 0.')

    # Batch size
    parser.add_argument('-trbs', '--train_batch_size', type=int, default=32,
                        help='Batch size. Default to 32')

    parser.add_argument('-tebs', '--test_batch_size', type=int, default=32,
                        help='Batch size. Default to 32')

    # Maximum learning rate
    parser.add_argument('-lr', '--lr', type=float, default=8e-5,
                        help='Maximum learning rate. Default to 8e-5.')

    # Max epochs
    parser.add_argument('-epochs', '--max_epochs', type=int, default=50,
                        help='Maximum number of epochs. Default to 50.')

    # Patience
    parser.add_argument('-patience', '--patience', type=int, default=10,
                        help='Number of epochs without improvement allowed ' +
                        'before stopping the training. Only weights associated to ' +
                        'the best validation score are kept following the training. ' +
                        'Default to 10.')

    # Weight decay
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-2,
                        help='Weight decay (L2 penalty coefficient). Default to 1e-2.')

    # Number of folds
    parser.add_argument('-nb_folds', '--nb_folds', default=5, type=int,
                        choices=list(range(6)),
                        help='Number of cross-validation folds to execute. Default to 5.')

    # Restart path
    parser.add_argument('-restart_from', '--restart_from', type=str, default=None,
                        help='If the path of a past experiment is provided, ' +
                        'pre-trained weights of each fold will be used to initialize model.')

    # Device ID
    parser.add_argument('-dev', '--device_id', type=int, default=0,
                        help='Cuda device ID. Default to 0.')

    # Memory fraction
    parser.add_argument('-memory', '--memory_frac', type=float, default=1,
                        help='Percentage of device allocated to the experiment. Default to 1')
    # Seed
    parser.add_argument('-seed', '--seed', type=int, default=SEED,
                        help=f'Seed value used for experiment reproducibility. Default to {SEED}.')

    # Warmup steps for the lr schedule
    parser.add_argument('-warmup', '--warmup_steps', type=int, default=900,
                        help='Number of warmup steps for the learning rate schedule')

    return parser.parse_args()


# Execution of the script
if __name__ == '__main__':

    # Retrieve environment settings
    SETTINGS: dict = vars(get_settings())
    EXPERIMENT_FOLDER: str = join(RECORDS, f"{datetime.now().strftime('%d_%m_%Y_%H:%M:%S')}")
    FIGURES_FOLDER: str = join(EXPERIMENT_FOLDER, "figures")
    CSV_FOLDER: str = join(EXPERIMENT_FOLDER, 'csv')
    JSON_FOLDER: str = join(EXPERIMENT_FOLDER, 'json')

    # Set the device
    if is_available():
        DEVICE = device(f"cuda:{SETTINGS['device_id']}")
    else:
        raise ValueError('A GPU is required for the experiment.')

    # Set the memory allocated to the device
    if 0 < SETTINGS['memory_frac'] < 1:
        set_per_process_memory_fraction(fraction=SETTINGS['memory_frac'], device=DEVICE)

    # Create the folders that will contain the results of the experiment
    mkdir(EXPERIMENT_FOLDER)
    mkdir(FIGURES_FOLDER)
    mkdir(CSV_FOLDER)
    mkdir(JSON_FOLDER)

    # Set the data folder
    postfix = f'synteny_{SETTINGS["synteny"]}'
    if SETTINGS['normalization'] is not None:
        postfix += f'_{SETTINGS["normalization"]}'

    tokens = False
    DATA_FOLDER: str = join(PFMG_DATA, postfix)

    # Save the settings of the experiment
    with open(join(EXPERIMENT_FOLDER, 'settings.json'), 'w', encoding="utf-8") as file:
        dump(SETTINGS, file, indent=True)

    # Initialize the metrics
    metrics = [m.Accuracy(), m.BalancedAccuracy(), m.F1Score(),
               m.Precision(), m.Recall(), m.AreaUnderROC()]

    # Initialize a dictionary to save test scores among pairs of species
    test_scores = {'all': {met.name: [] for met in metrics}}
    for pair in Maps.PAIR2ID:
        test_scores[pair] = {met.name: [] for met in metrics}

    # Initialize dictionaries to save train and valid scores over all data points
    train_scores = {met.name: [] for met in metrics}
    valid_scores = {met.name: [] for met in metrics}

    # Initialize a dictionary to save the training times and number of epochs
    epochs_and_times = {'total_epochs': [], 'best_epoch': [], 'times': []}

    # Set the seed
    cudnn.deterministic = True
    set_seed(seed_value=SETTINGS['seed'], n_gpu=1)

    # For each fold
    for i in range(SETTINGS['nb_folds']):
        # Start a timer
        start = time()

        # Create the training, validation and test datasets
        print(f'\n\n{"-"*10} Fold #{i} {"-"*10}')

        # Initialize the complete model
        reg = True
        model = WCNN(regression=reg, encoding_size=4)

        # Create the training set
        train_set = SeqDataset.from_path(path=join(DATA_FOLDER, f'Fold_{i}_train.pkl'),
                                         include_flip=SETTINGS['include_flipped_train'],
                                         include_reverse_complement=SETTINGS['reverse_complements'],
                                         tokens=tokens,
                                         proseq=SETTINGS['proseq'],
                                         remove_zeros=SETTINGS['remove_zeros'],
                                         regression=reg)

        # Create the validation set
        valid_set = SeqDataset.from_path(path=join(DATA_FOLDER, f'Fold_{i}_valid.pkl'),
                                         include_flip=SETTINGS['include_flipped_train'],
                                         include_reverse_complement=False,
                                         tokens=tokens,
                                         proseq=SETTINGS['proseq'],
                                         remove_zeros=SETTINGS['remove_zeros'],
                                         regression=reg)

        # Create the test set
        test_set = SeqDataset.from_path(path=join(DATA_FOLDER, f'Fold_{i}_test.pkl'),
                                        include_flip=SETTINGS['include_flipped_test'],
                                        include_reverse_complement=False,
                                        tokens=tokens,
                                        proseq=SETTINGS['proseq'],
                                        remove_zeros=SETTINGS['remove_zeros'],
                                        regression=reg)

        # Load weights from a previous experience if required
        if SETTINGS['restart_from'] is not None:
            model.load_state_dict(th_load(join(SETTINGS['restart_from'], f'fold_{i}.pt')))

        # Define the loss function
        if reg:
            if SETTINGS['loss'] == 'l2':
                loss_fct = MSE()
            elif SETTINGS['loss'] == 'l1':
                loss_fct = L1()
            elif SETTINGS['loss'] == 'sl1':
                loss_fct = SmoothL1(beta=SETTINGS['beta'])
            else:
                loss_fct = MultitaskLoss(sigma=train_set.labels_std,
                                         beta=SETTINGS['beta'])
        else:
            loss_fct = BinaryCrossEntropy()

        # Initialize the trainer
        trainer = CMTrainer(loss_function=loss_fct)

        if SETTINGS['max_epochs'] != 0:

            # Train the model
            model, last, best = trainer.train(model=model,
                                              dev=DEVICE,
                                              datasets=(train_set, valid_set),
                                              batch_sizes=(SETTINGS['train_batch_size'],
                                                        SETTINGS['test_batch_size']),
                                              metrics=metrics,
                                              lr=SETTINGS['lr'],
                                              max_epochs=SETTINGS['max_epochs'],
                                              patience=SETTINGS['patience'],
                                              weight_decay=SETTINGS['weight_decay'],
                                              record_path=join(EXPERIMENT_FOLDER, f'fold_{i}'),
                                              warmup_steps=SETTINGS['warmup_steps'],
                                              return_epochs=True)

            # Save the number of epochs and the training time
            epochs_and_times['total_epochs'].append(last+1)
            epochs_and_times['best_epoch'].append(best)
            epochs_and_times['times'].append(round((time() - start)/60, 2))
            print(f"Model training time: {epochs_and_times['times'][-1]}")

            # Move the json file saved by the trainer to the JSON folder
            move(join(EXPERIMENT_FOLDER, f'fold_{i}.json'), join(JSON_FOLDER, f'fold_{i}.json'))

            # Load history of metrics progression during training
            with open(join(JSON_FOLDER, f'fold_{i}.json'), 'r', encoding="utf-8") as file:
                metrics_progress = load(file)

            for k in metrics_progress['train'].keys():

                # Create figures for the progression of each metric during the training
                save_progress_figure(train_scores=metrics_progress['train'][k],
                                     valid_scores=metrics_progress['valid'][k],
                                     metric_name=k,
                                     figure_path=join(FIGURES_FOLDER, f'{k}_fold_{i}.pdf'))

                # Save scores associated to the best epoch
                if k != loss_fct.name:
                    train_scores[k].append(metrics_progress['train'][k][best])
                    valid_scores[k].append(metrics_progress['valid'][k][best])

        # Create the test dataloader
        dl = DataLoader(test_set, batch_size=SETTINGS['test_batch_size'], shuffle=False)

        # Evaluate the model on the test data
        predictions, targets, species = trainer.evaluate(model=model,
                                                         dev=DEVICE,
                                                         dataloader=dl,
                                                         metrics=metrics)

        # Delete the test dataloader to free some space
        del dl

        # For each pair of specie
        for pair in test_scores.keys():

            # Find the idx associated to the species pair and define a string used to save files
            if pair == 'all':
                idx = list(range(len(test_set)))
                file_prefix = 'all'
            else:
                binary_mask = Tensor(species) == Maps.PAIR2ID[pair]
                idx = (binary_mask).nonzero().squeeze().tolist()
                file_prefix = f'{pair[0]}_{pair[1]}'

            if len(idx) != 0:
                
                # Extract the predictions and the targets related to the specie
                sub_pred, sub_targets = predictions[idx], targets[idx]

                # Compute the metrics using this data
                if reg:
                    sub_class_pred = sub_pred >= 0
                    sub_class_targets = sub_targets >= 0
                else:
                    sub_class_pred = sub_pred >= 0.5
                    sub_class_targets = sub_targets

                for met in metrics:
                    if isinstance(met, m.BinaryClassificationMetric):
                        test_scores[pair][met.name].append(met(sub_class_pred, sub_class_targets))
                    else:
                        score = met(sub_pred, sub_targets)
                        test_scores[pair][met.name].append(score)

                # Save a json file with the predictions and the targets
                sub_pred, sub_targets = sub_pred.tolist(), sub_targets.tolist()
                json_file_path = join(JSON_FOLDER, f'{file_prefix}_pred_targets_{i}.json')
                with open(json_file_path, 'w', encoding="utf-8") as file:
                    dump({'predictions': sub_pred, 'targets': sub_targets}, file, indent=True)

                # Save a figure comparing predictions and targets
                save_comparison_figure(pred=sub_pred, targets=sub_targets,
                                       path=join(FIGURES_FOLDER, f'{file_prefix}_pred_targets_{i}.pdf'))

        # Print current test scores
        print(test_scores)

    # For pair of species (and all species together), save the test scores in a csv file
    for pair, scores in test_scores.items():
        if pair != 'all':
            save_table(data=scores, path=join(CSV_FOLDER, f'{pair[0]}_{pair[1]}_test_scores.csv'))
        else:
            save_table(data=scores, path=join(CSV_FOLDER, 'all_test_scores.csv'))

    if SETTINGS['max_epochs'] != 0:

        # Save training time and number of epochs
        save_table(data=epochs_and_times, path=join(CSV_FOLDER, 'epochs_and_times.csv'))

        # Save best training metrics obtained throughout the folds
        save_table(data=train_scores, path=join(CSV_FOLDER, 'all_train_scores.csv'))

        # Save best validation metrics obtained throughout the folds
        save_table(data=valid_scores, path=join(CSV_FOLDER, 'all_valid_scores.csv'))
