# NRC-AMII Legume Differential Expression Project

## Project Tree
This section gives an overview of the project organization.
```
├── data                               
│   ├── raw                             -> Raw data.
│       └── pea_faba_medicago_grasspea
│           └── orthogroups.csv         -> csv file with all the orthogroups.
│           └── complete_orthogroups.csv-> csv file with the orthogroup of 4 members.
├── experiments                         -> Experiment scripts.
├── settings                            -> Files with important directory paths.
└── src                                 -> Code modules and functions required to run the experiments.
    ├── data                            -> Data-related modules and function
    │   ├── modules                     -> Objects dedicated to data storage and preprocessing. 
    │   └── scripts                     -> Scripts used to create interim and processed data.
    ├── models                          -> Pytorch models.
    ├── optimization                    -> Training modules.
    └── utils                           -> Other utilities such as custom metrics and loss functions.
```
Note that each Python file in the GitHub repository is accompanied with a description of its purpose. 

## Environment Setup

Replicate the environment using the provided file.
```
conda env create --file environment.yml
```
Once the environment is created, activate it and return to the root of the project.
```
conda activate legume-dge
```

## Process data into 5-fold stratified cross-validation split 
We can now proceed to the last step of data preprocessing by dividing five times the interim data into training, validation and test sets.
Too avoid any gene appearing in more than one of a set within a same fold, we excute data sampling using orthogroups. This task is accomplished with the script named ```src/data/scripts/process_by_orthogroups.py```. The latter script comes with arguments enabling to find the interim data that we want to process.

Here are the possible arguments:
* ```--normalization, -norm (str) - {median, deseq, None}```:
  * Choice of normalization method. Default to ```None``` when not provided.
* ```--synteny, -s (int) - {0, 5}```:
  * Synteny level. Default to ```0```.
* ```--plot (bool)```:
  * If provided, pie charts are saved to visualize the stratification of the splits. Default to False.'
 
The experiments in the paper are replicated using the following command.
```
python src/data/scripts/process_by_orthogroups.py --synteny 0
```
The results will appear at ```data/processed/pea_faba_medicago_grasspea/synteny_0```.

## Run experiments
All main experiments can currently be executed with the following script:
```
python experiments/evaluation.py
```
Below, we list the possible arguments that can be given to the script.

### Arguments

* ```--proseq, -pro (bool)```:
  * If provided, pro-seq data will be use instead of QuantSeq. Default to ```False```.
* ```--normalization, -norm (str) - {median, deseq, None}```:
  * Choice of normalization method. Default to ```None```.
* ```--synteny, -s (int) - {0, 5}```:
  * Synteny level. Default to ```0```.
* ```--remove_zeros, -rz (bool)```:
  * If provided, pairs of orthologs with a target of 0 are removed. Default to ```False```.
* ```--reverse_complements, -rc (bool)```:
  * If provided, reverse complements of ortholog pairs are included in the training set. Default to ```False```.
* ```--include_flipped_train, -flip_tr (bool)```:
  * If provided, flipped versions of the pairs of orthologs will also be included in the training set. Default to ```False```.
* ```--include_flipped_test, -flip_test (bool)```:
  * If provided, flipped versions of the pairs of orthologs will also be included in the test set. Default to ```False```.
* ```--loss (str) - {l2, l1, sl1, mtl}```:
  * Choice of loss function. L2 is for the mean squared error loss. L1 is for the mean absolute error loss. sL1 is for the SmoothL1 loss. MTL is for the multitask loss. Default to ```l2```.
* ```--beta (float)```:
  * Beta parameter of the SmoothL1 or MTL loss. For the MTL loss, beta must be in the range ```[0, 1]```. Default to ```0.5```.
* ```--eval_metric, -em (str) - {rmse, mae, spearmanr}```:
  * Choice of evaluation metric used for early stopping. Default to ```spearmanr```.
* ```--train_batch_size, -trbs (int)```:
  * Training batch size. Default = ```32```.
* ```--test_batch_size, -tebs (int)```:
  * Test and validation batch size. Default = ```32```.
* ```--lr (float)```:
  * Maximum learning rate. Default = ```8e-5```.
* ```--max_epochs, -epochs (int)```:
  * Maximum number of epochs. Default = ```50```.
* ```--patience (int)```:
  * Number of epochs without improvement allowed before stopping the training. Only weights associated to the best validation score are kept following the training. Default = ```20```.
* ```--weight_decay, -wd (float)```:
  * Coefficient associated to the L2 penalty term in the loss function. Default = ```1e-2```.
* ```--nb_folds (int) - {1, 2, 3, 4, 5}```:
  * Number of cross-validation folds to execute. Default = ```5```.
* ```--restart_from (str)```:
  * If the path of a past experiment is provided, pre-trained weights of each fold will be used to initialize model. Default = ```None```.
* ```--device_id, -dev (int)```:
  * Cuda device ID. Default = ```0```.
* ```--memory_frac, -memory (float) - [0, 1]```:
  * Percentage of device allocated to the experiment. Default = ```1```.
* ```--seed (int)```:
  * Seed value used for experiment reproducibility. Default = ```1```.
* ```--warmup_steps, -warmup (int)```:
  * Number of warmup steps for the learning rate schedule. Default = ```900```.
