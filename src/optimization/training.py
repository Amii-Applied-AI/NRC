"""
Authors: Nicolas Raymond
         Fatima Davelouis
         Ruchika Verma

Description: Stores the Trainer and EarlyStopper classes used to train the models.           
"""
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from json import dump as json_dump
from numpy import append, array, exp, cos, pi
from numpy import inf as np_inf
from os import remove
from os.path import join
from tqdm import tqdm
from torch import autocast, device, float16, load, no_grad, save, Tensor
from torch.cuda.amp import GradScaler
from torch.nn import Module
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torch.utils.data import DataLoader, Dataset
from uuid import uuid4

# Project imports
from settings.paths import CHECKPOINTS
from src.utils.loss import Loss
from src.utils.metrics import BinaryClassificationMetric, Metric


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + cos(pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class Trainer(ABC):
    """
    Abstract Trainer class with common methods and 
    attributes of all other specialized trainer classes.
    """
    def __init__(self, loss_fct_name: str) -> None:
        """
        Sets the name of the loss function and initializes
        all protected attributes to their default value.
        
        Args:
            loss_fct (str): name of the loss function.
        """
        self._in_training: bool = False
        self._loss_fct_name: str = loss_fct_name
        self._optimizer: Optimizer = None
        self._scheduler: MultiStepLR = None
        self._valid_metric: str = None

        # Dictionary that keeps track of metrics evolution
        self._progress: dict[str, dict[str, list[float]]] = {'train': {self._loss_fct_name: []},
                                                             'valid': {self._loss_fct_name: []}}

    @property
    def loss_fct_name(self) -> str:
        """
        Returns the name of the loss function used by the trainer.
        """
        return self._loss_fct_name

    @abstractmethod
    def _update_weights(self,
                        model: Module,
                        dev: device,
                        dataloader: DataLoader,
                        metrics: list[Metric]) -> Module:
        """
        Executes one training epoch.
        
        Args:
            model (Module): neural network.
            dev (device): device on which to send the model and its inputs.
            dataloader (DataLoader): dataloader containing the training data.
            metrics (list[Metric]): list of metrics measured after the epoch.
            
        Returns:
            Module: optimized model.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self,
                 model: Module,
                 dev: device,
                 dataloader: DataLoader,
                 metrics: list[Metric]) -> None:
        """
        Evaluates the current model using the data provided in the given dataloader.

        Args:
            model (Module): neural network.
            dev (device): device on which to send the model and its inputs.
            dataloader (DataLoader): dataloader containing test or validation data.
            metrics (list[Metric]): list of metrics measured after the epoch.
        """

        raise NotImplementedError

    def train(self,
              model: Module,
              dev: device,
              datasets: tuple[Dataset, Dataset],
              batch_sizes: tuple[int, int],
              metrics: list[Metric],
              lr: float = 5e-5,
              max_epochs: int = 200,
              patience: int = 10,
              weight_decay: float = 1e-2,
              record_path: str = None,
              warmup_steps: int = 900,
              return_epochs: bool = False,
              collate_fn: Callable = None) -> Module | tuple[Module, int]:
        """
        Optimizes the weight of a neural network.
        
        Args:
            model (Module): neural network.
            dev (device): device on which to send the model and its inputs.
            datasets (tuple[Dataset, Dataset]): tuple of datasets. The first will be used
                                                for training and the second for validation.
            batch_sizes (tuple[int, int]): tuple of batch sizes. The first will be used for
                                           training and the second for validation.
            metrics: (list[Metric]): list of metrics. If not empty, the last one will be used
                                     for early stopping. Otherwise, the loss function is used.
            lr (float, optional): initial learning rate.
                                  Default to 5e-5.
            max_epochs (int, optional): maximal number of epochs executed during the training.
                                        Defaults to 200.
            patience (int, optional): number of consecutive epochs allowed without improvement
                                      in the validation score.
                                      Default to 10.
            weight_decay (float, optional): weight associated to the L2 penalty 
                                            in the loss function.
                                            Defaults to 1e-2.
            record_path (str, optional): path of the file (w/o extension) that will contain the
                                         scores recorded during the training. If no path is
                                         provided, no file will be saved.
                                         Default to None.
            warmup_steps(int, optional): number of warmup steps for lr schedule.
                                         Default to 900.
            return_epochs (bool, optional): if True, the number associated to the last training
                                            epoch and the best training epoch will be returned. 
                                            Default to False.

        Returns:
            Module: optimized model or (optimized model, last epoch, best epoch)
        """
        # Change the status of the trainer
        self._in_training = True

        # Set the optimizer
        self._optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=lr, weight_decay=weight_decay)

        # Set the scheduler
        self._scheduler = get_cosine_schedule_with_warmup(self._optimizer,
                                                          warmup_steps,
                                                          -(len(datasets[0]) // -batch_sizes[0]) * max_epochs)

        # Create the dataloaders
        train_dataloader = DataLoader(datasets[0],
                                      batch_sizes[0],
                                      shuffle=True,
                                      collate_fn=collate_fn)

        valid_dataloader = DataLoader(datasets[1],
                                      batch_sizes[1],
                                      shuffle=False,
                                      collate_fn=collate_fn)

        # Set the dictionary tracking the progress of the metrics
        for metric in metrics:
            for phase in self._progress.keys():
                self._progress[phase][metric.name] = []

        # Set the name of the metric used for the early stopping and initialize the EarlyStopper
        if len(metrics) != 0:
            self._valid_metric = metrics[-1].name
            early_stopper = EarlyStopper(patience, metrics[-1].to_maximize, record_path)
        else:
            self._valid_metric = self._loss_fct_name
            early_stopper = EarlyStopper(patience, False, record_path)

        # Execute the optimization process
        with tqdm(total=max_epochs, desc='Training') as pbar:
            for i in range(max_epochs):

                # Update the weights
                self._update_weights(model=model,
                                     dev=dev,
                                     dataloader=train_dataloader,
                                     metrics=metrics)

                # Process to the evaluation on the validation set
                self.evaluate(model=model,
                              dev=dev,
                              dataloader=valid_dataloader,
                              metrics=metrics)

                # Look for early stopping
                early_stopper(epoch=i,
                              score=self._get_last_valid_score(),
                              model=model)

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(self._get_postfix_dict())

                # Stop optimization if patience threshold is reached
                if early_stopper.stop:
                    print('\n' +
                          f'Training stopped at epoch {i} after {patience} epochs ' +
                          f'without improvement in the validation {self._valid_metric}')
                    break

        # Extraction of the parameters associated to the best validation score
        model.load_state_dict(early_stopper.get_best_params())

        # Removal of the checkpoint file created by the EarlyStopper
        # if no recording path was provided
        if record_path is None:
            early_stopper.remove_checkpoint()

        # Saving of the scores recorded during the training
        else:
            with open(f'{record_path}.json', 'w', encoding='utf-8') as file:
                json_dump(self._progress, file, indent=True)

        # Change the status of the trainer
        self._in_training = False

        if return_epochs:
            return model, i, early_stopper.best_epoch

        return model

    def _get_last_valid_score(self) -> float:
        """
        Extracts the most recent validation score obtained.
        
        Returns:
            float: most recent validation score
        """
        return self._progress['valid'][self._valid_metric][-1]

    def _get_postfix_dict(self) -> dict[str, float]:
        """
        Creates a dictionary containing the latest information 
        required to update the progress bar. 
        
        Returns:
            dict[str, float]: dictionary with current training and validation 
                              losses and metric scores.
        """
        # Build dict
        postfix_dict = {f'{metric} ({phase})': f'{self._progress[phase][metric][-1]:.2f}'
                        for phase in ['train', 'valid'] for metric in
                        [self._loss_fct_name, self._valid_metric]}

        postfix_dict['lr'] = f'{self._scheduler.get_last_lr()[0]}'

        return postfix_dict

class CMTrainer(Trainer):
    """
    Class encapsulating methods necessary to train a contrast model (CM).
    """
    def __init__(self,
                 loss_function: Loss) -> None:
        """
        Sets the loss function.

        Args:
            loss_function (Loss): loss function used for the training.
        """
        self.__loss_function: Loss = loss_function
        super().__init__(loss_fct_name=self.__loss_function.name)

    def _update_weights(self,
                        model: Module,
                        dev: device,
                        dataloader: DataLoader,
                        metrics: list[Metric]) -> Module:
        """
        Executes one training epoch.
        
        Args:
            model (Module): contrast model.
            dev (device): device on which to send the model and its inputs.
            dataloader (DataLoader): dataloader containing the training data.
            metrics (list[Metric]): list of metrics measured after the epoch.
            
        Returns:
            Module: optimized contrast model.
        """
        # Set the model in training mode and send it to the device
        model.train()
        model.to(dev)

        # Initialize variables to keep track of the progress of the loss,
        # the predictions, and the targets for one epoch
        total_loss, all_predictions, all_targets = 0, array([]), array([])

        # Set a GradScaler to train with mixed precision
        grad_scaler = GradScaler()

        # Execute batch gradient descent
        for seqs_aligned, targets, _ in dataloader:
            # Transfer data on the device
            targets = targets.to(dev)
            seqs_aligned = seqs_aligned.float().to(dev)

            # Clear the gradients
            self._optimizer.zero_grad()

            with autocast(device_type=dev.type, dtype=float16):
                predictions = model(seqs_aligned)
                loss = self.__loss_function(predictions, targets)

            # Update variables keeping track of the progress
            total_loss += loss.item()
            all_predictions = append(all_predictions, predictions.detach().to('cpu').numpy())
            all_targets = append(all_targets, targets.detach().to('cpu').numpy())

            # Do a gradient descent step
            grad_scaler.scale(loss).backward()
            grad_scaler.step(self._optimizer)
            grad_scaler.update()

            # Update the scheduler
            self._scheduler.step()

        if not self.__loss_function.is_for_regression:

            # During training, since predictions are logits, 
            # we need to turn them into probabilities
            all_predictions = 1/(1 + exp(-all_predictions))

        # Update overall performance progression
        self.__update_progress(predictions=all_predictions,
                               targets=all_targets,
                               mean_epoch_loss=total_loss/len(dataloader), 
                               metrics=metrics)

        return model

    def evaluate(self,
                 model: Module,
                 dev: device,
                 dataloader: DataLoader,
                 metrics: list[Metric]) -> None | tuple[Tensor, Tensor, Tensor]:

        """
        Evaluates the current model using the data provided in the dataloader.

        Args:
            model (Module): contrast model.
            dev (device): device on which to send the model and its inputs.
            dataloader (DataLoader): dataloader containing the test or validation data.
            metrics (list[Metric]): list of metrics measured after the epoch.

        Returns:
            None | tuple[Tensor, Tensor, Tensor]: predictions, targets, and species.
        """
        # Set the model in eval mode and send it to device
        model.eval()
        model.to(dev)

        # Initialize variables to keep track of the progress of the loss,
        # the predictions and the targets during the evaluation
        total_loss, all_predictions, all_targets, all_species  = 0, array([]), array([]), array([])

        with no_grad():

            # Proceed to the evaluation
            for seqs_aligned, targets, species in dataloader:
                # Transfer data on the device
                targets = targets.to(dev)
                seqs_aligned = seqs_aligned.float().to(dev)

                with autocast(device_type=dev.type, dtype=float16):
                    predictions = model(seqs_aligned)
                    loss = self.__loss_function(predictions, targets)

                # Update the variables keeping track of the progress
                total_loss += loss.item()
                all_predictions = append(all_predictions, predictions.detach().to('cpu').numpy())
                all_targets = append(all_targets, targets.detach().to('cpu').numpy())
                all_species = append(all_species, species.to('cpu').numpy())

        # Update overall performance progression (if the trainer is in training mode)
        if self._in_training:
            self.__update_progress(predictions=all_predictions,
                                   targets=all_targets,
                                   mean_epoch_loss=total_loss/len(dataloader),
                                   metrics=metrics,
                                   validation=True)

        # Otherwise, return the scores recorded, the predictions and the targets
        else:
            return all_predictions, all_targets, all_species

    def __update_progress(self,
                          predictions: list[float],
                          targets: list[float],
                          mean_epoch_loss: float,
                          metrics: list[Metric],
                          threshold: float = 0.5,
                          validation: bool = False) -> None:
        """
        Computes multiple performance metrics and save them in the '__progress' attribute.
        
        Args:
            predictions (list[float]): all predictions made during an epoch.
            targets (list[float]): targets associated to the predictions calculated
                                   during the epoch.
            mean_epoch_loss (float): mean of the loss values obtain for all batches.
            metrics (list[Metric]): list of metrics to measure.
            threshold (float, optional): Threshold used to assign label 1 to an observation
                                         (used for classification only). 
                                         Default to 0.5.
            validation (bool, optional): if True, indicates that the metrics were 
                                         recorded during validation. 
                                         Default to False.
        """
        # Set the correct section name in which to record the scores of the metrics
        section = 'valid' if validation else 'train'

        # Record the loss progress
        self._progress[section][self._loss_fct_name].append(mean_epoch_loss)

        # If we are in the context of regression
        if self.__loss_function.is_for_regression:

            # Generate the class of the predictions and the targets
            # based on their sign
            predicted_classes = predictions >= 0
            class_targets = targets >= 0

            for met in metrics:

                # Record the scores of the binary classification metrics
                if isinstance(met, BinaryClassificationMetric):
                    if not met.from_proba:
                        self._progress[section][met.name].append(met(predicted_classes,
                                                                     class_targets))
                # Record the scores of the regression metrics
                else:
                    self._progress[section][met.name].append(met(predictions, targets))

        # If we are in the context of classification
        else:

            # Generate the class of the predictions using the threshold
            predicted_classes = predictions >= threshold

            for met in metrics:

                # Record the scores of the binary classification metrics
                # using probabilities as prediction values
                if met.from_proba:
                    self._progress[section][met.name].append(met(predictions,
                                                                 targets))

                # Record the scores of the binary classification metrics
                # using classes as prediction values
                else:
                    self._progress[section][met.name].append(met(predicted_classes,
                                                                 targets))

class EarlyStopper:
    """
    Object in charge of monitoring validation score progress.
    """
    def __init__(self,
                 patience: int,
                 maximize: bool,
                 file_path: str = None) -> None:
        """
        Sets private and public attributes and then define the comparison
        method according to the given direction.
        
        Args:
            patience (int): number of epochs without improvement allowed.
            maximize (bool): if True, the metric used for early stopping must be maximized.
            file_path (str, optional): path of the file in which to save the best weights.
                                       Default to None.
        """
        # Set private attributes
        self.__best_epoch: int = 0
        self.__counter: int = 0
        self.__patience: int = patience
        self.__path_provided: bool = file_path is not None
        self.__stop: bool = False

        if self.__path_provided:
            self.__file_path: str = f'{file_path}.pt'
        else:
            self.__file_path: str = join(CHECKPOINTS, f"{uuid4()}.pt")

        # Set comparison method
        if maximize:
            self.__best_score: float = -np_inf
            self.__is_better: Callable[[float, float], bool] = lambda x, y: x > y

        else:
            self.__best_score: float = np_inf
            self.__is_better: Callable[[float, float], bool] = lambda x, y: x < y

    @property
    def best_epoch(self) -> int:
        """
        Returns the number of the epoch associated to the best score.

        Returns:
            int: epoch associated to the best score.
        """
        return self.__best_epoch

    @property
    def path_provided(self) -> bool:
        """
        Returns a bool indicating if a path was provided to save the weights.

        Returns:
            bool: True if a path was provided to the instance at its initialization.
        """
        return self.__path_provided

    @property
    def stop(self) -> bool:
        """
        Returns the value of the 'stop' attribute, which
        indicates if the early stopper as reached its budget (i.e., 'patience').

        Returns:
            bool: stop attribute that indicates if the early stopper 
                  as reached its budget (i.e., 'patience').
        """
        return self.__stop

    def __call__(self,
                 epoch: int,
                 score: float,
                 model: Module) -> None:
        """
        Compares the current best validation score against the given one
        and updates the object's attributes.
        
        Args:
            epoch (int): current epoch.
            score (float): new validation score.
            model (Module): current model for which we've seen the score.
        """
        # Increment the counter if the score is worst than the best score
        if not self.__is_better(score, self.__best_score):
            self.__counter += 1

            # Change early stopping status if the counter reach the patience
            if self.__counter >= self.__patience:
                self.__stop = True

        # Save the parameters of the model if the score is better
        # than the best score observed before
        else:
            self.__best_epoch = epoch
            self.__best_score = score
            save(model.state_dict(), self.__file_path)
            self.__counter = 0

    def remove_checkpoint(self) -> None:
        """
        Removes the checkpoint file.
        """
        remove(self.__file_path)

    def get_best_params(self) -> OrderedDict[str, Tensor]:
        """
        Return the last parameters that were save. 
        
        Returns:
            OrderedDict[str, tensor]: state dict containing parameters of the model.
        """
        return load(self.__file_path)
