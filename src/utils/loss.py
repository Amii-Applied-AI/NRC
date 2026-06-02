"""
Authors: Nicolas Raymond

Description: Definition of customized loss functions.
"""

from abc import ABC, abstractmethod
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits, l1_loss, \
    mse_loss, smooth_l1_loss

class Loss(ABC):
    """
    Loss abstract class.
    """
    def __init__(self,
                 name: str,
                 regression: bool) -> None:
        """
        Sets the name and the type of the loss function.

        Args:
            name (str): name of the loss function.
            regression (bool): if True, indicates that the loss is used for regression.
                               Otherwise, the loss is used for classification.
        """
        self.__name: str = name
        self.__regression: bool = regression

    @property
    def name(self) -> str:
        """
        Returns the name of the loss.

        Returns:
            str: name of the loss.
        """
        return self.__name

    @property
    def is_for_regression(self) -> bool:
        """
        Returns a bool indicating if the loss is used for regression.
        
        False indicates that the loss is used for classification.

        Returns:
            bool: bool indicating if the loss is used for regression.
        """
        return self.__regression

    @abstractmethod
    def __call__(self,
                 predicted_values: Tensor,
                 targets: Tensor) -> Tensor:
        """
        Calculates the loss based on the predicted values and the targets.
        
        Args:
            predicted_values (Tensor): values predicted by a model (N,).
            targets (Tensor): ground truth (N,).

        Returns:
            Tensor: loss (1,)
        """
        raise NotImplementedError

class BinaryCrossEntropy(Loss):
    """
    Binary cross entropy loss function.
    """
    def __init__(self) -> None:
        """
        Initializes the loss.
        """
        super().__init__(name='BCE', regression=False)

    def __call__(self,
                 predicted_values: Tensor,
                 targets: Tensor) -> Tensor:
        """
        Calculates the loss based on the predicted values and the targets.
        
        Args:
            predicted_values (Tensor): values predicted by a model (N,).
            targets (Tensor): ground truth (N,).

        Returns:
            Tensor: loss (1,)
        """
        return binary_cross_entropy_with_logits(predicted_values, targets)


class MSE(Loss):
    """
    Mean squared error loss.
    
    Each term in the sum is represented as follow:
    (y_hat - y)^2
    """
    def __init__(self) -> None:
        """
        Initializes the loss.
        """
        super().__init__(name='MSE', regression=True)

    def __call__(self,
                 predicted_values: Tensor,
                 targets: Tensor) -> Tensor:
        """
        Calculates the mean squared error.

        Args:
            predicted_values (Tensor): values predicted by a model (N,).
            targets (Tensor): ground truth (N,).

        Returns:
            Tensor: loss (1,)
        """

        return mse_loss(predicted_values, targets)


class L1(Loss):
    """
    Mean absolute error loss (L1 loss).
    
    Each term in the sum is represented as follow:
    |y_hat - y|
    """
    def __init__(self) -> None:
        """
        Initializes the loss.
        """
        super().__init__(name='L1', regression=True)

    def __call__(self,
                 predicted_values: Tensor,
                 targets: Tensor) -> Tensor:
        """
        Calculates the mean absolute error.

        Args:
            predicted_values (Tensor): values predicted by a model (N,).
            targets (Tensor): ground truth (N,).

        Returns:
            Tensor: loss (1,)
        """

        return l1_loss(predicted_values, targets)


class SmoothL1(Loss):
    """
    Smooth L1 loss.
    
    See https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html#torch.nn.SmoothL1Loss
    """
    def __init__(self, beta: float) -> None:
        """
        Initializes the loss.

        Args:
            beta (float): threshold determining with part of the piece-wise 
                          function to use.
            
        """
        super().__init__(name='sL1', regression=True)
        self.__beta: float = beta

    def __call__(self,
                 predicted_values: Tensor,
                 targets: Tensor) -> Tensor:
        """
        Calculates the smooth l1 loss.

        Args:
            predicted_values (Tensor): values predicted by a model (N,).
            targets (Tensor): ground truth (N,).

        Returns:
            Tensor: loss (1,)
        """

        return smooth_l1_loss(predicted_values, targets, beta=self.__beta)


class MultitaskLoss(Loss):
    """
    Loss that constitutes a weighted average of the mean squared error
    and the binary cross entropy.
    """
    def __init__(self,
                 sigma: float,
                 beta: float = 0.5) -> None:
        """
        Sets the name of the metric and saves the alpha and sigma parameters

        Args:
            sigma (float): loss scaling factor (std of targets in training set)
            beta (float): coefficient multiply mean squared error loss.
                          Default to 0.5.
        """
        super().__init__(name='MTL', regression=True)
        self.__sigma: float = sigma
        self.__beta: float = beta

    def __call__(self,
                 predicted_values: Tensor,
                 targets: Tensor) -> Tensor:
        """
        Calculates the weighted average of the MSE and the BCE losses.

        Args:
            predicted_values (Tensor): values predicted by a model (N,).
            targets (Tensor): ground truth (N,).

        Returns:
            Tensor: loss (1,)
        """
        regression_loss = mse_loss(predicted_values, targets)/(self.__sigma**2)
        classification_loss = binary_cross_entropy_with_logits(predicted_values/self.__sigma,
                                                               (targets > 0).float())

        return self.__beta*regression_loss + (1-self.__beta)*classification_loss
