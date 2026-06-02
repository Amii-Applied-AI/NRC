"""
Authors: Nicolas Raymond,
         Ruchika Verma

Description: Stores the class associated to the Washburn CNN model.
             
"""
from torch import cat, nn, Tensor, zeros

class WCNN(nn.Module):
    """
    CNN model based on the architecture proposed in: 
    
    "Evolutionarily informed deep learning methods for predicting
    relative transcript abundance from DNA sequence".
    
    Blocks are separated in two to avoid the following UserWarning:
    
    Using padding='same' with even kernel lengths and odd dilation may require
    a zero-padded copy of the input be created
    
    Padding values provided were chosen to replicate the behavior
    obtained using 'padding=same' with Keras:
    
    See https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121/3
    """

    def __init__(self,
                 regression: bool = False,
                 encoding_size: int = 4) -> None:
        """
        Initializes the layers of the model
        
        regression (bool): if True, no sigmoid activation function will be
                           used at the end of the network.
                           Default to False.
        """
        super().__init__()

        # Nucleotide encoding size
        self.__encoding_size: int = encoding_size

        # Block 1
        self.__block_1_0 = nn.Sequential(nn.Conv2d(1, 64,
                                                   kernel_size=(8,encoding_size),
                                                   padding='valid'),
                                         nn.ReLU())

        self.__block_1_1 = nn.Sequential(nn.Conv2d(64, 64,
                                                   kernel_size=(8,1),
                                                   padding='valid'),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=(8,1),
                                                      stride=(8,1),
                                                      padding=(2, 0)),
                                         nn.Dropout2d(0.25))

        # Block 2
        self.__block_2_0 = nn.Sequential(nn.Conv2d(64, 128,
                                                   kernel_size=(8,1),
                                                   padding='valid'),
                                         nn.ReLU())

        self.__block_2_1 = nn.Sequential(nn.Conv2d(128, 128,
                                                   kernel_size=(8,1),
                                                   padding='valid'),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=(8,1),
                                                      stride=(8,1),
                                                      padding=(1, 0)),
                                         nn.Dropout2d(0.25))

        # Block 3
        self.__block_3_0 = nn.Sequential(nn.Conv2d(128, 64,
                                                   kernel_size=(8,1),
                                                   padding='valid'),
                                         nn.ReLU())

        self.__block_3_1 = nn.Sequential(nn.Conv2d(64, 64,
                                                   kernel_size=(8,1),
                                                   padding='valid'),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=(8,1),
                                                      stride=(8,1),
                                                      padding=(1, 0)),
                                         nn.Dropout2d(0.25))

        # Linear layers (i.e. fully connected layers)
        self.__linear_layers = nn.Sequential(nn.Flatten(),
                                             nn.Linear(768, 128),
                                             nn.ReLU(),
                                             nn.Dropout1d(0.25),
                                             nn.Linear(128, 64),
                                             nn.ReLU(),
                                             nn.Linear(64, 1))

        # Last activation function
        self.__last_activation = nn.Identity() if regression else nn.Sigmoid()

    def forward(self,
                seq_a: Tensor,
                seq_b: Tensor) -> Tensor:
        """
        Executes a forward pass two batches of DNA sequences.
        
        Args:
            seq_a (tensor): batch of DNA sequences (BATCH SIZE, C, 3000)
            seq_b (tensor): batch of DNA sequences (BATCH SIZE, C, 3000)
            
        Returns:
            Tensor: gene expression difference (BATCH SIZE, ). 
        """
        # Reshape the tensors
        # (BATCH SIZE, C, 3000) -> (BATCH SIZE, 1, 3000, C)
        seq_a = seq_a.unsqueeze(dim=1).permute(0, 1, 3, 2)
        seq_b = seq_b.unsqueeze(dim=1).permute(0, 1, 3, 2)

        # Concatenate sequences and add 0 padding in the middle
        # (BATCH SIZE, 1, 3000, C), (BATCH SIZE, 1, 3000, C) -> (BATCH SIZE, 1, 6030, C)
        x = cat([seq_a,
                 zeros(seq_a.shape[0], 1, 30, self.__encoding_size).to(seq_a.device),
                 seq_b], dim=2)

        # Forward pass through block 1
        # (BATCH SIZE, 1, 6030, C) -> (BATCH SIZE, 64, 750, 1)
        x = self.__block_1_1(nn.functional.pad(self.__block_1_0(x), (0, 0, 3, 4)))

        # Forward pass through block 2
        # (BATCH SIZE, 64, 750, 1) -> (BATCH SIZE, 128, 750, 1)
        x = self.__block_2_0(nn.functional.pad(x, (0, 0, 3, 4)))
        # (BATCH SIZE, 128, 750, 1) -> (BATCH SIZE, 128, 94, 1)
        x = self.__block_2_1(nn.functional.pad(x, (0, 0, 3, 4)))

        # Forward pass through block 3
        # (BATCH SIZE, 128, 94, 1) -> (BATCH SIZE, 64, 94, 1)
        x = self.__block_3_0(nn.functional.pad(x, (0, 0, 3, 4)))
        # (BATCH SIZE, 64, 94, 1) -> (BATCH SIZE, 64, 12, 1)
        x = self.__block_3_1(nn.functional.pad(x, (0, 0, 3, 4)))

        # Forward pass in the fully connected layers followed by an activation function
        if self.training:
            return self.__linear_layers(x).squeeze(dim=-1)

        # Forward pass in the fully connected layers followed by an activation function
        return self.__last_activation(self.__linear_layers(x)).squeeze()
