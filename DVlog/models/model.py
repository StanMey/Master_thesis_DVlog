import math

import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    """The Positional Encoding layer used for processing the data before putting it into the Transformer Encoder.
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    http://nlp.seas.harvard.edu/annotated-transformer/
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)

        # run it for the even indices
        pe[:, 0, 0::2] = torch.sin(position * div_term)

        # check whether d_model is even or odd before running the odd indices (https://discuss.pytorch.org/t/transformer-example-position-encoding-function-works-only-for-even-d-model/100986/2)
        if d_model % 2 == 0:
            # the dimension is even so just do the normal thing
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        else:
            # the dimension is odd so adjust for this
            pe[:, 0, 1::2] = torch.cos(position * div_term)[:, 0:-1]

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class UnimodalTransformerEncoder(nn.Module):
    """_summary_
    
    """

    def __init__(self, data_shape: tuple[int, int], d_model: int = 256, kernel_size: int = 4, stride: int = 2, n_heads: int = 8):
        super().__init__()
        """
        :param data_shape: The shape of the input data
        :type data_shape: tuple[int, int]
        :param d_model: The dimension of the encoder representation (d_u in the paper), defaults to 256
        :type d_model: int, optional
        :param kernel_size: The size of the sliding window, defaults to 4
        :type kernel_size: int, optional
        :param stride: The stride of the window, defaults to 2
        :type stride: int, optional
        :param n_heads: The number of attention heads in the encoder, defaults to 8
        :type n_heads: int, optional
        """
        self.data_shape = data_shape
        self.d_model = d_model  # d_u
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_heads = n_heads

        # input
        # convolutional layers
        self.conv1 = torch.nn.Conv1d(in_channels=self.data_shape[-1], out_channels=self.data_shape[-1],
                                     kernel_size=self.kernel_size, stride=self.stride, padding=1)  # padding of 1 since we otherwise don't get to t/4
        self.conv2 = torch.nn.Conv1d(in_channels=self.data_shape[-1], out_channels=self.data_shape[-1],
                                     kernel_size=self.kernel_size, stride=self.stride, padding=1)

        # setup the embedding layer and pos encoding
        self.fc = nn.Linear(self.data_shape[-1], self.d_model)  # embedding layer to set the input data to d_u
        self.pos_encoder = PositionalEncoding(self.d_model)

        # the encoder itself (takes in (temporal, batch_size, d_model))
        self.encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.n_heads)

    def forward(self, x: Tensor) -> Tensor:
        # shape the data for the convolution layers (https://discuss.pytorch.org/t/how-to-apply-temporal-conv-on-1d-data/135363)
        x = torch.permute(x, (0, 2, 1))  # only swap the rows and columns and not the batch ([batch_size, embedding_dim, seq_len])
        x = self.conv1(x)
        x = self.conv2(x)

        # put the data through the embedding layer
        x = torch.permute(x, (0, 2, 1))  # swap the rows and columns back ([batch_size, seq_len, embedding_dim])
        x = self.fc(x)

        # reshape the data for the positional encoder ([seq_len, batch_size, embedding_dim])
        x = torch.permute(x, (1, 0, 2))
        x = self.pos_encoder(x)
        x = self.encoder_layer(x)
        # reshape the data back to for the representation ([batch_size, seq_len, embedding_dim])
        x = torch.permute(x, (1, 0, 2))
        return x


class DetectionLayer(nn.Module):
    """_summary_

    """

    def __init__(self, d_model: int, dropout: float = 0.2, use_std: bool = False):
        """
        :param d_model: The dimension of the unimodal representation (feature space)
        :type d_model: int
        :param dropout: The Dropout of the detection layer, defaults to 0.2
        :type dropout: float, optional
        :param use_std: Whether to use global average pooling or global standard deviation pooling, defaults to False
        :type dropout: bool, optional
        """
        super().__init__()
        self.d_model = d_model
        self.p_dropout = dropout
        self.use_std = use_std

        # self.gap = nn.AvgPool1d(self.d_model)
        self.dropout = nn.Dropout(self.p_dropout)
        self.fc = nn.Linear(self.d_model, 2)  # output to 2 neurons since softmax
        self.softmax = nn.Softmax(1)
    
    def forward(self, x):
        # x = x.transpose(1, 2) # only swap the rows and columns and not the batch ([batch_size, embedding_dim, seq_len])
        # apply the pooling
        if self.use_std:
            x = torch.std(x, 1)
        else:
            x = torch.mean(x, 1)

        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x,)
        return x


class UnimodalDVlogModel(nn.Module):
    """_summary_
    
    """

    def __init__(self, data_shape: tuple[int, int], d_model: int = 256, n_heads: int = 8, use_std: bool = False):
        """
        :param data_shape: The shape of the input data
        :type data_shape: tuple[int, int]
        :param d_model: The dimension of the encoder representation (d_u in the paper), defaults to 256
        :type d_model: int, optional
        :param n_heads: The number of attention heads in the encoder, defaults to 8
        :type n_heads: int, optional
        :type dropout: bool, optional
        :param use_std: Whether to use global average pooling or global standard deviation pooling, defaults to False
        :type use_std: bool, optional
        """
        super().__init__()
        # 
        self.data_shape = data_shape
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_std = use_std

        self.encoder = UnimodalTransformerEncoder(self.data_shape, self.d_model, n_heads=self.n_heads)
        self.detection_layer = DetectionLayer(self.d_model, use_std=use_std)

    def forward(self, x):
        x = self.encoder(x)
        x = self.detection_layer(x)
        return x