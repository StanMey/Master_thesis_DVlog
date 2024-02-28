import math

import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    """_summary_
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

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

    def __init__(self, d_model: int, sequence_length: int, kernel_size: int = 4, stride: int = 4, n_heads: int = 6):
        super().__init__()
        self.d_model = d_model
        self.seq_length = sequence_length
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_heads = n_heads

        # input
        self.conv1 = torch.nn.Conv1d(self.d_model, self.d_model, self.kernel_size, stride=self.stride)
        self.conv2 = 1
        self.pos_encoder = PositionalEncoding(self.d_model)

        # the encoder itself
        self.encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.n_heads)

    def forward(self, x):
        x = torch.transpose(self.conv1(x))
        x = self.pos_encoder(x)
        x = self.encoder_layer(x)
        return x


class DetectionLayer(nn.Module):
    """_summary_

    """

    def __init__(self, d_model: int, dropout: float = 0.2):
        """_summary_

        :param d_model: The dimension of the unimodal representation (feature space)
        :type d_model: int
        :param dropout: The Dropout of the detection layer, defaults to 0.2
        :type dropout: float, optional
        """
        super().__init__()
        self.d_model = d_model
        self.p_dropout = dropout

        self.gap = nn.AvgPool1d(self.d_model)
        self.dropout = nn.Dropout(self.p_dropout)
        self.fc = nn.Linear(self.d_model, 2)  # output to 2 neurons since softmax
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x = self.gap(x)
        x = self.dropout(x)
        x = x.view(1, -1)  # flatten to a row so the matrix multiplication works
        x = self.softmax(x)
        return x


class UnimodalDVlogModel(nn.Module):
    """_summary_
    
    """

    def __init__(self, d_model: int, sequence_length: int):
        super().__init__()
        # 
        self.d_model = d_model
        self.seq_length = sequence_length

        self.encoder = UnimodalTransformerEncoder(self.d_model, self.seq_length)
        self.detection_layer = DetectionLayer()

    def forward(self, x):
        return x