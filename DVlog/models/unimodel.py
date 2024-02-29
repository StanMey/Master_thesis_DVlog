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

    def __init__(self, d_model: int, kernel_size: int = 4, stride: int = 2, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_heads = n_heads

        # input
        self.conv1 = torch.nn.Conv1d(self.d_model, self.d_model, self.kernel_size, stride=self.stride)
        self.conv2 = torch.nn.Conv1d(self.d_model, self.d_model, self.kernel_size, stride=self.stride)
        self.pos_encoder = PositionalEncoding(self.d_model)

        # the encoder itself (takes in (batch_size, temporal, d_model))
        self.encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.n_heads, batch_first=True)

    def forward(self, x):
        # shape the data for the convolution layers (https://discuss.pytorch.org/t/how-to-apply-temporal-conv-on-1d-data/135363)
        print(x.shape)
        x = x.transpose(1, 2) # only swap the rows and columns and not the batch
        print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)

        #TODO reshape the data for the positional encoder
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
        x = x.squeeze() # flatten to a row so the matrix multiplication works
        x = self.dropout(x)
        x = self.softmax(x)
        return x


class UnimodalDVlogModel(nn.Module):
    """_summary_
    
    """

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        # 
        self.d_model = d_model
        self.n_heads = n_heads

        self.encoder = UnimodalTransformerEncoder(self.d_model, n_heads=self.n_heads)
        self.detection_layer = DetectionLayer(self.d_model)

    def forward(self, x):
        x = self.encoder(x)
        x = self.detection_layer(x)
        return x