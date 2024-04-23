import torch

from torch import nn
from model import UnimodalTransformerEncoder, DetectionLayer


class CrissCrossAttentionModule(nn.Module):

    def __init__(self, d_model: int = 256, n_heads: int = 8):
        """
        :param d_model: The dimension of the encoder representation (d_u in the paper), defaults to 256
        :type d_model: int, optional
        :param n_heads: The number of attention heads in the encoder, defaults to 8
        :type n_heads: int, optional
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # setup the first cross attention blocks
        self.left_attention_block = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_heads, batch_first=True)
        self.middle_attention_block = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_heads, batch_first=True)
        self.right_attention_block = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_heads, batch_first=True)

        # layer norm
        self.layer_norm = nn.LayerNorm(self.d_model)

        # the last transformer before the multimodal representation
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model*3, nhead=self.n_heads, batch_first=True)

    def forward(self, first_input, second_input, third_input):
        # inputs representation shape: [batch_size, seq_len, input_dim]
        U_l, U_m, U_r = first_input, second_input, third_input

        # perform cross-attention
        left_attent, _ = self.left_attention_block(query=U_l, key=U_m, value=U_r)
        mid_attent, _ = self.left_attention_block(query=U_m, key=U_r, value=U_l)
        right_attent, _ = self.right_attention_block(query=U_r, key=U_l, value=U_m)
        
        # add residual and layer normalization
        U_l = left_attent + U_l
        U_l = self.layer_norm(U_l)

        U_m = mid_attent + U_m
        U_m = self.layer_norm(U_m)

        U_r = right_attent + U_r
        U_r = self.layer_norm(U_r)

        # fuse cross-modal information
        U_av = torch.cat((U_l, U_m, U_r), 2)

        # perform last multimodal encoder
        z = self.encoder_layer(U_av)
        return z


class TrimodalDVlogModel(nn.Module):
    """_summary_
    
    """

    def __init__(self, first_shape: tuple[int, int], second_shape: tuple[int, int], third_shape: tuple[int, int], d_model: int = 256, uni_n_heads: int = 8, cross_n_heads: int = 16, use_std: bool = False):
        """
        :param first_shape: The input shape of the first input feature
        :type first_shape: tuple[int, int]
        :param second_shape: The input shape of the second input feature
        :type second_shape: tuple[int, int]
        :param third_shape: The input shape of the third input feature
        :type third_shape: tuple[int, int]
        :param d_model: The dimension of the encoder representation (d_u in the paper), defaults to 256
        :type d_model: int, optional
        :param uni_n_heads: The number of attention heads in the unimodal encoders, defaults to 8
        :type uni_n_heads: int, optional
        :param cross_n_heads: The number of attention heads in the cross-attention module, defaults to 16
        :type cross_n_heads: int, optional
        :param dropout: , defaults to 
        :type dropout: bool, optional
        :param use_std: Whether to use global average pooling or global standard deviation pooling, defaults to False
        :type use_std: bool, optional
        """
        super().__init__()
        # 
        self.first_input_shape = first_shape
        self.second_input_shape = second_shape
        self.third_input_shape = third_shape

        self.d_model = d_model
        self.uni_n_heads = uni_n_heads
        self.cross_n_heads = cross_n_heads
        self.use_std = use_std

        # setup both encoders
        self.first_encoder = UnimodalTransformerEncoder(self.first_input_shape, self.d_model, n_heads=self.uni_n_heads)
        self.second_encoder = UnimodalTransformerEncoder(self.second_input_shape, self.d_model, n_heads=self.uni_n_heads)
        self.third_encoder = UnimodalTransformerEncoder(self.third_input_shape, self.d_model, n_heads=self.uni_n_heads)

        # setup the cross attention and prediction layer
        self.tricrossattention = CrissCrossAttentionModule(self.d_model, n_heads=self.cross_n_heads)
        self.detection_layer = DetectionLayer(self.d_model*2, use_std=use_std)

    def forward(self, features):
        # extract both features
        x_1, x_2, x_3 = features

        # run the features through the encoders
        x_1 = self.first_encoder(x_1)
        x_2 = self.second_encoder(x_2)
        x_3 = self.third_encoder(x_3)

        # apply the cross attention and the detection layer
        x = self.tricrossattention(x_1, x_2, x_3)
        x = self.detection_layer(x)
        return x