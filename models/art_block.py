import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .resnet import ResnetBlock


class ART_block(nn.Module):
    def __init__(self, config, input_dim, img_size=512, transformer=None):
        super(ART_block, self).__init__()
        self.transformer = transformer
        self.config = config
        ngf = 4
        mult = 4
        use_bias = False
        norm_layer = nn.BatchNorm2d
        padding_type = "reflect"
        if self.transformer:
            # Downsampling block
            model = [
                nn.Conv2d(
                    ngf * 4,
                    ngf * 8,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * 8),
                nn.ReLU(inplace=True),
            ]
            model += [
                nn.Conv2d(
                    ngf * 8,
                    1024,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(1024),
                nn.ReLU(inplace=True),
            ]
            setattr(self, "downsample", nn.Sequential(*model))
            # Patch embeddings
            self.embeddings = Embeddings(
                config,
                img_size=img_size,
            )
            # Upsampling block
            model = [
                nn.ConvTranspose2d(
                    self.config.hidden_size,
                    ngf * 8,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * 8),
                nn.ReLU(True),
            ]
            model += [
                nn.ConvTranspose2d(
                    ngf * 8,
                    ngf * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * 4),
                nn.ReLU(True),
            ]
            setattr(self, "upsample", nn.Sequential(*model))
            # Channel Compression
            self.cc = ChannelCompression(ngf * 8, ngf * 4)
        # Residual CNN
        model = [
            ResnetBlock(
                ngf * mult,
                padding_type=padding_type,
                norm_layer=norm_layer,
                use_dropout=False,
                use_bias=use_bias,
            )
        ]
        setattr(self, "residual_cnn", nn.Sequential(*model))

    def forward(self, x):
        if self.transformer:
            ## Downsampling
            down_sampled = self.downsample(x)

            ## Embedding + Positional Encodings
            embedding_output = self.embeddings(down_sampled)

            ## Transformer Encoder
            transformer_out, attn_weights = self.transformer(embedding_output)

            ## Patch Deflattening
            (
                B,
                n_patch,
                hidden,
            ) = (
                transformer_out.size()
            )  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
            h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
            transformer_out = transformer_out.permute(0, 2, 1)  # [B, hidden, n_patch]
            transformer_out = transformer_out.contiguous().view(B, hidden, h, w)

            ## Upsampling
            upsampled = self.upsample(transformer_out)

            ## Concatenating transformer output and resnet output
            x = torch.cat([upsampled, x], dim=1)

            ## Channel Compression
            x = self.cc(x)

        ## Residual CNN
        x = self.residual_cnn(x)

        return x


class ChannelCompression(nn.Module):
    """
    A class that performs channel compression on the input tensor.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        stride (int, optional): The stride value. Defaults to 1.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ChannelCompression, self).__init__()

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = None

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        """
        Compresses the channels of the input tensor.

        Args:
            input_tensor (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The compressed tensor.
        """

        out = self.block(x)
        out += x if self.skip is None else self.skip(x)
        out = F.relu(out)
        return out


class Embeddings(nn.Module):
    def __init__(self, config, img_size):
        super(Embeddings, self).__init__()
        self.config = config
        img_size = _pair(img_size)
        grid_size = config.patches["grid"]
        patch_size = (
            img_size[0] // 16 // grid_size[0],
            img_size[1] // 16 // grid_size[1],
        )
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        n_patches = (img_size[0] // patch_size_real[0]) * (
            img_size[1] // patch_size_real[1]
        )
        in_channels = 1024
        # Learnable patch embeddings
        self.patch_embeddings = nn.Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        # learnable positional encodings
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, n_patches, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.positional_encoding
        embeddings = self.dropout(embeddings)
        return embeddings
