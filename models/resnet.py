import torch.nn as nn


class ResnetBlock(nn.Module):
    """
    A class that implements a residual block for a ResNet model.

    Args:
        dim (int): The number of input and output channels.
        padding_type (str): The type of padding to use. Options are "reflect", "replicate", or "zero".
        norm_layer (nn.Module): The normalization layer to use.
        use_dropout (bool): Whether to use dropout or not.
        use_bias (bool): Whether to use bias or not.
        dim2 (int, optional): The number of channels for the second convolutional layer. Defaults to None.
    """

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, dim2=None):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """
        Builds the convolutional block with the specified parameters.

        Args:
            dim (int): The number of input and output channels.
            padding_type (str): The type of padding to use. Options are "reflect", "replicate", or "zero".
            norm_layer (nn.Module): The normalization layer to use.
            use_dropout (bool): Whether to use dropout or not.
            use_bias (bool): Whether to use bias or not.

        Returns:
            nn.Sequential: The convolutional block.
        """
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """
        Implements the forward pass of the ResnetBlock module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # print(x.shape)
        # print(self.conv_block(x).shape)
        out = x + self.conv_block(x)
        return out
