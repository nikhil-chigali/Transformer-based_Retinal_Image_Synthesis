import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class Encoder(nn.Module):
    """
    A module in a neural network that performs encoding on input data.

    Args:
        config (dict): Configuration for the Encoder module.

    Attributes:
        layer (nn.ModuleList): List of Block modules that perform transformations.
        encoder_norm (nn.LayerNorm): Normalization layer to normalize the output.

    Example Usage:
        config = {
            'hidden_size': 512,
            'transformer': {
                'num_layers': 6
            }
        }
        encoder = Encoder(config)
        hidden_states = torch.randn(32, 512)
        output = encoder(hidden_states)
    """

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer.num_layers):
            layer = Block(config)
            self.layer.append(deepcopy(layer))

    def forward(self, hidden_states):
        """
        Performs the forward pass of the Encoder module.

        Args:
            hidden_states (torch.Tensor): Input hidden states.

        Returns:
            torch.Tensor: Final output after applying transformations.
        """
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = nn.MultiheadAttention(
            config.hidden_size,
            config.transformer.num_heads,
            config.transformer.attention_dropout_rate,
        )

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x, x, x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) neural network module.

    Args:
        config (dict): Configuration dictionary containing the following keys:
            - hidden_size (int): Size of the input and output tensors.
            - transformer (dict): Dictionary containing the following keys:
                - mlp_dim (int): Dimension of the intermediate MLP layer.
                - dropout_rate (float): Dropout rate for regularization.

    Attributes:
        fc1 (nn.Linear): First fully connected layer of the MLP.
        fc2 (nn.Linear): Second fully connected layer of the MLP.
        act_fn (function): Activation function (GELU) applied after the first linear transformation.
        dropout (nn.Dropout): Dropout layer applied after the activation function.

    Methods:
        _init_weights(): Initializes the weights of the linear layers.
        forward(x): Performs the forward pass of the MLP module.

    """

    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.transformer.mlp_dim)
        self.fc2 = nn.Linear(config.transformer.mlp_dim, config.hidden_size)
        self.act_fn = F.gelu
        self.dropout = nn.Dropout(config.transformer.dropout_rate)

        self._init_weights()

    def _init_weights(self):
        """
        Initializes the weights of the linear layers using Xavier uniform initialization for weights
        and normal distribution with a small standard deviation for biases.
        """
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        """
        Performs the forward pass of the MLP module.

        Args:
            x (torch.Tensor): Input tensor to be transformed.

        Returns:
            torch.Tensor: Transformed tensor after applying linear transformations, activation function,
                and dropout regularization.
        """
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
