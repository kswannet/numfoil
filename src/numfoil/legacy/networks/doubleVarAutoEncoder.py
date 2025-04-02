import torch
from torch import nn
from torchinfo import summary


class doubleVarAutoEncoder(nn.Module):
    def __init__(self, *, input_dim, h_dim, z_dim, dropRate):
        """A basic autoencoder class.
        (``self, *, ...'' makes args into kwargs without defaults)

        Args:
            input_dim (int): size of the input tensor (input layer)
            h_dim (list): size of the hidden layers, ``len(h_dim)'' is number of hidden layers
            z_dim (int): size of latent tensor (bottleneck layer)
            dropRate (float): Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim      # dimension of the input layer
        self.h_dim = h_dim              # dimensions of the hidden layers
        self.z_dim = z_dim
        self.dropRate = dropRate

        # some variables to facilitate network creation
        num_layers = len(h_dim) + 2                 # number of layers is number of hidden + input + output
        layer_dim = [input_dim] + h_dim + [z_dim]   # lengths of all layers

        # encoder network layers
        self.encoderLayers = nn.ModuleList(
            [nn.Linear(layer_dim[i], layer_dim[i+1]) for i in range(num_layers-2)]
        )
        self.sigmaLayer = nn.Linear(h_dim[-1], z_dim)
        self.muLayer = nn.Linear(h_dim[-1], z_dim)

        # decoder layers
        self.decoderLayers = nn.ModuleList(
            [nn.Linear(layer_dim[i+1], layer_dim[i]) for i in reversed(range(num_layers-1))]
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropRate)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """Use the encoder network to convert input into the latent space.
        Line by line:
            - For each layer in encoder list, except for the last one:
            - Pass tensor through the layer and apply the activation function.
            - Pass tensor through the final layer, and apply activation
              function.

        Final layer is kept separate in case a different, or no activation
        function is used on the output of the final layer

        Args:
            h (torch.Tensor): input tensor. Called h as it gets overwritten by
                intermediate results when passing through layers and act
                functions.

        Returns:
            torch.Tensor: the output of the final layer/activation function
        """
        for layer in self.encoderLayers:                    # For each layer in encoder list, except for the last one
            # h = self.sigmoid(layer(h))                    # Pass tensor through the layer and apply the activation function
            h = self.dropout(self.sigmoid(layer(h)))        # Pass tensor through the layer, activation function, and dropout
        sigma, mu = self.sigmaLayer(h), self.muLayer(h)
        return sigma, mu

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Use the decoder network to convert latent vector to original format.
        Line by line:
            - For each layer in decoder list, except for the last one:
            - Pass tensor through the layer and apply the activation function.
            - Pass tensor through the final layer, and apply activation
              function.

        Final layer is kept separate in case a different, or no activation
        function is used on the output of the final layer

        Args:
            h (torch.Tensor): input latent tensor. Called h as it gets
                overwritten by intermediate results when passing through layers
                and act functions.

        Returns:
            torch.Tensor: the output of the final layer/activation function
        """
        for layer in self.decoderLayers[:-1]:               # For each layer in encoder list, except for the last one
            # h = self.sigmoid(layer(h))                      # Pass tensor through the layer and apply the activation function
            h = self.dropout(self.sigmoid(layer(h)))        # Pass tensor through the layer, activation function, and dropout
        return self.tanh(self.decoderLayers[-1](h))         # final layer and (optional) activation function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the entire network.
            - encode input to latent space
            - decode latent space back to original format

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor, reconstructed input
        """
        sigma, mu = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z = mu + epsilon * sigma
        return self.decode(z), sigma, mu


if __name__ == "__main__":
    net = betaVarAutoEncoder(
        input_dim=199,
        h_dim=[64, 32],
        z_dim=4,
        dropRate=0.3
        )
    print(str(summary(net, input_size=[16, 199])))
