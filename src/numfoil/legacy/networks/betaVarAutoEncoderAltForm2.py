import torch
from torch import nn
from torchinfo import summary


class betaVarAutoEncoder(nn.Module):
    def __init__(self, *, input_dim: int, hyperParam, actFun=nn.LeakyReLU(), dropRate=0):
        """A basic autoencoder class.
        Alternate formulation using mm.Sequential instead of nn.ModuleList.

        (``self, *, ...'' makes args into kwargs without defaults)

        Args:
            input_dim (int): size of the input tensor (input layer)
            h_dim (list): size of the hidden layers, ``len(h_dim)'' is number of hidden layers
            z_dim (int): size of latent tensor (bottleneck layer)
            actFun (nn.functional): activation function for all but the last layer [DEFUALT=nn.LeakyReLU()]
            dropRate (float): Dropout rate [DEFAULT=0]
        """
        super().__init__()
        self.input_dim = input_dim      # dimension of the input layer
        self.hyperParam = hyperParam
        h_dim = hyperParam.H_DIM              # dimensions of the hidden layers
        z_dim = hyperParam.Z_DIM              # size of latent vector
        self.actFun = actFun            # activation function
        self.dropRate = hyperParam.dropRate        # dropout rate

        layer_dim = [input_dim] + h_dim + [z_dim]   # lengths of all encoder layers (decoder is reverse)

        ### encoder network ###
        self.encoder = nn.Sequential()
        for i in range(len(h_dim)):
            self.encoder.add_module(f"layer_{i}", nn.Linear(layer_dim[i], layer_dim[i+1]))  # add linear layer
            self.encoder.add_module(f"activation_{i}", self.actFun)                         # add activation function
            self.encoder.add_module(f"dropout_{i}", nn.Dropout(dropRate))                   # add dropout layer
        del self.encoder[-1]            # remove dropout layer completely

        self.meanLayer = nn.Linear(h_dim[-1], z_dim)
        self.logvarLayer = nn.Linear(h_dim[-1], z_dim)

        ### decoder network ###
        self.decoder = nn.Sequential()
        for i, j in enumerate(reversed(range(len(h_dim)+1))):
            self.decoder.add_module(f"layer_{i}", nn.Linear(layer_dim[j+1], layer_dim[j]))  # add linear layer
            self.decoder.add_module(f"activation_{i}", self.actFun)                         # add activation function
            self.decoder.add_module(f"dropout_{i}", nn.Dropout(dropRate))                   # add dropout layer
        # self.decoder[-2] = nn.Tanh()    # change last activation to get proper output format
        del self.decoder[-1]            # remove dropout layer completely

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Use the encoder network to convert input into the latent space.
        Line by line:
            - Encode input x to intermediate output of hidden layers h
            - Pass through the final 2 output layers of the encoder to get mean and logvar
            - return mean and logvar of the latent vector

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            mean (torch.Tensor): the output of the final layer/activation function
            logvar (torch.Tensor): the output of the final layer/activation function
        """
        h = self.encoder(x)
        mean, logvar = self.meanLayer(h), self.logvarLayer(h)
        return mean, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """reparameterization trick

        Args:
            mean (torch.Tensor): latent mean vector, same size as z
            logvar (torch.Tensor): latent logvar vector, same size as z

        Returns:
            z (torch.Tensor): latent vector
        """
        std = torch.exp(0.5 * logvar)   # standard deviation
        eps = torch.randn_like(std)     # epsilon (random values)
        z = mean + eps * std            # reparameterization trick
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the entire network.
            - encode input to latent space distribution defined by mean and logvar
            - reparameterization trick to obtain latent vector
            - decode latent vector back to original input form

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output container  # //output tensor, reconstructed input
        """
        mean, logvar = self.encode(x)           # encode input to latent distributions
        z = self.reparameterize(mean, logvar)   # apply reparameterization to obtain latent vector
        # res = Container(x_=self.decoder(z), z=z, mean=mean, logvar=logvar)
        return self.decoder(z), z, mean, logvar


if __name__ == "__main__":
    net = betaVarAutoEncoder(
        input_dim=199,
        h_dim=[64, 32],
        z_dim=4,
        dropRate=0.5
        )
    print(str(summary(net, input_size=[16, 199])))
