import torch
from torch import nn
from torchinfo import summary


class AutoEncode(nn.Module):
    def __init__(self, *, input_dim, h_dim, z_dim, actFun=nn.LeakyReLU(), dropRate=0):
        """A basic autoencode class.
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
        self.h_dim = h_dim              # dimensions of the hidden layers
        self.z_dim = z_dim              # size of latent vector
        self.actFun = actFun            # activation function
        self.dropRate = dropRate        # dropout rate

        layer_dim = [input_dim] + h_dim + [z_dim]   # lengths of all layers

        # encode network
        self.encode = nn.Sequential()
        for i in range(len(h_dim)+1):
            self.encode.add_module(f"fcEnc_{i}", nn.Linear(layer_dim[i], layer_dim[i+1]))
            self.encode.add_module(f"actEnc_{i}", self.actFun)
            self.encode.add_module(f"dropEnc_{i}", nn.Dropout(dropRate))
        # self.decode[-1].p = 0          # remove dropout from output layer (drop latent variable)
        # del self.encode[-1]            # remove dropout layer completely

        # decode network
        self.decode = nn.Sequential()
        for i, j in enumerate(reversed(range(len(h_dim)+1))):
            self.decode.add_module(f"fcDec_{i}", nn.Linear(layer_dim[j+1], layer_dim[j]))
            self.decode.add_module(f"actEnc_{i}", self.actFun)
            self.decode.add_module(f"dropEnc_{i}", nn.Dropout(dropRate))
        self.decode[-2] = nn.Tanh()    # change last activation to get proper output format
        self.decode[-1].p = 0          # remove dropout from output layer
        # del self.decode[-1]            # remove dropout layer completely

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the entire network.
            - encode input to latent space
            - decode latent space back to original format

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor, reconstructed input
        """
        return self.decode(self.encode(x))
        # return self.encode(x)


if __name__ == "__main__":
    net = AutoEncode(
        input_dim=199,
        h_dim=[64, 32],
        z_dim=4,
        dropRate=0.3
        )
    print(str(summary(net, input_size=[16, 199])))
