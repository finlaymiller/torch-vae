import torch
from torch import Tensor, nn
from torch.nn import functional as F
from types_helpers import EncoderOutput, LossOutput, ModelOutput


class VanillaVAE(nn.Module):
    name = "VanillaVAE"

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        input_dim: int,
        hidden_dims: list[int] = None,
        kld_weight: float = 1.0,
        verbose: bool = False,
    ):
        super(VanillaVAE, self).__init__()

        self.latent_dim = embed_dim
        self.input_dim = input_dim
        self.in_channels = in_channels
        self.verbose = verbose
        self.kld_weight = kld_weight

        # Default hidden dimensions
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims

        # Compute the output size after the encoder
        self.last_conv_size = 4  # self._compute_conv_output_size(input_dim, len(hidden_dims))
        if self.verbose:
            print(f"last convolution will have size {self.last_conv_size}")
        self.flattened_size = self.last_conv_size * hidden_dims[-1]
        if self.verbose:
            print(f"final flattened size {self.flattened_size}")

        # Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self._init_weights(self.encoder, "encoder")

        # Latent space
        self.fc_mu = nn.Linear(self.flattened_size, self.latent_dim)
        self.fc_var = nn.Linear(self.flattened_size, self.latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(self.latent_dim, self.flattened_size)
        hidden_dims.reverse()

        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )
        self.decoder = nn.Sequential(*modules)
        self._init_weights(self.decoder, "decoder")

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        self._init_weights(self.final_layer, "final layer")

    def _compute_conv_output_size(self, dim: int, num_layers: int, kernel: int = 3, stride: int = 2, padding: int = 1):
        """
        Computes the size of the output after a series of convolutional layers.

        Parameters
        ----------
        height : int
            Height of the input image.
        width : int
            Width of the input image.
        num_layers : int
            Number of convolutional layers.

        Returns
        -------
        size : int
            Size of the final feature map (assumes square feature maps).
        """
        for _ in range(num_layers):
            dim = (dim - kernel + stride * padding) // stride + 1
        return dim

    def encode(self, x: Tensor) -> EncoderOutput:
        """
        Encodes the input by passing through the convolutional network
        and outputs the latent variables.

        Parameters
        ----------
        x : Tensor
            Input tensor [N x C x H x W]

        Returns
        -------
        mu : Tensor
            mean of latent variables
        log_var : Tensor
            log variance of latent variables
        pre_latents : Tensor
            result of encoding before ???
        """
        if self.verbose:
            print(f"encoder input: {x.shape} [{x.min()}, {x.max()}]")

        x_hat = self.encoder(x)
        if self.verbose:
            print(f"encoder result: {x_hat.shape} [{x_hat.min()}, {x_hat.max()}]")

        x_hat = x_hat.flatten(start_dim=1)
        if self.verbose:
            print(f"flatten result: {x_hat.shape} [{x_hat.min()}, {x_hat.max()}]")

        mu = self.fc_mu(x_hat)
        if self.verbose:
            print(f"mu: {mu.shape} [{mu.min()}, {mu.max()}]")

        log_var = self.fc_var(x_hat)
        if self.verbose:
            print(f"log_var: {log_var.shape} [{log_var.min()}, {log_var.max()}]")

        return EncoderOutput(mu=mu, log_var=log_var, pre_latents=x_hat)

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent variables onto the image space by de-convolving,
        matching the convolutions performed by the encoder.

        Parameters
        ----------
        z : Tensor
            Latent variable [B x D]

        Returns
        -------
        result : Tensor
            [B x C x H x W]
        """
        x = self.decoder_input(z)
        if self.verbose:
            (f"decoder input result: {x.shape}  [{x.min()}, {x.max()}]")

        x = x.view(-1, 256, 2, 2)  # TODO: get proper second dime size from hidden layers
        x = self.decoder(x)
        if self.verbose:
            print(f"decoder result: {x.shape}  [{x.min()}, {x.max()}]")

        x = self.final_layer(x)
        if self.verbose:
            print(f"final layer result: {x.shape}  [{x.min()}, {x.max()}]")

        return x

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: torch.Tensor) -> ModelOutput:
        encoding = self.encode(x)
        z = self.reparameterize(encoding["mu"], encoding["log_var"])
        return ModelOutput(output=self.decode(z), input=x, encoded=encoding, latents=z)

    def loss(self, output: ModelOutput):
        r"""
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}

        Parameters
        ----------
        input : Tensor
        output : Tensor
        logvar : Tensor
        mu : Tensor

        Returns
        -------
        loss : Tensor
        reconstruction loss : Tensor
        KLD loss : Tensor
        """
        loss_reconstruction = F.binary_cross_entropy(output["output"], output["input"])
        mu = output["encoded"]["mu"]
        log_var = output["encoded"][
            "log_var"
        ]  # option to clamp: torch.clamp(output["encoded"]["log_var"], min=-10, max=10)

        loss_kld = -0.5 * torch.mean(torch.sum(1 + log_var - mu**2 - torch.exp(log_var), dim=-1))

        loss = loss_reconstruction + self.kld_weight * loss_kld

        # option: increase kld_weight over time
        # self.kld_weight = min(self.kld_weight * 1.005, 1.0)

        return LossOutput(
            loss=loss,
            reconstruction_loss=loss_reconstruction.detach(),
            kld_loss=-loss_kld.detach(),
        )

    def _init_weights(self, module: nn.Sequential, name: str):
        # encoder
        for m in module.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        print(f"{name} layer weight initialization complete")
        for name, param in m.named_parameters():
            if "weight" in name:
                print(f"{name}:")
                print(f"  Shape: {param.shape}")
                print(f"  Mean: {param.data.mean().item():.6f}")
                print(f"  Std Dev: {param.data.std().item():.6f}")
                print(f"  Min: {param.data.min().item():.6f}")
                print(f"  Max: {param.data.max().item():.6f}")
                print(f"  Norm: {param.data.norm().item():.6f}\n")

    # utility -----------------------------------------------------------------
    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)["output"]
