import torch
from torch import Tensor, nn
from torch.nn import functional as F
from types_helpers import EncoderOutput, LossOutput, ModelOutput


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) class.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input data.
    hidden_dim : int
        Dimensionality of the hidden layer.
    latent_dim : int
        Dimensionality of the latent space.
    """

    name = "VAE"

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),  # Swish activation function
            # nn.Linear(hidden_dim // 4, hidden_dim // 8),
            # nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 4, 2 * latent_dim),  # 2 for mean and variance.
        )
        self.softplus = nn.Softplus()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4),
            nn.SiLU(),  # Swish activation function
            # nn.Linear(hidden_dim // 8, hidden_dim // 4),
            # nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: Tensor, eps: float = 1e-8):
        """
        Encodes the input data into the latent space.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        eps : float, default=1e-8
            Small value to avoid numerical instability.

        Returns
        -------
        z : torch.distributions.MultivariateNormal
            Normal distribution of the encoded data.
        """
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)

        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)

    def reparameterize(self, dist):
        """
        Reparameterizes the encoded data to sample from the latent space.

        Parameters
        ----------
        dist : torch.distributions.MultivariateNormal
            Normal distribution of the encoded data.

        Returns
        -------
        sample : torch.Tensor
            Sampled data from the latent space.
        """
        return dist.rsample()

    def decode(self, z: Tensor) -> Tensor:
        """
        Decodes the data from the latent space to the original input space.

        Parameters
        ----------
        z : torch.Tensor)
            Data in the latent space.

        Returns
        -------
        reconstruction : torch.Tensor
            Reconstructed data in the original input space.
        """
        return self.decoder(z)

    def forward(self, x: Tensor):
        """
        Performs a forward pass of the VAE.

        Parameters
        ----------
        x : torch.Tensor
            Input data.
        compute_loss : bool
            Whether to compute the loss or not.

        Returns
        -------
            ModelOutput: Model output dataclass.
        """
        dist = self.encode(x)
        z = self.reparameterize(dist)
        recon_x = self.decode(z)

        # # compute loss terms
        # loss_recon = F.binary_cross_entropy(recon_x, x + 0.5, reduction="none").sum(-1).mean()
        # std_normal = torch.distributions.MultivariateNormal(
        #     torch.zeros_like(z, device=z.device),
        #     scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
        # )
        # loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()

        # loss = loss_recon + loss_kl

        return ModelOutput(output=recon_x, input=x, encoded=dist, latents=z)


class VanillaVAE(nn.Module):
    """
    Implementation of the classical VAE

    Parameters
    ----------
    latent_dim: int
        dimension of the latent space
    encoder: nn.Module
        implements the Encoder part
    full_decoder: nn.Module
        implements the Decoder part
    flatten: nn.Module
        einops Rearrange layer to ease the pre process of the latent work
    fc_mu: nn.Module
        linear Module which "learns" the mean vector of the latents distribution
    fc_var: nn.Module
        linear Module which "learns" the log-var vector of the latents distribution
    """

    name = "VanillaVAE"

    def __init__(
        self, in_channels: int, embed_dim: int, input_dim: int, hidden_dims: list[int] = None, verbose: bool = False
    ):
        super(VanillaVAE, self).__init__()

        self.latent_dim = embed_dim
        self.input_dim = input_dim
        self.in_channels = in_channels
        self.verbose = verbose
        self.kld_weight = 0.0

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

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
        )

        self._init_weights()

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
        Encodes the input and returns mu and log_var.
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
        Decodes the latent representation z.
        """
        x = self.decoder_input(z)
        if self.verbose:
            (f"decoder input result: {x.shape}  [{x.min()}, {x.max()}]")

        x = x.view(-1, 256, 2, 2)
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

    def forward(self, x):
        """
        Forward pass through the VAE.
        """
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
        loss_reconstruction = F.mse_loss(output["output"], output["input"])
        mu = output["encoded"]["mu"]
        log_var = output["encoded"]["log_var"]  # torch.clamp(output["encoded"]["log_var"], min=-10, max=10)
        # log_var_exp = (log_var + 1e-8).exp()
        # kld_loss = torch.mean(
        #     -0.5 * torch.sum(input=1 + log_var - mu ** 2 - log_var_exp, dim=1),
        #     dim=0,
        # )  # TODO move this to its own fn

        loss_kld = -0.5 * torch.mean(torch.sum(1 + log_var - mu**2 - torch.exp(log_var), dim=-1))

        loss = loss_reconstruction + self.kld_weight * loss_kld
        return LossOutput(
            loss=loss,
            reconstruction_loss=loss_reconstruction.detach(),
            kld_loss=-loss_kld.detach(),
        )

        # reconstruction_function = nn.MSELoss(reduction='sum')
        # MSE = reconstruction_function(recon_x, x)

        # # https://arxiv.org/abs/1312.6114 (Appendix B)
        # # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        # KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        # KLD = torch.sum(KLD_element).mul_(-0.5)

        # return MSE + beta*KLD

    def _init_weights(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        print("layer weight initialization complete")
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
