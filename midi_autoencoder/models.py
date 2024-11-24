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
        self,
        in_channels,
        latent_dim,
        learning_rate: float = 0.001,
        weight_decay: float = 0.00001,
        kld_weight: float = 0.00025,
        scheduler_gamma: float = 0.1,
        hidden_dims: list[int] = None,
        **kwargs,
    ):
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.kld_weight = kld_weight
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.last_conv_size = 4

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims

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
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
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
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Upsample(size=(28, 28), mode="bilinear", align_corners=False),
        )
        hidden_dims.reverse()

    def encode(self, input: Tensor, eps: float = 1e-8) -> EncoderOutput:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.

        Parameters
        ----------
        input : Tensor
            Input tensor to encoder [N x C x H x W]
        eps : float, default=1e-8
            Small value to avoid numerical instability.

        Returns
        -------
        encoder_output : EncoderOutput
            List of latent codes, mu, and logvar
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        mu = self.fc_mu(result)
        log_var = self.fc_var(result) + eps  # to avoid Inf

        return EncoderOutput(mu=mu, log_var=log_var, pre_latents=result)

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes onto the image space.

        Parameters
        ----------
        z : Tensor
            latent representation [B x D]

        Returns
        -------
        decoded : Tensor
            [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).

        Parameters
        ----------
        mu : Tensor
            Mean of the latent Gaussian [B x D]
        logvar : Tensor
            Standard deviation of the latent Gaussian [B x D]
        Returns
        -------
        sample : Tensor
            [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor) -> ModelOutput:
        encoding = self.encode(input)
        z = self.reparameterize(encoding["mu"], encoding["log_var"])
        return ModelOutput(output=self.decode(z), input=input, encoded=encoding, latents=z)

    def loss(self, output_model: ModelOutput):
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
        recons_loss = F.mse_loss(output_model["output"], output_model["input"])
        log_var = torch.clamp(output_model["encoded"]["log_var"], min=-10, max=10)
        log_var_exp = (log_var + 1e-8).exp()
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - output_model["encoded"]["mu"] ** 2 - log_var_exp, dim=1),
            dim=0,
        )  # TODO move this to its own fn
        loss = recons_loss + self.kld_weight * kld_loss
        return LossOutput(
            loss=loss,
            reconstruction_loss=recons_loss.detach(),
            kld_loss=-kld_loss.detach(),
        )

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
