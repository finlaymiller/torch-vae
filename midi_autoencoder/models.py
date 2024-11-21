import torch
import torch.nn.init as init
from torch import Tensor, nn
from torch.nn import functional as F
from types_helpers import EncoderOutput, LossOutput, ModelOutput


class VanillaVAE(nn.Module):
    """Implementation of the classical VAE
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

    def __init__(
        self,
        in_channels,
        latent_dim,
        learning_rate: float = 0.001,
        weight_decay: float = 0.00001,
        kld_weight: float = 0.00025,
        scheduler_gamma: float = 0.1,
        hidden_dims=None,
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

        # self.final_layer = nn.Sequential(
        # nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.BatchNorm2d(hidden_dims[-1]),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(hidden_dims[-1], out_channels=1, kernel_size=3, padding=1),
        #     nn.Tanh(),
        # )

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Upsample(size=(28, 28), mode="bilinear", align_corners=False),
        )

    def encode(self, input: Tensor) -> EncoderOutput:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        mu = self.fc_mu(result)
        log_var = self.fc_var(result) + 1e-8  # to avoid Inf

        return EncoderOutput(mu=mu, log_var=log_var, pre_latents=result)

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 256, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
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
        :param args:
        :param kwargs:
        :return:
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

    def _weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            init.ones_(m.weight)
            init.zeros_(m.bias)

    def on_after_backward(self):
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
        self.log("grad_norm", grad_norm)

    def on_epoch_end(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

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
