import torch
from torch import nn, Tensor, optim
from torch.nn import functional as F
import torch.nn.init as init
from typing import List
import lightning as L

from utils.types_helpers import EncoderOutput, ModelOutput, LossOutput


class VanillaVAE(L.LightningModule):
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

    def __init__(self, in_channels, latent_dim, exp_params, hidden_dims=None, **kwargs):
        super(VanillaVAE, self).__init__()

        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.exp_params = exp_params
        self.last_conv_size = 4

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims
        decoder_hidden_dims = self.hidden_dims[::-1]

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # CWH = [32, 64, 64]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # CWH = [64, 32, 32]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),  # CWH = [128, 16, 16]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),  # CWH = [256, 8, 8]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),
            nn.Flatten(),  # 16384
        )
        # modules = []
        # for i, h_dim in enumerate(hidden_dims):
        #     stride = 2
        #     # if i == len(hidden_dims) - 1:
        #     #     stride = 1  # Set stride=1 for the last layer to prevent size reduction
        #     modules.append(
        #         nn.Sequential(
        #             nn.Conv2d(
        #                 in_channels,
        #                 out_channels=h_dim,
        #                 kernel_size=4,
        #                 stride=2,
        #                 padding=1,
        #             ),
        #             nn.BatchNorm2d(h_dim),
        #             nn.LeakyReLU(),
        #         )
        #     )
        #     in_channels = h_dim

        # self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_var = nn.Linear(256 * 8 * 8, latent_dim)
        # self.fc_mu = nn.Linear(2048, latent_dim)
        # self.fc_var = nn.Linear(2048, latent_dim)
        # self.flatten = nn.Flatten()

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # CWH = [128, 16, 16]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # CWH = [64, 32, 32]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # CWH = [32, 64, 64]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),  # CWH = [3, 128, 128]
            nn.BatchNorm2d(3),
            nn.LeakyReLU(True),
            nn.Tanh(),
        )
        # modules = []
        # # self.decoder_input = nn.Linear(latent_dim, 2048)
        # # self.decoder_input = nn.Linear(latent_dim, self.hidden_dims[-1] * 4)
        # self.decoder_input = nn.Linear(
        #     latent_dim, self.hidden_dims[-1] * self.last_conv_size**2
        # )
        # for i in range(len(decoder_hidden_dims) - 1):
        #     stride = 2
        #     padding = 1
        #     if i == len(decoder_hidden_dims) - 2:
        #         # Adjust the padding for the second last layer
        #         padding = 3
        #     modules.append(
        #         nn.Sequential(
        #             nn.ConvTranspose2d(
        #                 decoder_hidden_dims[i],
        #                 decoder_hidden_dims[i + 1],
        #                 # kernel_size=4, MNIST
        #                 kernel_size=3,
        #                 stride=stride,
        #                 padding=padding,
        #                 output_padding=0,
        #             ),
        #             nn.BatchNorm2d(decoder_hidden_dims[i + 1]),
        #             nn.LeakyReLU(),
        #         )
        #     )
        # self.decoder = nn.Sequential(*modules)

        # self.final_layer = nn.Sequential(
        #     nn.Conv2d(128, out_channels=3, kernel_size=3, padding=1),
        #     nn.Tanh(),
        # )
        # modules = []
        # hidden_dims.reverse()
        # for i in range(len(hidden_dims) - 1):
        #     modules.append(
        #         nn.Sequential(
        #             nn.ConvTranspose2d(
        #                 hidden_dims[i],
        #                 hidden_dims[i + 1],
        #                 kernel_size=3,
        #                 stride=2,
        #                 padding=1,
        #                 output_padding=1,
        #             ),
        #             nn.BatchNorm2d(hidden_dims[i + 1]),
        #             nn.LeakyReLU(),
        #         )
        #     )

        # self.decoder = nn.Sequential(*modules)

        # self.final_layer = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         hidden_dims[-1],
        #         hidden_dims[-1],
        #         kernel_size=3,
        #         stride=2,
        #         padding=1,
        #         output_padding=1,
        #     ),
        #     nn.BatchNorm2d(hidden_dims[-1]),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(hidden_dims[-1], out_channels=1, kernel_size=3, padding=1),
        #     nn.Tanh(),
        # )

    def encode(self, input: Tensor) -> EncoderOutput:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # print(f"encoder output: {result.shape}")
        # print(f"mu: {self.fc_mu}")

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
        # result = result.view(-1, 512, 2, 2)
        # result = result.view(-1, 2048, self.last_conv_size, self.last_conv_size)
        # print(f"decoder input: {result.shape}")

        result = self.decoder(result)
        # print(f"decoder output: {result.shape}")
        # result = self.final_layer(result)
        # print(f"final output: {result.shape}")
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
        return ModelOutput(
            output=self.decode(z), input=input, encoded=encoding, latents=z
        )

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
            -0.5
            * torch.sum(
                1 + log_var - output_model["encoded"]["mu"] ** 2 - log_var_exp, dim=1
            ),
            dim=0,
        )  # TODO move this to its own fn
        loss = recons_loss + self.exp_params["kld_weight"] * kld_loss
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

    ########################### lightning functions ###########################
    def training_step(self, batch, batch_idx):
        x, _ = batch
        output = self.forward(x)
        loss = self.loss(output)
        self.log("train_loss", loss["loss"], prog_bar=True)
        self.log("reconstruction_loss", loss["reconstruction_loss"], prog_bar=False)
        self.log("kld_loss", loss["kld_loss"], prog_bar=False)
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        output = self.forward(x)
        loss = self.loss(output)
        self.log("val_loss", loss["loss"], prog_bar=True)
        self.log(
            "val_reconstruction_loss",
            loss["reconstruction_loss"],
            prog_bar=False,
        )
        self.log("val_kld_loss", loss["kld_loss"], prog_bar=False)
        return loss["loss"]

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.exp_params["learning_rate"],
            weight_decay=self.exp_params["weight_decay"],
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=self.exp_params["scheduler_gamma"]
        )
        return {
            "optimizer": optimizer,
            "gradient_clip_val": 1.0,
            "scheduler": scheduler,
        }

    def on_after_backward(self):
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
        self.log("grad_norm", grad_norm)

    def on_epoch_end(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)  # type: ignore

    ################################# utility #################################
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
