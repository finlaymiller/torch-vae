import os
import yaml
import argparse
import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from rich import print
from torchinfo import summary
import torchvision
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder, MNIST
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from model import VanillaVAE

IT = 7
IMG_PATH = f"logs/VanillaVAE/reconstructions/reconstruction_{IT}.pt"


class ReconstructionLogger(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        val_samples = next(iter(trainer.train_dataloader))  # type: ignore
        x, _ = val_samples
        x = x.to(pl_module.device)
        output = pl_module.forward(x)
        reconstructions = output["output"]
        # y, _ = next(iter(trainer.val_dataloaders))  # type: ignore
        # y = y.to(pl_module.device)
        # output = pl_module.forward(y)
        # val_reconst = output["output"]
        # grid = make_grid([x[0], reconstructions[0], y[0], val_reconst[0]])
        grid = make_grid([x[0], reconstructions[0]]) # TODO: break out this functionality, dump first sample to console, latent vars, output

        tensor_filename = IMG_PATH
        if os.path.exists(tensor_filename):
            existing_tensors = torch.load(tensor_filename)
            new_frame = grid.permute(1, 2, 0).to(torch.uint8)
            existing_tensors = torch.cat(
                (existing_tensors, new_frame.unsqueeze(0)), dim=0
            )
        else:
            os.makedirs("logs/VanillaVAE/reconstructions", exist_ok=True)
            existing_tensors = grid.permute(1, 2, 0).to(torch.uint8).unsqueeze(0)
        torch.save(existing_tensors, tensor_filename)

        # if trainer.current_epoch % 50 == 0:
        trainer.logger.experiment.add_image(
            f"reconstruction {trainer.current_epoch:03d}",
            grid,
            global_step=trainer.global_step,
        )


def main(config):
    # system setup
    L.seed_everything(config["exp_params"]["manual_seed"], workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(mode=True)  # type: ignore
    device = (
        torch.device(config["trainer_params"]["device"])
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device: '{device}'")

    # model setup
    model = VanillaVAE(
        config["model_params"]["in_channels"],
        config["model_params"]["latent_dim"],
        config["exp_params"],
        hidden_dims=config["model_params"]["hidden_dims"],
    )
    model.to(device)
    summary(
        model,
        input_size=(
            config["data_params"]["train_batch_size"],
            config["model_params"]["in_channels"],
            28,
            28,
            # 3,
            # 128,
            # 128,
        ),
        device=device.type,
    )
    # dataset setup
    img_transforms = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0], std=[1]), # TODO: properly standardize (mean=0.5)
        ]
    )
    full_dataset = torchvision.datasets.MNIST(
        root="/media/nova/Datasets/mnist",
        train=True,
        download=False,
        transform=img_transforms,
    )
    # full_dataset = ImageFolder(
    #     config["data_params"]["data_path"], transform=img_transforms
    # )
    print(f"loaded {len(full_dataset)} images")
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data_params"]["train_batch_size"],
        shuffle=True,
        num_workers=config["data_params"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data_params"]["val_batch_size"],
        shuffle=False,
        num_workers=config["data_params"]["num_workers"],
    )

    # lightning setup
    logger = TensorBoardLogger(
        save_dir=config["logging_params"]["save_dir"],
        name=config["logging_params"]["name"],
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="best_model",
        save_top_k=1,
        mode="min",
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
    )
    reconstruction_callback = ReconstructionLogger()
    trainer = L.Trainer(
        logger=logger,
        # overfit_batches=1,
        # limit_train_batches=100,
        log_every_n_steps=config["logging_params"]["log_interval"],
        accelerator="gpu",
        devices=config["trainer_params"]["devices"],
        max_epochs=config["trainer_params"]["max_epochs"],
        default_root_dir=config["logging_params"]["save_dir"],
        enable_checkpointing=True,
        # val_check_interval=0.1,
        check_val_every_n_epoch=config["trainer_params"]["check_val_interval"],
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            reconstruction_callback,
        ],
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("training complete, saving video")
    frames = torch.load(IMG_PATH)
    torchvision.io.write_video(
        f"logs/VanillaVAE/videos/reconstruction_{IT}.mp4", frames.cpu(), fps=25
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic runner for VAE models")
    parser.add_argument(
        "--config",
        "-c",
        dest="filename",
        metavar="FILE",
        help="path to the config file",
        default="configs/vae.yaml",
    )

    args = parser.parse_args()
    with open(args.filename, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    main(config)
