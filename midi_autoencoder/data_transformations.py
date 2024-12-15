import torch
from torchvision.transforms import v2

NORMALIZATION = {
    # "mnist": [(0.1307,), (0.3081,)],  # TODO: investigate this
    "mnist": [(0.5,), (1.0,)],
    "vae-lines": [(0.5,), (1.0,)],
    "vae-lines-large": [(0.5,), (1.0,)],
}

VALID_TRANSFORMS = ["mnist", "vae-lines", "vae-lines-large"]


def get_transform(transform_type: str = "noaug", image_size: int = 32, args=None) -> tuple[v2.Transform, v2.Transform]:
    if args is None:
        args = {}
    mean, std = NORMALIZATION[args.get("normalization", "mnist")]
    if "mean" in args:
        mean = args["mean"]
    if "std" in args:
        std = args["std"]

    if transform_type == "noaug":
        # No augmentations, just resize and normalize.
        # N.B. If the raw training image isn't square, there is a small
        # "augmentation" as we will randomly crop a square (of length equal to
        # the shortest side) from it. We do this because we assume inputs to
        # the network must be square.
        train_transform = v2.Compose(
            [
                v2.Resize(image_size),  # Resize shortest side to image_size
                v2.RandomCrop(image_size),  # If it is not square, *random* crop
                v2.ToImage(),  # recommended replacement for toTensor() 1/2
                v2.ToDtype(torch.float32, scale=True),  # ------------- 2/2
                v2.Normalize(mean=mean, std=std),
            ]
        )
        test_transform = v2.Compose(
            [
                v2.Resize(image_size),  # Resize shortest side to image_size
                v2.CenterCrop(image_size),  # If it is not square, center crop
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )

    elif transform_type == "midi":
        # N.B. If the raw training image isn't square, there is a small
        # "augmentation" as we will randomly crop a square (of length equal to
        # the shortest side) from it. We do this because we assume inputs to
        # the network must be square.
        train_transform = v2.Compose(
            [
                v2.Resize(image_size),  # Resize shortest side to image_size
                v2.RandomCrop(image_size),  # If it is not square, *random* crop
                v2.ToImage(),  # recommended replacement for toTensor() 1/2
                v2.ToDtype(torch.float32, scale=True),  # ------------- 2/2
                v2.Normalize(mean=mean, std=std),
                v2.Grayscale(),
            ]
        )
        test_transform = v2.Compose(
            [
                v2.Resize(image_size),  # Resize shortest side to image_size
                v2.CenterCrop(image_size),  # If it is not square, center crop
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
                v2.Grayscale(),
            ]
        )

    elif transform_type == "digits":
        # Appropriate for smaller images containing digits, as in MNIST.
        # - Zoom in randomly with scale (small range of how much to zoom in by)
        # - Stretch with random aspect ratio
        # - Don't flip the images (that would change the digit)
        # - Randomly adjust brightness/contrast/saturation
        # - (No rotation or skew)
        # TODO: reimplement this:
        # train_transform = timm.data.create_transform(
        #     input_size=image_size,
        #     is_training=True,
        #     scale=(0.7, 1.0),  # reduced scale range
        #     ratio=(3.0 / 4.0, 4.0 / 3.0),  # default imagenet ratio range
        #     hflip=0.0,
        #     vflip=0.0,
        #     color_jitter=0.4,  # default imagenet color jitter
        #     interpolation="random",
        #     mean=mean,
        #     std=std,
        # )
        train_transform = v2.Compose(
            [
                v2.Resize(image_size),  # Resize shortest side to image_size
                v2.CenterCrop(image_size),  # If it is not square, center crop
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )
        # For testing:
        # - Resize to desired size only, with a center crop step included in
        #   case the raw image was not square.
        test_transform = v2.Compose(
            [
                v2.Resize(image_size),
                v2.CenterCrop(image_size),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

    else:
        raise NotImplementedError

    return (train_transform, test_transform)
