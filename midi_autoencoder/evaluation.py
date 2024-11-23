"""
Evaluation routines.
"""

import numpy as np
import torch
import torch.nn.functional as F
from rich import print
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate(
    dataloader,
    model,
    device,
    partition_name="Val",
    verbosity=1,
):
    r"""
    Evaluate model performance on a dataset.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Dataloader for the dataset to evaluate on.
    model : torch.nn.Module
        Model to evaluate.
    device : torch.device
        Device to run the model on.
    partition_name : str, default="Val"
        Name of the partition being evaluated.
    verbosity : int, default=1
        Verbosity level.
    is_distributed : bool, default=False
        Whether the model is distributed across multiple GPUs.

    Returns
    -------
    results : dict
        Dictionary of evaluation results.
    """
    model.eval()

    stim_all = []
    xent_all = []
    reconst_all = []
    latent_all = []

    # confirm data range of dataset
    stim_min = torch.inf
    stim_max = -torch.inf
    reconst_min = torch.inf
    reconst_max = -torch.inf

    for stimuli, _ in dataloader:
        if stimuli.min() < stim_min:
            stim_min = stimuli.min()
        if stimuli.max() > stim_max:
            stim_max = stimuli.max()

        stimuli = stimuli.to(device)
        with torch.no_grad():
            output = model(stimuli)
            reconstruction = output["output"]
            latent = output["latents"]
            xent = F.cross_entropy(reconstruction, stimuli, reduction="none")

        if reconstruction.min() < reconst_min:
            reconst_min = reconstruction.min()
        if reconstruction.max() > reconst_max:
            reconst_max = reconstruction.max()

        xent_all.append(xent.cpu().numpy())
        stim_all.append(stimuli.cpu().numpy())
        reconst_all.append(reconstruction.cpu().numpy())
        latent_all.append(latent.cpu().numpy())

    print(f"input has range  [{stim_min:.03f}, {stim_max:.03f}]")
    print(f"output has range [{reconst_min:.03f}, {reconst_max:.03f}]")

    # Concatenate the targets and predictions from each batch
    xent = np.concatenate(xent_all)
    stimuli = np.concatenate(stim_all)
    reconst = np.concatenate(reconst_all)
    latents = np.concatenate(latent_all)
    # If the dataset size was not evenly divisible by the world size,
    # DistributedSampler will pad the end of the list of samples
    # with some repetitions. We need to trim these off.
    n_samples = len(dataloader.dataset)
    xent = xent[:n_samples]
    stimuli = stimuli[:n_samples]
    reconst = reconst[:n_samples]
    latents = latents[:n_samples]
    # Create results dictionary
    results = {}
    results["count"] = len(stimuli)
    results["cross-entropy"] = np.mean(xent)
    # Note that these evaluation metrics have all been converted to percentages
    results["mse"] = 100.0 * mean_squared_error(stimuli.flatten(), reconst.flatten())
    results["mae"] = 100.0 * mean_absolute_error(stimuli.flatten(), reconst.flatten())
    # Could expand to other metrics too

    if verbosity >= 1:
        print(f"\n{partition_name} evaluation results:")
        for k, v in results.items():
            if "count" in k:
                print(f"  {k + ' ':.<21s}{v:7d}")
            elif "entropy" in k:
                print(f"  {k + ' ':.<24s} {v:9.5f} nat")
            else:
                print(f"  {k + ' ':.<24s} {v:6.2f} %")

    return results
