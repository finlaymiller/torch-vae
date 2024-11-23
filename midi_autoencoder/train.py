import copy
import os
import shutil
import time
from contextlib import nullcontext
from datetime import datetime

import data_transformations
import datasets
import torch
import utils
from evaluation import evaluate
from models import VanillaVAE
from rich import print
from torch import nn

BASE_BATCH_SIZE = 128


def run(config):
    """
    Run training job (one worker if using distributed training).

    Parameters
    ----------
    config : argparse.Namespace or OmegaConf
        The configuration for this experiment.
    """
    if config.log_wandb:
        # Lazy import of wandb, since logging to wandb is optional
        import wandb

    if config.seed is not None:
        utils.set_rng_seeds_fixed(config.seed)

    if config.deterministic:
        print("Running in deterministic cuDNN mode. Performance may be slower, but more reproducible.")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print()
    print("Configuration:")
    print()
    print(config)
    print()
    print(f"Found {torch.cuda.device_count()} GPUs and {utils.get_num_cpu_available()} CPUs.")

    # Check which device to use
    use_cuda = not config.no_cuda and torch.cuda.is_available()

    if not use_cuda:
        device = torch.device("cpu")
    elif config.local_rank is not None:
        device = f"cuda:{config.local_rank}"
    else:
        device = "cuda"

    print(f"Using device '{device}'")

    # RESTORE OMITTED CONFIG FROM RESUMPTION CHECKPOINT =======================
    checkpoint = None
    config.model_output_dir = None
    if config.checkpoint_path:
        config.model_output_dir = os.path.dirname(config.checkpoint_path)
    if not config.checkpoint_path:
        # Not trying to resume from a checkpoint
        pass
    elif not os.path.isfile(config.checkpoint_path):
        # Looks like we're trying to resume from the checkpoint that this job
        # will itself create. Let's assume this is to let the job resume upon
        # preemption, and it just hasn't been preempted yet.
        print(f"Skipping premature resumption from preemption: no checkpoint file found at '{config.checkpoint_path}'")
    else:
        print(f"Loading resumption checkpoint '{config.checkpoint_path}'")
        # Map model parameters to be load to the specified gpu.
        checkpoint = torch.load(config.checkpoint_path, map_location=device)
        keys = vars(get_parser().parse_args("")).keys()
        keys = set(keys).difference(["resume", "gpu", "global_rank", "local_rank", "cpu_workers"])
        for key in keys:
            if getattr(checkpoint["config"], key, None) is None:
                continue
            if getattr(config, key) is None:
                print(f"  Restoring config value for {key} from checkpoint: {getattr(checkpoint['config'], key)}")
                setattr(config, key, getattr(checkpoint["config"], key, None))
            elif getattr(config, key) != getattr(checkpoint["config"], key):
                print(
                    f"  Warning: config value for {key} differs from checkpoint:"
                    f" {getattr(config, key)} (ours) vs {getattr(checkpoint['config'], key)} (checkpoint)"
                )

    if checkpoint is None:
        # Our epochs go from 1 to n_epoch, inclusive
        start_epoch = 1
    else:
        # Continue from where we left off
        start_epoch = checkpoint["epoch"] + 1
        if config.seed is not None:
            # Make sure we don't get the same behaviour as we did on the
            # first epoch repeated on this resumed epoch.
            utils.set_rng_seeds_fixed(config.seed + start_epoch, all_gpu=False)

    # MODEL ===================================================================
    # We have to build the model before we load the dataset because it will
    # inform us about what size images we should produce in the preprocessing pipeline.
    n_class, raw_img_size, img_channels = datasets.image_dataset_sizes(config.dataset_name)
    if not hasattr(config, "image_size") or config.image_size is None:
        print(f"Setting model input image size to dataset's image size: {raw_img_size}")
        config.image_size = raw_img_size
    print(f"loading model for '{config.dataset_name}' dataset")
    model = VanillaVAE(img_channels, config.n_features)
    # model = VAE(input_dim=raw_img_size**2, hidden_dim=512, latent_dim=2)
    # holdover from original script, for compatibility
    encoder_config = {
        "input_size": raw_img_size,
        "n_feature": config.n_features,
    }

    # Configure model for distributed training --------------------------------
    print("\nModel architecture:")
    print(model)

    if config.cpu_workers is None:
        config.cpu_workers = utils.get_num_cpu_available()

    if not use_cuda:
        print("Using CPU (this will be slow)")
    else:
        if config.local_rank is not None:
            torch.cuda.set_device(config.local_rank)
        model = model.to(device)

    # DATASET =================================================================
    print(f"Loading '{config.dataset_name}' dataset")
    # Transforms --------------------------------------------------------------
    transform_args = {}
    if config.dataset_name in data_transformations.VALID_TRANSFORMS:
        transform_args["normalization"] = config.dataset_name
        print(f"normalization type set to {config.dataset_name}")

    transform_train, transform_eval = data_transformations.get_transform(
        config.transform_type, config.image_size, transform_args
    )

    # Dataset -----------------------------------------------------------------
    dataset_args = {
        "dataset": config.dataset_name,
        "root": config.data_dir,
        "prototyping": config.prototyping,
        "download": config.allow_download_dataset,
    }
    if config.protoval_split_id is not None:
        dataset_args["protoval_split_id"] = config.protoval_split_id
    (
        dataset_train,
        dataset_val,
        dataset_test,
        distinct_val_test,
    ) = datasets.fetch_dataset(
        **dataset_args,
        transform_train=transform_train,
        transform_eval=transform_eval,
    )
    eval_set = "Val" if distinct_val_test else "Test"

    # Dataloader --------------------------------------------------------------
    config.world_size = int(os.environ.get("WORLD_SIZE", 1))
    config.batch_size = config.batch_size_per_gpu * config.world_size

    dl_train_kwargs = {
        "batch_size": config.batch_size_per_gpu,
        "drop_last": True,
        "sampler": None,
        "shuffle": True,
        "worker_init_fn": utils.worker_seed_fn,
    }
    dl_test_kwargs = {
        "batch_size": config.batch_size_per_gpu,
        "drop_last": False,
        "sampler": None,
        "shuffle": False,
        "worker_init_fn": utils.worker_seed_fn,
    }
    if use_cuda:
        cuda_kwargs = {"num_workers": config.cpu_workers, "pin_memory": True}
        dl_train_kwargs.update(cuda_kwargs)
        dl_test_kwargs.update(cuda_kwargs)

    dl_val_kwargs = copy.deepcopy(dl_test_kwargs)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, **dl_train_kwargs)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, **dl_val_kwargs)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, **dl_test_kwargs)

    # OPTIMIZATION ============================================================
    print(f"Loading optimizer '{config.optimizer}'")
    # Optimizer ---------------------------------------------------------------
    # Set up the optimizer

    # Bigger batch sizes mean better estimates of the gradient, so we can use a
    # bigger learning rate. See https://arxiv.org/abs/1706.02677
    # Hence we scale the learning rate linearly with the total batch size.
    config.lr = config.lr_relative * config.batch_size / BASE_BATCH_SIZE

    # Freeze the encoder, if requested
    if config.freeze_encoder:
        for m in model.encoder.parameters():
            m.requires_grad = False

    # Set up a parameter group for each component of the model, allowing
    # them to have different learning rates (for fine-tuning encoder).
    params = []
    if not config.freeze_encoder:
        params.append(
            {
                "params": model.encoder.parameters(),
                "lr": config.lr * config.lr_encoder_mult,
                "name": "encoder",
            }
        )
    params.append(
        {
            "params": model.decoder.parameters(),
            "lr": config.lr * config.lr_decoder_mult,
            "name": "decoder",
        }
    )

    # Fetch the constructor of the appropriate optimizer from torch.optim
    optimizer = getattr(torch.optim, config.optimizer)(params, lr=config.lr, weight_decay=config.weight_decay)

    # Scheduler ---------------------------------------------------------------
    # Set up the learning rate scheduler
    if config.scheduler.lower() == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            [p["lr"] for p in optimizer.param_groups],
            epochs=config.epochs,
            steps_per_epoch=len(dataloader_train),
        )
    else:
        raise NotImplementedError(f"Scheduler {config.scheduler} not supported.")

    # Loss function -----------------------------------------------------------
    # Set up loss function
    criterion = nn.MSELoss()

    # LOGGING =================================================================
    # Setup logging and saving

    # If we're using wandb, initialize the run, or resume it if the job was preempted.
    if config.log_wandb and config.global_rank == 0:
        wandb_run_name = config.run_name
        if wandb_run_name is not None and config.run_id is not None:
            wandb_run_name = f"{wandb_run_name}__{config.run_id}"
        EXCLUDED_WANDB_CONFIG_KEYS = [
            "log_wandb",
            "wandb_entity",
            "wandb_project",
            "global_rank",
            "local_rank",
            "run_name",
            "run_id",
            "model_output_dir",
        ]
        utils.init_or_resume_wandb_run(
            config.model_output_dir,
            name=wandb_run_name,
            id=config.run_id,
            entity=config.wandb_entity,
            project=config.wandb_project,
            config=wandb.helper.parse_config(config, exclude=EXCLUDED_WANDB_CONFIG_KEYS),
            job_type="train",
            tags=["prototype" if config.prototyping else "final"],
        )
        # If a run_id was not supplied at the command prompt, wandb will
        # generate a name. Let's use that as the run_name.
        if config.run_name is None:
            config.run_name = wandb.run.name
            print(f"Copied run name '{config.run_name}' from wandb")
        if config.run_id is None:
            config.run_id = wandb.run.id
            print(f"Copied run id '{config.run_id}' from wandb")

    # If we still don't have a run name, generate one from the current time.
    if config.run_name is None:
        config.run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"Generated run name '{config.run_name}'")
    if config.run_id is None:
        config.run_id = utils.generate_id()
        print(f"Generated run id '{config.run_id}'")

    # If no checkpoint path was supplied, but models_dir was, we will automatically
    # determine the path to which we will save the model checkpoint.
    # If both are empty, we won't save the model.
    if not config.checkpoint_path and config.models_dir:
        config.model_output_dir = os.path.join(
            config.models_dir,
            config.dataset_name,
            f"{config.run_name}__{config.run_id}",
        )
        config.checkpoint_path = os.path.join(config.model_output_dir, "checkpoint_latest.pt")
        if config.log_wandb and config.global_rank == 0:
            wandb.config.update({"checkpoint_path": config.checkpoint_path}, allow_val_change=True)

    if config.checkpoint_path is None:
        print("Model will not be saved.")
    else:
        print(f"Model will be saved to '{config.checkpoint_path}'")

    # RESUME ==================================================================
    # Now that everything is set up, we can load the state of the model,
    # optimizer, and scheduler from a checkpoint, if supplied.

    # Initialize step related variables as if we're starting from scratch.
    # Their values will be overridden by the checkpoint if we're resuming.
    total_step = 0
    n_samples_seen = 0

    best_stats = {"best_epoch": 0}

    if checkpoint is not None:
        print(f"Loading state from checkpoint (epoch {checkpoint['epoch']})")
        # Map model to be loaded to specified single gpu.
        total_step = checkpoint["total_step"]
        n_samples_seen = checkpoint["n_samples_seen"]
        model.encoder.load_state_dict(checkpoint["encoder"])
        model.decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_stats["best_epoch"] = checkpoint.get("best_epoch", 0)

    # TRAIN ===================================================================
    print()
    print("Configuration:")
    print()
    print(config)
    print()

    # Ensure model is on the correct device
    model = model.to(device)

    timing_stats = {}
    t_end_epoch = time.time()
    for epoch in range(start_epoch, config.epochs + 1):
        t_start_epoch = time.time()
        if config.seed is not None:
            # If the job is resumed from preemption, our RNG state is currently set the
            # same as it was at the start of the first epoch, not where it was when we
            # stopped training. This is not good as it means jobs which are resumed
            # don't do the same thing as they would be if they'd run uninterrupted
            # (making preempted jobs non-reproducible).
            # To address this, we reset the seed at the start of every epoch. Since jobs
            # can only save at the end of and resume at the start of an epoch, this
            # makes the training process reproducible. But we shouldn't use the same
            # RNG state for each epoch - instead we use the original seed to define the
            # series of seeds that we will use at the start of each epoch.
            epoch_seed = utils.determine_epoch_seed(config.seed, epoch=epoch)
            # We want each GPU to have a different seed to the others to avoid
            # correlated randomness between the workers on the same batch.
            # We offset the seed for this epoch by the GPU rank, so every GPU will get a
            # unique seed for the epoch. This means the job is only precisely
            # reproducible if it is rerun with the same number of GPUs (and the same
            # number of CPU workers for the dataloader).
            utils.set_rng_seeds_fixed(epoch_seed + config.global_rank, all_gpu=False)
            if isinstance(getattr(dataloader_train, "generator", None), torch.Generator):
                # Finesse the dataloader's RNG state, if it is not using the global state.
                dataloader_train.generator.manual_seed(epoch_seed + config.global_rank)
            if isinstance(getattr(dataloader_train.sampler, "generator", None), torch.Generator):
                # Finesse the sampler's RNG state, if it is not using the global RNG state.
                dataloader_train.sampler.generator.manual_seed(config.seed + epoch + 10000 * config.global_rank)

        if hasattr(dataloader_train.sampler, "set_epoch"):
            # Handling for DistributedSampler.
            # Set the epoch for the sampler so that it can shuffle the data
            # differently for each epoch, but synchronized across all GPUs.
            dataloader_train.sampler.set_epoch(epoch)

        # Train ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Note the number of samples seen before this epoch started, so we can
        # calculate the number of samples seen in this epoch.
        n_samples_seen_before = n_samples_seen
        # Run one epoch of training
        train_stats, total_step, n_samples_seen = train_one_epoch(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            dataloader=dataloader_train,
            device=device,
            epoch=epoch,
            n_epoch=config.epochs,
            total_step=total_step,
            n_samples_seen=n_samples_seen,
        )
        t_end_train = time.time()

        timing_stats["train"] = t_end_train - t_start_epoch
        n_epoch_samples = n_samples_seen - n_samples_seen_before
        train_stats["throughput"] = n_epoch_samples / timing_stats["train"]

        print(f"\nTraining epoch {epoch}/{config.epochs} summary:")
        print(f"  Steps ..............{len(dataloader_train):8d}")
        print(f"  Samples ............{n_epoch_samples:8d}")
        if timing_stats["train"] > 172800:
            print(f"  Duration ...........{timing_stats['train']/86400:11.2f} days")
        elif timing_stats["train"] > 5400:
            print(f"  Duration ...........{timing_stats['train']/3600:11.2f} hours")
        elif timing_stats["train"] > 120:
            print(f"  Duration ...........{timing_stats['train']/60:11.2f} minutes")
        else:
            print(f"  Duration ...........{timing_stats['train']:11.2f} seconds")
        print(f"  Throughput .........{train_stats['throughput']:11.2f} samples/sec")
        print(f"  Loss ...............{train_stats['loss']:14.5f}")

        # Validate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Evaluate on validation set
        t_start_val = time.time()

        eval_stats = evaluate(
            dataloader=dataloader_val,
            model=model,
            device=device,
            partition_name=eval_set,
        )
        t_end_val = time.time()
        timing_stats["val"] = t_end_val - t_start_val
        eval_stats["throughput"] = len(dataloader_val.dataset) / timing_stats["val"]

        print(f"Evaluating epoch {epoch}/{config.epochs} summary:")
        if timing_stats["val"] > 172800:
            print(f"  Duration ...........{timing_stats['val']/86400:11.2f} days")
        elif timing_stats["val"] > 5400:
            print(f"  Duration ...........{timing_stats['val']/3600:11.2f} hours")
        elif timing_stats["val"] > 120:
            print(f"  Duration ...........{timing_stats['val']/60:11.2f} minutes")
        else:
            print(f"  Duration ...........{timing_stats['val']:11.2f} seconds")
        print(f"  Throughput .........{eval_stats['throughput']:11.2f} samples/sec")
        print(f"  Cross-entropy ......{eval_stats['cross-entropy']:14.5f}")

        # Save model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        t_start_save = time.time()
        if config.model_output_dir and (not config.global_rank == 0):
            utils.safe_save_model(
                {
                    "encoder": model.encoder,
                    "decoder": model.decoder,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                },
                config.checkpoint_path,
                config=config,
                epoch=epoch,
                total_step=total_step,
                n_samples_seen=n_samples_seen,
                encoder_config=encoder_config,
                transform_args=transform_args,
                **best_stats,
            )
            if config.save_best_model and best_stats["best_epoch"] == epoch:
                ckpt_path_best = os.path.join(config.model_output_dir, "best_model.pt")
                print(f"Copying model to {ckpt_path_best}")
                shutil.copyfile(config.checkpoint_path, ckpt_path_best)

        t_end_save = time.time()
        timing_stats["saving"] = t_end_save - t_start_save

        # Log to wandb ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Overall time won't include uploading to wandb, but there's nothing
        # we can do about that.
        timing_stats["overall"] = time.time() - t_end_epoch
        t_end_epoch = time.time()

        # Send training and eval stats for this epoch to wandb
        if config.log_wandb and config.global_rank == 0:
            pre = "training/epochwise"
            wandb.log(
                {
                    "training/stepwise/epoch": epoch,
                    "training/stepwise/epoch_progress": epoch,
                    "training/stepwise/n_samples_seen": n_samples_seen,
                    f"{pre}/epoch": epoch,
                    **{f"{pre}/train/{k}": v for k, v in train_stats.items()},
                    **{f"{pre}/{eval_set}/{k}": v for k, v in eval_stats.items()},
                    **{f"{pre}/duration/{k}": v for k, v in timing_stats.items()},
                },
                step=total_step,
            )
            # Record the wandb time as contributing to the next epoch
            timing_stats = {"wandb": time.time() - t_end_epoch}
        else:
            # Reset timing stats
            timing_stats = {}
        # Print with flush=True forces the output buffer to be printed immediately
        print("", flush=True)

    if start_epoch > config.epochs:
        print("Training already completed!")
    else:
        print(f"Training complete! (Trained epochs {start_epoch} to {config.epochs})")

    # TEST ====================================================================
    print(f"\nEvaluating final model (epoch {config.epochs}) performance")
    # Evaluate on test set
    print("\nEvaluating final model on test set...")
    eval_stats = evaluate(
        dataloader=dataloader_test,
        model=model,
        device=device,
        partition_name="Test",
    )
    # Send stats to wandb
    if config.log_wandb and config.global_rank == 0:
        wandb.log({**{f"eval/test/{k}": v for k, v in eval_stats.items()}}, step=total_step)

    if distinct_val_test:
        # Evaluate on validation set
        print(f"\nEvaluating final model on {eval_set} set...")
        eval_stats = evaluate(
            dataloader=dataloader_val,
            model=model,
            device=device,
            partition_name=eval_set,
        )
        # Send stats to wandb
        if config.log_wandb and config.global_rank == 0:
            wandb.log(
                {**{f"eval/{eval_set}/{k}": v for k, v in eval_stats.items()}},
                step=total_step,
            )

    # Create a copy of the train partition with evaluation transforms
    # and a dataloader using the evaluation configuration (don't drop last)
    print("\nEvaluating final model on train set under test conditions (no augmentation, dropout, etc)...")
    dataset_train_eval = datasets.fetch_dataset(
        **dataset_args,
        transform_train=transform_eval,
        transform_eval=transform_eval,
    )[0]
    dl_train_eval_kwargs = copy.deepcopy(dl_test_kwargs)
    dataloader_train_eval = torch.utils.data.DataLoader(dataset_train_eval, **dl_train_eval_kwargs)
    eval_stats = evaluate(
        dataloader=dataloader_train_eval,
        model=model,
        device=device,
        partition_name="Train",
    )
    # Send stats to wandb
    if config.log_wandb and config.global_rank == 0:
        wandb.log({**{f"eval/train/{k}": v for k, v in eval_stats.items()}}, step=total_step)


def train_one_epoch(
    config,
    model,
    optimizer,
    scheduler,
    criterion,
    dataloader,
    device="cuda",
    epoch=1,
    n_epoch=None,
    total_step=0,
    n_samples_seen=0,
):
    r"""
    Train the model for one epoch.

    Parameters
    ----------
    config : argparse.Namespace or OmegaConf
        The global config object.
    model : torch.nn.Module
        The model network.
    optimizer : torch.optim.Optimizer
        The optimizer.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler.
    criterion : torch.nn.Module
        The loss function.
    dataloader : torch.utils.data.DataLoader
        A dataloader for the training set.
    device : str or torch.device, default="cuda"
        The device to use.
    epoch : int, default=1
        The current epoch number (indexed from 1).
    n_epoch : int, optional
        The total number of epochs scheduled to train for.
    total_step : int, default=0
        The total number of steps taken so far.
    n_samples_seen : int, default=0
        The total number of samples seen so far.

    Returns
    -------
    results: dict
        A dictionary containing the training performance for this epoch.
    total_step : int
        The total number of steps taken after this epoch.
    n_samples_seen : int
        The total number of samples seen after this epoch.
    """
    # Put the model in train mode
    model.train()

    if config.log_wandb:
        # Lazy import of wandb, since logging to wandb is optional
        import wandb

    loss_epoch = 0

    if config.print_interval is None:
        # Default to printing to console every time we log to wandb
        config.print_interval = config.log_interval

    t_end_batch = time.time()
    t_start_wandb = t_end_wandb = None
    for batch_idx, (stimuli, y_true) in enumerate(dataloader):
        t_start_batch = time.time()
        batch_size_this_gpu = stimuli.shape[0]

        # Move training inputs and targets to the GPU
        stimuli = stimuli.to(device)
        y_true = y_true.to(device)

        # Forward pass --------------------------------------------------------
        # Perform the forward pass through the model
        t_start_model = time.time()
        # N.B. To accurately time steps on GPU we need to use torch.cuda.Event
        ct_forward = torch.cuda.Event(enable_timing=True)
        ct_forward.record()
        with torch.no_grad() if config.freeze_encoder else nullcontext():
            output = model.forward(stimuli)
            reconstruction = output["output"]
        # Reset gradients
        optimizer.zero_grad()
        # Measure loss
        loss = criterion(reconstruction, stimuli)

        # Backward pass -------------------------------------------------------
        # Now the backward pass
        ct_backward = torch.cuda.Event(enable_timing=True)
        ct_backward.record()
        loss.backward()

        # Update --------------------------------------------------------------
        # Use our optimizer to update the model parameters
        ct_optimizer = torch.cuda.Event(enable_timing=True)
        ct_optimizer.record()
        optimizer.step()

        # Step the scheduler each batch
        scheduler.step()

        # Increment training progress counters
        total_step += 1
        batch_size_all = batch_size_this_gpu * config.world_size
        n_samples_seen += batch_size_all

        # Logging -------------------------------------------------------------
        # Log details about training progress
        t_start_logging = time.time()
        ct_logging = torch.cuda.Event(enable_timing=True)
        ct_logging.record()

        loss_batch = loss.item()
        loss_epoch += loss_batch

        if epoch <= 1 and batch_idx == 0:
            # Debugging
            print("stimuli.shape =", stimuli.shape)
            print("logits.shape  =", reconstruction.shape)
            print("loss.shape    =", loss.shape)
            # Debugging intensifies
            print("y_true =", y_true)
            print("logits[0] =", reconstruction[0])
            print("loss =", loss.detach().item())

        # Log sample training images to show on wandb
        if config.log_wandb and batch_idx <= 1:
            # Log 8 example training images from each GPU
            img_indices = [offset + relative for offset in [0, batch_size_this_gpu // 2] for relative in [0, 1, 2, 3]]
            img_indices = sorted(set(img_indices))
            stimuli_images = stimuli[img_indices]
            reconst_images = reconstruction[img_indices]
            paired_images = [torch.cat((img1, img2), dim=2) for img1, img2 in zip(reconst_images, stimuli_images)]
            rows = [torch.cat(paired_images[i : i + 4], dim=2) for i in range(0, len(paired_images), 4)]
            final_grid = torch.cat(rows, dim=1)
            if config.global_rank == 0:
                wandb.log(
                    {"training/stepwise/train/reconstruction": wandb.Image(final_grid)},
                    step=total_step,
                )

        # Log to console
        if batch_idx <= 2 or batch_idx % config.print_interval == 0 or batch_idx >= len(dataloader) - 1:
            print(
                f"Train Epoch:{epoch:4d}" + (f"/{n_epoch}" if n_epoch is not None else ""),
                " Step:{:4d}/{}".format(batch_idx + 1, len(dataloader)),
                " Loss:{:8.5f}".format(loss_batch),
                " LR: {}".format(scheduler.get_last_lr()),
            )

        # Log to wandb
        if config.log_wandb and config.global_rank == 0 and batch_idx % config.log_interval == 0:
            # Create a log dictionary to send to wandb
            # Epoch progress interpolates smoothly between epochs
            epoch_progress = epoch - 1 + (batch_idx + 1) / len(dataloader)
            # Throughput is the number of samples processed per second
            throughput = batch_size_all / (t_start_logging - t_end_batch)
            log_dict = {
                "training/stepwise/epoch": epoch,
                "training/stepwise/epoch_progress": epoch_progress,
                "training/stepwise/n_samples_seen": n_samples_seen,
                "training/stepwise/train/throughput": throughput,
                "training/stepwise/train/loss": loss_batch,
            }
            # Track the learning rate of each parameter group
            for lr_idx in range(len(optimizer.param_groups)):
                if "name" in optimizer.param_groups[lr_idx]:
                    grp_name = optimizer.param_groups[lr_idx]["name"]
                elif len(optimizer.param_groups) == 1:
                    grp_name = ""
                else:
                    grp_name = f"grp{lr_idx}"
                if grp_name != "":
                    grp_name = f"-{grp_name}"
                grp_lr = optimizer.param_groups[lr_idx]["lr"]
                log_dict[f"training/stepwise/lr{grp_name}"] = grp_lr
            # Synchronize ensures everything has finished running on each GPU
            torch.cuda.synchronize()
            # Record how long it took to do each step in the pipeline
            pre = "training/stepwise/duration"
            if t_start_wandb is not None:
                # Record how long it took to send to wandb last time
                log_dict[f"{pre}/wandb"] = t_end_wandb - t_start_wandb
            log_dict[f"{pre}/dataloader"] = t_start_batch - t_end_batch
            log_dict[f"{pre}/preamble"] = t_start_model - t_start_batch
            log_dict[f"{pre}/forward"] = ct_forward.elapsed_time(ct_backward) / 1000
            log_dict[f"{pre}/backward"] = ct_backward.elapsed_time(ct_optimizer) / 1000
            log_dict[f"{pre}/optimizer"] = ct_optimizer.elapsed_time(ct_logging) / 1000
            log_dict[f"{pre}/overall"] = time.time() - t_end_batch
            t_start_wandb = time.time()
            log_dict[f"{pre}/logging"] = t_start_wandb - t_start_logging
            # Send to wandb
            wandb.log(log_dict, step=total_step)
            t_end_wandb = time.time()

        # Record the time when we finished this batch
        t_end_batch = time.time()

    results = {
        "loss": loss_epoch / len(dataloader),
    }
    return results, total_step, n_samples_seen


def get_parser():
    r"""
    Build argument parser for the command line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
        CLI argument parser.
    """
    import argparse
    import sys

    # Use the name of the file called to determine the name of the program
    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py" or prog == "__main__":
        # If the file is called __main__.py, go up a level to the module name
        prog = os.path.split(__file__)[1]
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Train MIDI classification model.",
        add_help=False,
    )
    # Help arg ----------------------------------------------------------------
    group = parser.add_argument_group("Help")
    group.add_argument(
        "--help",
        "-h",
        action="help",
        help="Show this help message and exit.",
    )
    # Dataset args ------------------------------------------------------------
    group = parser.add_argument_group("Dataset")
    group.add_argument(
        "--dataset",
        dest="dataset_name",
        type=str,
        default="mnist",
        help="Name of the dataset to learn. Default: %(default)s",
    )
    group.add_argument(
        "--prototyping",
        dest="protoval_split_id",
        nargs="?",
        const=0,
        type=int,
        help=(
            "Use a subset of the train partition for both train and val."
            " If the dataset doesn't have a separate val and test set with"
            " public labels (which is the case for most datasets), the train"
            " partition will be reduced in size to create the val partition."
            " In all cases where --prototyping is enabled, the test set is"
            " never used during training. Generally, you should use"
            " --prototyping throughout the model exploration and hyperparameter"
            " optimization phases, and disable it for your final experiments so"
            " they can run on a completely held-out test set."
        ),
    )
    group.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Directory within which the dataset can be found."
            " Default is ~/Datasets, except on Vector servers where it is"
            " adjusted as appropriate depending on the dataset's location."
        ),
    )
    group.add_argument(
        "--allow-download-dataset",
        action="store_true",
        help="Attempt to download the dataset if it is not found locally.",
    )
    group.add_argument(
        "--transform-type",
        type=str,
        default="digits",
        help="Name of augmentation stack to apply to training data. Default: %(default)s",
    )
    group.add_argument(
        "--image-size",
        type=int,
        help="Size of images to use as model input. Default: encoder's default.",
    )
    # Architecture args -------------------------------------------------------
    group = parser.add_argument_group("Architecture")
    group.add_argument(
        "--model",
        "--encoder",
        "--arch",
        "--architecture",
        dest="arch",
        type=str,
        default="VanillaVAE",
        help="Name of model architecture. Default: %(default)s",
    )
    group.add_argument(
        "--pretrained",
        action="store_true",
        help="Use default pretrained model weights, taken from hugging-face hub.",
    )
    group.add_argument(
        "--freeze-encoder",
        action="store_true",
    )
    group.add_argument(
        "--n_features",
        type=int,
        default=100,
        help="Number of hidden features. Default: %(default)s",
    )
    # Optimization args -------------------------------------------------------
    group = parser.add_argument_group("Optimization routine")
    group.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to train for. Default: %(default)s",
    )
    group.add_argument(
        "--lr",
        dest="lr_relative",
        type=float,
        default=0.01,
        help=(
            f"Maximum learning rate, set per {BASE_BATCH_SIZE} batch size."
            " The actual learning rate used will be scaled up by the total"
            " batch size (across all GPUs). Default: %(default)s"
        ),
    )
    group.add_argument(
        "--lr-encoder-mult",
        dest="lr_encoder_mult",
        type=float,
        default=1.0,
        help="Multiplier for encoder learning rate, relative to overall LR.",
    )
    group.add_argument(
        "--lr-decoder-mult",
        dest="lr_decoder_mult",
        type=float,
        default=1.0,
        help="Multiplier for decoder learning rate, relative to overall LR.",
    )
    group.add_argument(
        "--weight-decay",
        "--wd",
        dest="weight_decay",
        type=float,
        default=0.0,
        help="Weight decay. Default: %(default)s",
    )
    group.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help="Name of optimizer (case-sensitive). Default: %(default)s",
    )
    group.add_argument(
        "--scheduler",
        type=str,
        default="OneCycle",
        help="Learning rate scheduler. Default: %(default)s",
    )
    # Output checkpoint args --------------------------------------------------
    group = parser.add_argument_group("Output checkpoint")
    group.add_argument(
        "--models-dir",
        type=str,
        default="models",
        metavar="PATH",
        help="Output directory for all models. Ignored if --checkpoint is set. Default: %(default)s",
    )
    group.add_argument(
        "--checkpoint",
        dest="checkpoint_path",
        default="",
        type=str,
        metavar="PATH",
        help=(
            "Save and resume partially trained model and optimizer state from this checkpoint."
            " Overrides --models-dir."
        ),
    )
    group.add_argument(
        "--save-best-model",
        action="store_true",
        help="Save a copy of the model with best validation performance.",
    )
    # Reproducibility args ----------------------------------------------------
    group = parser.add_argument_group("Reproducibility")
    group.add_argument(
        "--seed",
        type=int,
        help="Random number generator (RNG) seed. Default: not controlled",
    )
    group.add_argument(
        "--deterministic",
        action="store_true",
        help="Disable non-deterministic features of cuDNN.",
    )
    # Hardware configuration args ---------------------------------------------
    group = parser.add_argument_group("Hardware configuration")
    group.add_argument(
        "--batch-size",
        dest="batch_size_per_gpu",
        type=int,
        default=BASE_BATCH_SIZE,
        help=(
            "Batch size per GPU. The total batch size will be this value times"
            " the total number of GPUs used. Default: %(default)s"
        ),
    )
    group.add_argument(
        "--global-rank",
        dest="global_rank",
        type=int,
        default=0,
        help="Global rank for distributed computing. Default: %(default)s",
    )
    group.add_argument(
        "--cpu-workers",
        "--workers",
        dest="cpu_workers",
        type=int,
        help="Number of CPU workers per node. Default: number of CPUs available on device.",
    )
    group.add_argument(
        "--no-cuda",
        action="store_true",
        help="Use CPU only, no GPUs.",
    )
    group.add_argument(
        "--gpu",
        dest="local_rank",
        default=None,
        type=int,
        help="Index of GPU to use when training a single process. (Ignored for distributed training.)",
    )
    # Logging args ------------------------------------------------------------
    group = parser.add_argument_group("Debugging and logging")
    group.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Number of batches between each log to wandb (if enabled). Default: %(default)s",
    )
    group.add_argument(
        "--print-interval",
        type=int,
        default=None,
        help="Number of batches between each print to STDOUT. Default: same as LOG_INTERVAL.",
    )
    group.add_argument(
        "--log-wandb",
        action="store_true",
        help="Log results with Weights & Biases https://wandb.ai",
    )
    group.add_argument(
        "--disable-wandb",
        "--no-wandb",
        dest="disable_wandb",
        action="store_true",
        help="Overrides --log-wandb and ensures wandb is always disabled.",
    )
    group.add_argument(
        "--wandb-entity",
        type=str,
        help=(
            "The entity (organization) within which your wandb project is"
            ' located. By default, this will be your "default location" set on'
            " wandb at https://wandb.ai/settings"
        ),
    )
    group.add_argument(
        "--wandb-project",
        type=str,
        default="midi_autoencoder",
        help="Name of project on wandb, where these runs will be saved. Default: %(default)s",
    )
    group.add_argument(
        "--run-name",
        dest="run_name",
        type=str,
        default=None,
        help="Human-readable identifier for the model run or job. Used to name the run on wandb. Default: %(default)s",
    )
    group.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Unique identifier for the model run or job. Used as the run ID on wandb.",
    )

    return parser


def cli():
    r"""Command-line interface for model training."""
    parser = get_parser()
    config = parser.parse_args()
    # Handle disable_wandb overriding log_wandb and forcing it to be disabled.
    if config.disable_wandb:
        config.log_wandb = False
    del config.disable_wandb
    # Set protoval_split_id from prototyping, and turn prototyping into a bool
    config.prototyping = config.protoval_split_id is not None
    return run(config)


if __name__ == "__main__":
    cli()
