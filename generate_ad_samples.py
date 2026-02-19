import hashlib
import json
import os
import random
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import einops
import hydra
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from kornia import augmentation as aug
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import nn
from tqdm import tqdm

import wandb
from attack import AttackFramework, FGSMAttack, MIFGSMAttack, PGDAttack
from config_schema import MainConfig
from surrogates.FeatureExtractors import (
    Blip1FeatureExtractor,
    BlipFeatureExtractor,
    ClipFeatureExtractor,
    Dinov2FeatureExtractor,
    VisionTransformerFeatureExtractor,
)
from surrogates.FeatureExtractors.blip1 import MODEL_DICT as BLIP1_MODEL_DICT
from surrogates.FeatureExtractors.blip2 import MODEL_DICT as BLIP_MODEL_DICT
from surrogates.FeatureExtractors.clip import MODEL_DICT as CLIP_MODEL_DICT
from surrogates.FeatureExtractors.dino import MODEL_DICT as DINOV2_MODEL_DICT
from surrogates.FeatureExtractors.vit import MODEL_DICT as VIT_MODEL_DICT
from surrogates.loss import EnsWeightedMultiAlignmentLoss
from utils import ensure_dir, hash_training_config, setup_wandb

model_mappings = [
    (BLIP_MODEL_DICT, BlipFeatureExtractor),
    (DINOV2_MODEL_DICT, Dinov2FeatureExtractor),
    (CLIP_MODEL_DICT, ClipFeatureExtractor),
    (BLIP1_MODEL_DICT, Blip1FeatureExtractor),
    (VIT_MODEL_DICT, VisionTransformerFeatureExtractor),
]

MODEL_TO_CLASS = {}
for model_dict, extractor in model_mappings:
    MODEL_TO_CLASS.update({model_name: extractor for model_name in model_dict})


def get_models(cfg: MainConfig) -> List[nn.Module]:
    """
    Initializes and returns a list of models based on the configuration.

    Args:
        cfg: The configuration object.

    Returns:
        A list of initialized model instances.

    Raises:
        ValueError: If ensemble=False but multiple backbones specified,
                    or if an unknown backbone is specified.
    """
    if not cfg.model.ensemble and len(cfg.model.backbone) > 1:
        raise ValueError("When ensemble=False, only one backbone can be specified")

    models = []
    for backbone_name in cfg.model.backbone:
        if backbone_name in MODEL_TO_CLASS:
            model_cls = MODEL_TO_CLASS[backbone_name]
            model = (
                model_cls(backbone_name)
                .eval()
                .to(cfg.model.device)
                .requires_grad_(False)
            )
            models.append(model)
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
    return models


def get_ensemble_loss(cfg: MainConfig, models: List[nn.Module]) -> nn.Module:
    """
    Creates and returns the appropriate ensemble loss function based on the config.
    """
    beta = getattr(cfg.optim, "beta", 1.0)  # Default beta to 1.0 if not specified
    print(f"Using EnsWeightedMultiAlignmentLoss with beta={beta}")
    return EnsWeightedMultiAlignmentLoss(models, beta=beta)


def set_environment(seed: int = 2023):
    """
    Sets random seeds for reproducibility.

    Args:
        seed: The seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# Transform PIL.Image to PyTorch Tensor
def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(
        np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
    )
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())


# Dataset with image paths
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """
    Custom dataset that returns image path along with image and label.
    """

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, path) where target is class_index of the target class.
        """
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)


def log_retrival_info(use_retrieval, retrieval_path, target_num):
    """Print information about retrieval settings"""
    if use_retrieval:
        if not retrieval_path:
            print(
                "Warning: use_retrieval=True but data.retrieval_path is not specified. Retrieval cannot proceed."
            )
            use_retrieval = False  # Disable retrieval if path is missing
        elif target_num <= 1:
            print(
                f"Warning: use_retrieval=True but model.target_num is {target_num}. Set target_num > 1 to use retrieval."
            )
            use_retrieval = False  # Disable retrieval if only one target is expected
        else:
            print(
                f"Using retrieval: Loading {target_num - 1} additional targets per image from {retrieval_path}"
            )
    elif target_num > 1:
        print(
            f"Warning: use_retrieval=False but target_num > 1 ({target_num}). Will replicate the primary target {target_num} times."
        )
    else:
        print("Not using retrieval. Using single target image per source.")


def load_retrieved_images(
    primary_target_path: str,
    retrieval_dir: str,
    num_retrieved: int,
    transform_fn: Callable,
    device: str,
) -> List[torch.Tensor]:
    """
    Loads and transforms retrieved images corresponding to a primary target image.

    Args:
        primary_target_path: Path to the original target image.
        retrieval_dir: Base directory where retrieved images are stored.
        num_retrieved: Number of top retrieved images to load (e.g., target_num - 1).
        transform_fn: The transformation function to apply to loaded images.
        device: The device to move the tensors to.

    Returns:
        A list of transformed retrieved image tensors, moved to the specified device.
        Returns an empty list if retrieval_dir is None or retrieval fails.
    """
    if not retrieval_dir or num_retrieved <= 0:
        return []

    retrieved_images = []
    try:
        base_name = os.path.splitext(os.path.basename(primary_target_path))[0]
        img_retrieval_folder = os.path.join(retrieval_dir, base_name)

        if not os.path.isdir(img_retrieval_folder):
            print(f"Warning: Retrieval folder not found: {img_retrieval_folder}")
            return []  # Return empty list if folder doesn't exist

        # Load top 'num_retrieved' images (now starting from 1.jpg)
        for i in range(1, num_retrieved + 1):  # Start from 1, go up to num_retrieved
            img_path = os.path.join(img_retrieval_folder, f"{i}.jpg")
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                transformed_img = transform_fn(img).to(device)
                retrieved_images.append(transformed_img)
            else:
                print(f"Warning: Retrieved image not found: {img_path}. Skipping.")

    except Exception as e:
        print(
            f"Error loading retrieved images for {primary_target_path}: {e}. "
            "Proceeding without retrieved images for this sample."
        )
        return []  # Return empty on error
    if len(retrieved_images) != num_retrieved:
        print(
            f"Warning: Expected {num_retrieved} retrieved images for {primary_target_path}, but found {len(retrieved_images)}."
        )
        while len(retrieved_images) < num_retrieved and len(retrieved_images) > 0:
            retrieved_images.append(retrieved_images[-1])

    return retrieved_images


@hydra.main(version_base=None, config_path="config", config_name="ensemble_3models")
def main(cfg: MainConfig):
    """Hydra entry point."""
    _main(cfg)


def _main(cfg: MainConfig):
    """Main execution function."""
    set_environment()

    # Construct wandb run name with prefix
    prefix = getattr(cfg.wandb, "run_name_prefix", "")
    run_name = f"{prefix}-image_gen" if prefix else "image_gen"

    setup_wandb(cfg, tags=["image_generation", "ensemble_retrieval"], name=run_name)
    wandb.define_metric("batch")
    wandb.define_metric("*", step_metric="batch")

    models = get_models(cfg)
    ensemble_loss = get_ensemble_loss(cfg, models)
    config_hash = hash_training_config(cfg)
    print(f"Saving images into directory hash: {config_hash}")

    # Define image transformations
    transform_fn = transforms.Compose(
        [
            transforms.Resize(
                cfg.model.input_res,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(cfg.model.input_res),
            transforms.ToTensor(),  # Converts to [0, 1] float tensor
        ]
    )

    # Prepare datasets and dataloaders
    clean_data = ImageFolderWithPaths(cfg.data.cle_data_path, transform=transform_fn)
    target_data = ImageFolderWithPaths(cfg.data.tgt_data_path, transform=transform_fn)

    data_loader_imagenet = torch.utils.data.DataLoader(
        clean_data, batch_size=cfg.data.batch_size, shuffle=False, num_workers=4
    )
    # Load only primary target images initially
    data_loader_target = torch.utils.data.DataLoader(
        target_data, batch_size=cfg.data.batch_size, shuffle=False, num_workers=4
    )

    # Define cropping augmentations
    source_crop = [
        transforms.RandomResizedCrop(cfg.model.input_res, scale=[0.5, 1.0]),
        transforms.RandomResizedCrop(cfg.model.input_res, scale=[0.5, 1.0]),
        transforms.RandomResizedCrop(cfg.model.input_res, scale=[0.5, 1.0]),
    ]
    change_iters = [150, 275]

    target_crop = transforms.Compose(
        [
            transforms.RandomResizedCrop(cfg.model.input_res, scale=[0.95, 1.0]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),  # Optional: Keep if needed
        ]
    )

    # Get configuration values
    target_num = getattr(cfg.model, "target_num", 1)
    retrieval_path = getattr(cfg.data, "retrieval_path", None)
    use_retrieval = getattr(cfg.optim, "use_retrieval", False)
    log_retrival_info(use_retrieval, retrieval_path, target_num)

    # Main processing loop
    num_batches = min(
        len(data_loader_imagenet),
        len(data_loader_target),
        cfg.data.num_samples // cfg.data.batch_size
        + (1 if cfg.data.num_samples % cfg.data.batch_size else 0),
    )

    for i, ((image_org, _, path_org), (image_tgt_primary, _, path_tgt)) in enumerate(
        tqdm(
            zip(data_loader_imagenet, data_loader_target),
            total=num_batches,
            desc="Processing Batches",
        )
    ):
        if i * cfg.data.batch_size >= cfg.data.num_samples:
            break

        current_batch_size = image_org.shape[0]
        image_org = image_org.to(cfg.model.device)
        image_tgt_primary = image_tgt_primary.to(cfg.model.device)

        all_image_tgts_list = []

        if use_retrieval:
            # Load retrieved images for each item in the batch
            for batch_idx in range(current_batch_size):
                primary_target_tensor = image_tgt_primary[batch_idx]
                primary_target_path = path_tgt[batch_idx]

                retrieved_tensors = load_retrieved_images(
                    primary_target_path=primary_target_path,
                    retrieval_dir=retrieval_path,
                    num_retrieved=target_num - 1,
                    transform_fn=transform_fn,
                    device=cfg.model.device,
                )
                sample_targets = [primary_target_tensor]
                if len(retrieved_tensors) == target_num - 1:
                    sample_targets.extend(retrieved_tensors)
                else:
                    print(
                        f"Padding target images for {primary_target_path} due to insufficient retrieved images."
                    )
                    num_missing = target_num - 1 - len(retrieved_tensors)
                    sample_targets.extend(retrieved_tensors)
                    sample_targets.extend([primary_target_tensor] * num_missing)

                # Stack tensors for this sample (target_num, C, H, W)
                all_image_tgts_list.append(torch.stack(sample_targets))

            all_image_tgts_stacked = torch.stack(all_image_tgts_list, dim=0)
            all_image_tgts = einops.rearrange(
                all_image_tgts_stacked, "b t c h w -> (b t) c h w"
            )

        else:
            all_image_tgts = image_tgt_primary

        # --- Call the attack function ---
        attack_imgpair(
            cfg=cfg,
            ensemble_loss=ensemble_loss,
            batch_index=i,  # Use batch index instead of img_index
            image_org=image_org,
            path_org=path_org,
            image_tgts=all_image_tgts,  # Pass potentially multiple targets
            source_crop=source_crop,
            target_crop=target_crop,
            target_num=target_num,  # Pass the intended target_num
            config_hash=config_hash,  # Pass hash for saving
            change_iters=change_iters,
        )

    wandb.finish()


def attack_imgpair(
    cfg: MainConfig,
    ensemble_loss: nn.Module,
    batch_index: int,
    image_org: torch.Tensor,
    path_org: List[str],
    image_tgts: torch.Tensor,  # Renamed from image_tgt
    source_crop: Callable,
    target_crop: Callable,
    target_num: int,  # Explicitly receive target_num
    config_hash: str,
    change_iters: List[int],
):
    """
    Performs the adversarial attack for a batch of image pairs.

    Args:
        cfg: Configuration object.
        ensemble_loss: The loss function module.
        batch_index: The index of the current batch.
        image_org: Batch of original source images.
        path_org: List of paths for the original source images.
        image_tgts: Batch of target images (potentially multiple per source,
                   stacked along the batch dim: batch_size * target_num).
        source_crop: Augmentation function for source images.
        target_crop: Augmentation function for target images.
        target_num: The number of target images intended per source image.
        config_hash: Hash string of the config for saving results.
    """
    # Ensure images are on the correct device (might be redundant if already done)
    image_org = image_org.to(cfg.model.device)
    image_tgts = image_tgts.to(cfg.model.device)

    attack_type = cfg.attack

    # Create attacker instance
    attacker = AttackFramework.create(
        attack_type=attack_type,
        cfg=cfg,
        ensemble_loss=ensemble_loss,
        source_crop=source_crop,
        target_crop=target_crop,
        change_iters=change_iters,
    )

    # Generate adversarial images for the batch
    # Pass image_tgts and target_num to the attack method
    # The attack method in base.py expects img_index, we pass batch_index
    adv_images_batch = attacker.attack(
        img_index=batch_index,  # Pass batch index for logging context
        image_org=image_org,
        image_tgt=image_tgts,  # Pass the (potentially stacked) targets
        target_num=target_num,  # Pass the intended number of targets
        log_wandb=False,
        log_interval=25,
    )

    # Save images individually
    current_batch_size = image_org.shape[0]
    for path_idx in range(current_batch_size):
        # Extract original folder and name
        folder, name = os.path.split(path_org[path_idx])
        folder = os.path.basename(
            folder
        )  # Get the immediate parent folder name (e.g., 'n01440764')

        # Define save path using config hash
        folder_to_save = os.path.join(cfg.data.output, "img", config_hash, folder)
        ensure_dir(folder_to_save)

        # Construct save filename (convert JPEG to PNG)
        if name.lower().endswith(".jpeg") or name.lower().endswith(".jpg"):
            save_name = os.path.splitext(name)[0] + ".png"
        elif name.lower().endswith(".png"):
            save_name = name
        else:
            print(f"Warning: Unknown image extension for {name}. Saving as .png")
            save_name = os.path.splitext(name)[0] + ".png"  # Default to png

        save_path = os.path.join(folder_to_save, save_name)

        # Save the corresponding adversarial image
        try:
            torchvision.utils.save_image(adv_images_batch[path_idx], save_path)
        except Exception as e:
            print(f"Error saving image {save_path}: {e}")


if __name__ == "__main__":
    main()
