"""
m_attack_v2.py — Standalone script to generate adversarial images
and save them flat into the m_attack_v2_output/ folder.

Designed for a SINGLE GPU (no multiprocessing). Iterates the source/target
dataloaders sequentially, runs the M-Attack-V2 attack, and saves each
adversarial image as <name>.png directly inside data.output.

Usage:
    uv run python m_attack_v2.py

Config comes from config/ensemble_3models.yaml via Hydra:
    - data.num_samples    (1000)
    - data.output         ("./m_attack_v2_output")
    - data.cle_data_path  (source / clean images)
    - data.tgt_data_path  (target images)
    - model.device        ("cuda:0")

------------------------------------------------------------------------------
What are "retrieved target images"?
------------------------------------------------------------------------------
M-Attack-V2 pushes a clean image's features toward a TARGET image's features.
Instead of relying on a single target image (which gives a noisy/unstable
semantic reference), it also uses a few extra images that are semantically
similar to that target. These extras are the "retrieved target images".

- They are pre-computed offline by retrieval.py (CLIP nearest-neighbours from
  a COCO reference pool) and stored under:
      resources/retrieved_embeddings/<target_stem>/1.jpg, 2.jpg, ...
- model.target_num = N  means: 1 primary target + (N-1) retrieved targets.
- optim.use_retrieval = true enables loading them.
- This is the paper's ATA (Auxiliary Target Alignment): averaging alignment
  over the primary target plus its neighbours gives a more stable target
  signal and a stronger attack.
- If some retrieved images are missing, we pad with the primary target.
"""

import os
import random
from typing import Callable, List

# Cap thread usage at 4 (env vars must be set before heavy imports)
os.environ["NUMEXPR_MAX_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
# Disable Weights & Biases entirely for this standalone script
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

import einops
import hydra
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from tqdm import tqdm

from attack import AttackFramework
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

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_TO_CLASS = {}
for _dict, _cls in [
    (BLIP_MODEL_DICT, BlipFeatureExtractor),
    (DINOV2_MODEL_DICT, Dinov2FeatureExtractor),
    (CLIP_MODEL_DICT, ClipFeatureExtractor),
    (BLIP1_MODEL_DICT, Blip1FeatureExtractor),
    (VIT_MODEL_DICT, VisionTransformerFeatureExtractor),
]:
    MODEL_TO_CLASS.update({name: _cls for name in _dict})


# Cap total parallelism at 4 threads.
# Two dataloaders run concurrently (source + target), so 2 workers each = 4 total.
MAX_THREADS = 4
DATALOADER_WORKERS = MAX_THREADS // 2  # 2 workers per loader -> 4 total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 2023):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)


def build_models(cfg: MainConfig, device: str) -> List[nn.Module]:
    models = []
    for backbone_name in cfg.model.backbone:
        if backbone_name not in MODEL_TO_CLASS:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        model = (
            MODEL_TO_CLASS[backbone_name](backbone_name)
            .eval()
            .to(device)
            .requires_grad_(False)
        )
        models.append(model)
    return models


def build_loss(cfg: MainConfig, models: List[nn.Module]) -> nn.Module:
    beta = getattr(cfg.optim, "beta", 1.0)
    return EnsWeightedMultiAlignmentLoss(models, beta=beta)


def load_retrieved_images(
    primary_target_path: str,
    retrieval_dir: str,
    num_retrieved: int,
    transform_fn: Callable,
    device: str,
) -> List[torch.Tensor]:
    """Load the (num_retrieved) neighbour images for a given primary target."""
    if not retrieval_dir or num_retrieved <= 0:
        return []

    retrieved = []
    base_name = os.path.splitext(os.path.basename(primary_target_path))[0]
    folder = os.path.join(retrieval_dir, base_name)

    if not os.path.isdir(folder):
        return []

    for i in range(1, num_retrieved + 1):
        img_path = os.path.join(folder, f"{i}.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            retrieved.append(transform_fn(img).to(device))

    # Pad with the last found neighbour if fewer than expected
    while len(retrieved) < num_retrieved and len(retrieved) > 0:
        retrieved.append(retrieved[-1])

    return retrieved


def build_target_tensor(
    image_tgt_primary: torch.Tensor,
    path_tgt: List[str],
    use_retrieval: bool,
    target_num: int,
    retrieval_path: str,
    transform_fn: Callable,
    device: str,
) -> torch.Tensor:
    """Stack primary + retrieved targets into (batch_size * target_num, C, H, W)."""
    if not (use_retrieval and target_num > 1):
        return image_tgt_primary

    batch_size = image_tgt_primary.shape[0]
    all_targets = []
    for b in range(batch_size):
        primary = image_tgt_primary[b]
        retrieved = load_retrieved_images(
            primary_target_path=path_tgt[b],
            retrieval_dir=retrieval_path,
            num_retrieved=target_num - 1,
            transform_fn=transform_fn,
            device=device,
        )
        sample_targets = [primary]
        sample_targets.extend(retrieved)
        # Pad with primary if retrieval came up short
        while len(sample_targets) < target_num:
            sample_targets.append(primary)
        all_targets.append(torch.stack(sample_targets))

    stacked = torch.stack(all_targets, dim=0)
    return einops.rearrange(stacked, "b t c h w -> (b t) c h w")


def run_attack_batch(
    cfg: MainConfig,
    attacker: AttackFramework,
    batch_index: int,
    image_org: torch.Tensor,
    image_tgts: torch.Tensor,
    target_num: int,
    device: str,
) -> torch.Tensor:
    """Run the attack on one batch and return the adversarial image tensor."""
    image_org = image_org.to(device)
    image_tgts = image_tgts.to(device)

    multi_pass_num = getattr(cfg.optim, "multi_pass_num", 1)
    kwargs = {
        "img_index": batch_index,
        "image_org": image_org,
        "image_tgt": image_tgts,
        "target_num": target_num,
        "log_wandb": False,
        "log_interval": 1,
        "disable_tqdm": True,  # the outer progress bar tracks batches
    }
    if cfg.attack == "pgd_multi_pass":
        kwargs["multi_pass_num"] = multi_pass_num

    return attacker.attack(**kwargs)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="config", config_name="ensemble_3models")
def main(cfg: MainConfig):
    set_seed()
    torch.set_num_threads(MAX_THREADS)

    # Resolve device — single GPU
    device = getattr(cfg.model, "device", "cuda:0")
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU (very slow).")
        device = "cpu"

    output_dir = cfg.data.output
    ensure_dir(output_dir)
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"Device: {device}  |  Max threads: {MAX_THREADS}  |  wandb: disabled")
    print(f"Num samples: {cfg.data.num_samples}  |  Batch size: {cfg.data.batch_size}")
    print(f"Attack: {cfg.attack}  |  Backbones: {list(cfg.model.backbone)}")

    # Transforms
    transform_fn = transforms.Compose([
        transforms.Resize(
            cfg.model.input_res,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(cfg.model.input_res),
        transforms.ToTensor(),
    ])

    # Datasets
    full_source = ImageFolderWithPaths(cfg.data.cle_data_path, transform=transform_fn)
    full_target = ImageFolderWithPaths(cfg.data.tgt_data_path, transform=transform_fn)

    num_samples = min(cfg.data.num_samples, len(full_source), len(full_target))
    print(f"Processing {num_samples} samples")

    indices = list(range(num_samples))
    source_subset = torch.utils.data.Subset(full_source, indices)
    target_subset = torch.utils.data.Subset(full_target, indices)

    loader_source = torch.utils.data.DataLoader(
        source_subset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=DATALOADER_WORKERS,
    )
    loader_target = torch.utils.data.DataLoader(
        target_subset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=DATALOADER_WORKERS,
    )

    # Retrieval config
    target_num = getattr(cfg.model, "target_num", 1)
    retrieval_path = getattr(cfg.data, "retrieval_path", None)
    use_retrieval = getattr(cfg.optim, "use_retrieval", False)

    # Build models, loss and attacker once (single GPU, reused across batches)
    models = build_models(cfg, device)
    ensemble_loss = build_loss(cfg, models)

    source_crop = [
        transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale_1),
        transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale_2),
        transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale_3),
    ]
    target_crop = transforms.Compose([
        transforms.RandomResizedCrop(cfg.model.input_res, scale=[0.95, 1.0]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
    ])

    attacker = AttackFramework.create(
        attack_type=cfg.attack,
        cfg=cfg,
        ensemble_loss=ensemble_loss,
        source_crop=source_crop,
        target_crop=target_crop,
        change_iters=cfg.model.changer_iters,
    )

    # Sequential loop over batches (single GPU)
    num_batches = min(len(loader_source), len(loader_target))
    saved = 0
    pbar = tqdm(
        zip(loader_source, loader_target),
        total=num_batches,
        desc="Generating adversarial images",
    )
    for batch_i, (
        (image_org, _, path_org),
        (image_tgt_primary, _, path_tgt),
    ) in enumerate(pbar):
        image_org = image_org.to(device)
        image_tgt_primary = image_tgt_primary.to(device)

        image_tgts = build_target_tensor(
            image_tgt_primary=image_tgt_primary,
            path_tgt=path_tgt,
            use_retrieval=use_retrieval,
            target_num=target_num,
            retrieval_path=retrieval_path,
            transform_fn=transform_fn,
            device=device,
        )

        adv_images = run_attack_batch(
            cfg=cfg,
            attacker=attacker,
            batch_index=batch_i,
            image_org=image_org,
            image_tgts=image_tgts,
            target_num=target_num,
            device=device,
        )

        # Save adversarial images flat into output_dir as <name>.png
        for b in range(image_org.shape[0]):
            name = os.path.basename(path_org[b])
            save_name = os.path.splitext(name)[0] + ".png"
            save_path = os.path.join(output_dir, save_name)
            try:
                torchvision.utils.save_image(adv_images[b], save_path)
                saved += 1
            except Exception as e:
                print(f"Error saving {save_path}: {e}")

        pbar.set_postfix(saved=saved)

    print(f"\nDone! Generated {saved} adversarial images in: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
