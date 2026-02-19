import concurrent.futures
import itertools
import os
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import hydra
import openai
import requests
import torch
import torchvision
from google import genai
from omegaconf import OmegaConf
from openai import OpenAI
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

import wandb
from config_schema import MainConfig
from utils import (
    create_batches,
    encode_image,
    ensure_dir,
    get_api_key_count,
    get_api_keys,
    get_output_paths,
    hash_training_config,
    setup_wandb,
)

# Define valid image extensions
VALID_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".JPEG"]


def setup_gemini(api_key: str):
    return genai.Client(api_key=api_key)


def setup_claude(api_key: str):
    return anthropic.Anthropic(api_key=api_key)


def setup_gpt4o(api_key: str):
    return OpenAI(
        api_key=api_key,
    )


def get_media_type(image_path: str) -> str:
    """Get the correct media type based on file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg", ".jpeg"]:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    else:
        raise ValueError(f"Unsupported image extension: {ext}")


class ImageDescriptionGenerator:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

        # Normalize model name for client setup
        client_type = model_name.lower()

        # Use more precise model family detection
        if any(
            gemini_model in client_type
            for gemini_model in [
                "gemini",
                "gemini2.5",
                "gemini2.5pro",
                "gemini2.5flash",
            ]
        ):
            self.client = setup_gemini(api_key)
        elif any(
            claude_model in client_type
            for claude_model in [
                "claude",
                "claude3.7",
                "claude3.7t",
                "claude4",
                "claude4.0",
                "claude-4",
                "claude-4.0",
                "claude4t",
                "claude4.0t",
                "claude-4t",
                "claude-4.0t",
            ]
        ):
            self.client = setup_claude(api_key)
        elif any(
            gpt_model in client_type
            for gpt_model in ["gpt4o", "gpt-4o", "gpt5", "gpt-5"]
        ):
            self.client = setup_gpt4o(api_key)
        elif client_type in ["o3"]:
            self.client = setup_gpt4o(api_key)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def generate_description(self, image_path: str) -> str:
        model_name_lower = self.model_name.lower()
        if model_name_lower == "gemini":
            return self._generate_gemini(image_path)
        elif model_name_lower == "gemini2.5":  # Defaulting gemini2.5 to pro
            return self._generate_gemini25pro(image_path)
        elif model_name_lower == "gemini2.5pro":
            return self._generate_gemini25pro(image_path)
        elif model_name_lower == "gemini2.5flash":
            return self._generate_gemini25flash(image_path)
        elif model_name_lower == "claude":
            return self._generate_claude(image_path)
        elif model_name_lower == "claude3.7":
            return self._generate_claude37(image_path)
        elif model_name_lower == "claude3.7t":  # t for thinking
            return self._generate_claude37_thinking(image_path)
        elif model_name_lower == "gpt4o":
            return self._generate_gpt4o(image_path)
        elif model_name_lower.startswith("gpt5") or model_name_lower.startswith(
            "gpt-5"
        ):
            return self._generate_gpt5(image_path)
        elif model_name_lower == "o3":
            return self._generate_o3(image_path)
        elif model_name_lower in [
            "claude4",
            "claude4.0",
            "claude-4",
            "claude-4.0",
        ]:
            return self._generate_claude40(image_path)
        elif model_name_lower in [
            "claude4t",
            "claude4.0t",
            "claude-4t",
            "claude-4.0t",
        ]:
            return self._generate_claude40_thinking(image_path)
        else:
            raise ValueError(
                f"Generation logic not defined for model: {self.model_name}"
            )

    @retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(10))
    def _generate_gemini(self, image_path: str) -> str:
        image = Image.open(image_path)
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=["Describe this image, no longer than 25 words.", image],
        )
        return response.text.strip()

    def _generate_gemini25pro(self, image_path: str) -> str:
        image = Image.open(image_path)
        response = self.client.models.generate_content(
            model="gemini-2.5-pro-preview-03-25",
            contents=["Describe this image, no longer than 25 words.", image],
        )
        return response.text.strip()

    def _generate_gemini25flash(self, image_path: str) -> str:
        image = Image.open(image_path)
        response = self.client.models.generate_content(
            model="gemini-2.5-flash-preview-04-17",
            contents=["Describe this image, no longer than 25 words.", image],
        )
        return response.text.strip()

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_claude(self, image_path: str) -> str:
        base64_image = encode_image(image_path)
        media_type = get_media_type(image_path)
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",  # Assuming this is the base model for 'claude' alias
            max_tokens=300,
            temperature=1.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one concise sentence, no longer than 20 words.",
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image,
                            },
                        },
                    ],
                }
            ],
        )
        return response.content[0].text

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_claude37_thinking(
        self, image_path: str
    ) -> str:  # Renamed for clarity
        base64_image = encode_image(image_path)
        media_type = get_media_type(image_path)
        response = self.client.messages.create(
            model="claude-3-7-sonnet-20250219",  # Verify if this is the correct model string for Claude 3.7
            max_tokens=1600,
            temperature=1.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one concise sentence, no longer than 20 words.",
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image,
                            },
                        },
                    ],
                }
            ],
            thinking={"type": "enabled", "budget_tokens": 1024},
        )
        return response.content[1].text

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_claude37(self, image_path: str) -> str:
        base64_image = encode_image(image_path)
        media_type = get_media_type(image_path)
        response = self.client.messages.create(
            model="claude-3-7-sonnet-20250219",  # Verify if this is the correct model string for Claude 3.7
            max_tokens=300,
            temperature=1.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one concise sentence, no longer than 20 words.",
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image,
                            },
                        },
                    ],
                }
            ],
        )
        return response.content[0].text

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_claude40(self, image_path: str) -> str:
        base64_image = encode_image(image_path)
        media_type = get_media_type(image_path)
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            temperature=1.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one concise sentence, no longer than 20 words.",
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image,
                            },
                        },
                    ],
                }
            ],
        )
        return response.content[0].text

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_claude40_thinking(self, image_path: str) -> str:
        base64_image = encode_image(image_path)
        media_type = get_media_type(image_path)
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1600,
            temperature=1.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one concise sentence, no longer than 20 words.",
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image,
                            },
                        },
                    ],
                }
            ],
            thinking={"type": "enabled", "budget_tokens": 1024},
        )

        if len(response.content) > 1:
            return response.content[1].text
        return response.content[0].text

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_gpt4o(self, image_path: str) -> str:
        base64_image = encode_image(image_path)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one concise sentence, no longer than 20 words.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=100,
        )
        return response.choices[0].message.content

    def _get_gpt5_reasoning_config(self) -> Optional[Dict[str, str]]:
        name = self.model_name.lower()
        if "thinking" not in name:
            return None

        if "thinking-high" in name:
            return {"reasoning_effort": "high", "verbosity": "high"}
        if "thinking-medium" in name or name.endswith("thinking"):
            return {"reasoning_effort": "medium", "verbosity": "medium"}
        if "thinking-low" in name:
            return {"reasoning_effort": "low", "verbosity": "low"}
        if "thinking-minimal" in name:
            return {"reasoning_effort": "minimal", "verbosity": "low"}
        return {"reasoning_effort": "medium", "verbosity": "medium"}

    def _generate_gpt5(self, image_path: str) -> str:
        reasoning_config = self._get_gpt5_reasoning_config()
        if reasoning_config:
            return self._generate_gpt5_with_reasoning(image_path, reasoning_config)
        return self._generate_gpt5_standard(image_path)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_gpt5_standard(self, image_path: str) -> str:
        base64_image = encode_image(image_path)
        response = self.client.chat.completions.create(
            model="gpt-5-2025-08-07",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one concise sentence, no longer than 20 words.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_completion_tokens=400,
        )
        return response.choices[0].message.content

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_gpt5_with_reasoning(
        self, image_path: str, reasoning_config: Dict[str, str]
    ) -> str:
        base64_image = encode_image(image_path)
        reasoning_effort = reasoning_config.get("reasoning_effort", None)
        max_token = 400 if reasoning_effort == "low" else 4000
        response = self.client.chat.completions.create(
            model="gpt-5-2025-08-07",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one concise sentence, no longer than 20 words.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_completion_tokens=max_token,
            reasoning_effort=reasoning_config["reasoning_effort"],
        )
        return response.choices[0].message.content

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_o3(self, image_path: str) -> str:
        base64_image = encode_image(image_path)
        response = self.client.chat.completions.create(
            model="o3",
            reasoning_effort="medium",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one concise sentence, no longer than 20 words.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_completion_tokens=5000,
        )
        return response.choices[0].message.content


def save_descriptions(descriptions: List[Tuple[str, str]], output_file: str):
    """Save image descriptions to file."""
    ensure_dir(os.path.dirname(output_file))
    with open(output_file, "w", encoding="utf-8") as f:
        for filename, desc in descriptions:
            f.write(f"{filename}: {desc}\n")


class ParallelImageDescriptionProcessor:
    """
    Processes images in parallel using multiple API models with multiple API keys.

    Uses threading to make concurrent API calls to different models and
    distributes requests across multiple API keys to avoid rate limits.
    """

    def __init__(
        self, model_names: List[str], output_dir: str, parallel_images: int = 4
    ):
        """
        Initialize with multiple model generators.

        Args:
            model_names: List of model names to use (e.g., ["gpt4o", "claude", "gemini"])
            output_dir: Directory to save description outputs
            parallel_images: Number of images to process in parallel
        """
        self.model_names = model_names
        self.output_dir = output_dir
        self.parallel_images = parallel_images
        self.descriptions = {model: {"tgt": [], "adv": []} for model in model_names}
        self.lock = Lock()  # For thread-safe operations

        # Create model clients with all available API keys
        self.model_clients = {}
        for model_name in model_names:
            # Map specific model names to base names for API key retrieval
            api_key_model_name = model_name.lower()
            if "gemini" in api_key_model_name:
                api_key_model_name = "gemini"
            elif "claude" in api_key_model_name:
                api_key_model_name = "claude"
            elif (
                "gpt" in api_key_model_name
            ):  # Assuming gpt4o keys might be stored under 'gpt4o' or similar
                api_key_model_name = (
                    "gpt4o"  # Adjust if your key file uses a different base name
                )
            elif "o3" in api_key_model_name:  # Added for o3
                api_key_model_name = "gpt4o"  # Assuming o3 keys are stored under 'o3'

            api_keys = get_api_keys(api_key_model_name)
            if not api_keys:
                print(
                    f"Warning: No API keys found for base model '{api_key_model_name}' (derived from '{model_name}'). Skipping {model_name}."
                )
                self.model_clients[model_name] = []
                continue

            # Important: Initialize the generator with the *original* model name
            self.model_clients[model_name] = [
                ImageDescriptionGenerator(model_name, api_key) for api_key in api_keys
            ]

    def process_batch(self, image_pairs: List[Tuple[str, str, str]]):
        """
        Process a batch of image pairs with all models, distributing across API keys.

        Args:
            image_pairs: List of tuples (file, tgt_path, adv_path)
        """
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}

            # Submit tasks for all models and image pairs
            for model_name in self.model_names:
                # Get all generators for this model
                generators = self.model_clients[model_name]
                if not generators:
                    print(f"Warning: No API keys available for {model_name}, skipping")
                    continue

                # Pair each image with a generator (API key), cycling through generators
                for (file, tgt_path, adv_path), generator in zip(
                    image_pairs, itertools.cycle(generators)
                ):

                    # Target image task
                    tgt_future = executor.submit(
                        self._generate_and_capture_description,
                        model_name,
                        generator,
                        tgt_path,
                        file,
                        "tgt",
                    )
                    futures[tgt_future] = (model_name, file, "tgt")

                    # Adversarial image task
                    adv_future = executor.submit(
                        self._generate_and_capture_description,
                        model_name,
                        generator,
                        adv_path,
                        file,
                        "adv",
                    )
                    futures[adv_future] = (model_name, file, "adv")

            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                model_name, file, img_type = futures[future]
                try:
                    # Result is already stored in self.descriptions
                    future.result()
                except Exception as e:
                    print(f"Error with {model_name} on {file} ({img_type}): {e}")

    def _generate_and_capture_description(
        self,
        model_name: str,
        generator: ImageDescriptionGenerator,
        image_path: str,
        file: str,
        img_type: str,
    ) -> str:
        """
        Generate a description and store it in the appropriate collection.

        Args:
            model_name: Name of the model
            generator: The generator instance
            image_path: Path to the image
            file: File name for logging
            img_type: Image type ('tgt' or 'adv')

        Returns:
            The generated description
        """
        try:
            description = generator.generate_description(image_path)

            # Thread-safe update of descriptions and logging
            with self.lock:
                self.descriptions[model_name][img_type].append((file, description))
                wandb.log({f"descriptions/{model_name}/{file}/{img_type}": description})

            return description
        except Exception as e:
            print(
                f"Error generating description with {model_name} for {image_path}: {e}"
            )
            raise

    def process_all_images(self, image_pairs: List[Tuple[str, str, str]]):
        """
        Process all image pairs in batches to efficiently manage API rate limits.

        Args:
            image_pairs: List of tuples (file, tgt_path, adv_path)
        """
        # Create batches of image pairs
        batches = create_batches(image_pairs, self.parallel_images)

        # Process each batch
        for i, batch in enumerate(tqdm(batches, desc="Processing image batches")):
            print(f"Processing batch {i+1}/{len(batches)} ({len(batch)} images)")
            self.process_batch(batch)

    def save_all_descriptions(self):
        """Save descriptions for all models to their respective files."""
        for model_name in self.model_names:
            # Save target descriptions
            save_descriptions(
                self.descriptions[model_name]["tgt"],
                os.path.join(self.output_dir, f"target_{model_name}.txt"),
            )

            # Save adversarial descriptions
            save_descriptions(
                self.descriptions[model_name]["adv"],
                os.path.join(self.output_dir, f"adversarial_{model_name}.txt"),
            )


@hydra.main(version_base=None, config_path="config", config_name="ensemble_3models")
def main(cfg: MainConfig):
    _main(cfg)


def _main(cfg: MainConfig):
    # Initialize wandb using shared utility
    setup_wandb(cfg)

    # Get config hash and setup paths
    if cfg.get("generated_img_hash") is not None:
        config_hash = cfg.generated_img_hash
        print(f"Using provided generated_img_hash: {config_hash}")
    else:
        config_hash = hash_training_config(cfg)
        print(f"Using training output for config hash: {config_hash}")

    # Get output paths using shared utility
    paths = get_output_paths(cfg, config_hash)
    ensure_dir(paths["desc_output_dir"])

    try:
        # Initialize parallel image processor with all models
        model_names = cfg.blackbox.model_name
        if isinstance(model_names, str):
            model_names = [model_names]  # Convert to list if a single string

        # Get parallel image processing parameter
        parallel_images = getattr(cfg.blackbox, "parallel_images", 4)

        processor = ParallelImageDescriptionProcessor(
            model_names=model_names,
            output_dir=paths["desc_output_dir"],
            parallel_images=parallel_images,
        )

        # Collect all image pairs first
        print("Collecting image pairs...")
        image_pairs = []
        for root, _, files in os.walk(paths["output_dir"]):
            for file in tqdm(files, desc="Finding image pairs"):
                # Check if file has valid image extension
                if any(
                    file.lower().endswith(ext.lower()) for ext in VALID_IMAGE_EXTENSIONS
                ):
                    try:
                        # Get adversarial path
                        adv_path = os.path.join(root, file)
                        # Extract just the filename without extension
                        filename_base = os.path.splitext(os.path.basename(adv_path))[0]

                        # Try each valid extension for target image
                        target_found = False
                        tgt_path = None
                        for ext in VALID_IMAGE_EXTENSIONS:
                            candidate_path = os.path.join(
                                cfg.data.tgt_data_path, "1", filename_base + ext
                            )
                            if os.path.exists(candidate_path):
                                tgt_path = candidate_path
                                target_found = True
                                break

                        if target_found:
                            # Add to image pairs
                            image_pairs.append((file, tgt_path, adv_path))
                        else:
                            print(
                                f"Target image not found for {filename_base} with any valid extension, skip it."
                            )

                    except Exception as e:
                        print(f"Error processing {file}: {e}")

        # Process all image pairs
        print(f"Processing {len(image_pairs)} image pairs...")
        processor.process_all_images(image_pairs)

        # Save all descriptions
        processor.save_all_descriptions()
        print(f"Descriptions saved to {paths['desc_output_dir']}")

    except (FileNotFoundError, KeyError) as e:
        print(f"Error: {e}")
        return

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
