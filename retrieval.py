import os
import torch
import numpy as np
import shutil
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
try:
    from torchgeo.datasets import PatternNet
except ImportError:
    PatternNet = None
    print("Warning: torchgeo not available. PatternNet dataset will not be supported.")


def resolve_device(device: str = "auto") -> torch.device:
    """
    Resolve runtime device for retrieval.

    Args:
        device: "auto", "cpu", "cuda", or "cuda:<index>"

    Returns:
        torch.device selected for retrieval.
    """
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)

    if resolved.type == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")

    if resolved.type == "cuda" and resolved.index is not None:
        count = torch.cuda.device_count()
        if resolved.index >= count:
            if count > 0:
                print(
                    f"Warning: Requested {resolved} but only {count} CUDA device(s) "
                    "available. Falling back to cuda:0."
                )
                return torch.device("cuda:0")
            print("Warning: CUDA requested but no devices found. Falling back to CPU.")
            return torch.device("cpu")

    return resolved


class ImageEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch16", device=None):
        """
        Initialize the ImageEmbedder with CLIP model from Hugging Face

        Args:
            model_name: HuggingFace CLIP model name
            device: torch device to use (will use CUDA if available when None)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"Using device: {self.device}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.text_model = None  # remove text model
        self.model.text_projection = None
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed_dataset_batched(self, dataset, batch_size=32, num_workers=4):
        """
        Generate embeddings for images in a dataset using batched processing.
        Assumes the dataset returns pre-processed tensors.

        Args:
            dataset: A dataset instance (like ImageFolderWithPaths) that yields (processed_tensor, label, path)
            batch_size: Batch size for processing
            num_workers: Number of workers for data loading

        Returns:
            Dictionary mapping image paths to embeddings (as tensors)
        """
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # CRITICAL: Must be False for consistent PatternNet indexing
            num_workers=num_workers,
            pin_memory=True,  # Helps speed up CPU to GPU transfer
        )

        embeddings = {}

        for batch in tqdm(dataloader, desc="Embedding images in batches"):
            # Unpack the batch - pixel_values_batch, labels_batch, paths_batch
            # The pixel_values are already processed tensors by the dataset's __getitem__
            pixel_values_batch, _, paths = batch

            # Move the batch to the correct device
            pixel_values_batch = pixel_values_batch.to(self.device)

            with torch.no_grad():
                # Pass the batch directly to the model
                outputs = self.model.get_image_features(pixel_values=pixel_values_batch)

            # Normalize the features
            image_features = outputs / outputs.norm(dim=-1, keepdim=True)

            # Store embeddings with paths as keys - store as tensors directly
            for i, path in enumerate(paths):
                embeddings[path] = image_features[i].cpu()  # Store as tensor

        return embeddings

    def save_embeddings(self, embeddings, save_path):
        """
        Save embeddings to a file using PyTorch's native save

        Args:
            embeddings: Dictionary mapping paths to embedding tensors
            save_path: Path to save the embeddings
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Ensure all values are tensors
        processed_embeddings = {
            path: torch.from_numpy(emb) if isinstance(emb, np.ndarray) else emb
            for path, emb in embeddings.items()
        }

        try:
            torch.save(processed_embeddings, save_path)
            print(f"Embeddings saved to {save_path}")
        except Exception as e:
            print(f"Error saving embeddings: {e}")

    def load_embeddings(self, load_path):
        """
        Load embeddings from a file using PyTorch's native load.
        Embeddings are always loaded onto the CPU first for consistent handling.

        Args:
            load_path: Path to load the embeddings from

        Returns:
            Dictionary of embeddings (tensors on CPU)
        """
        try:
            # Load embeddings onto CPU first
            embeddings = torch.load(load_path, map_location="cpu")
            print(f"Loaded embeddings from {load_path} to CPU")
            return embeddings
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return None


class ImageRetriever:
    def __init__(
        self, embedder=None, target_embeddings=None, reference_embeddings=None
    ):
        """
        Initialize the ImageRetriever

        Args:
            embedder: ImageEmbedder instance (optional)
            target_embeddings: Pre-loaded target embeddings (optional)
            reference_embeddings: Pre-loaded reference embeddings (optional)
        """
        self.embedder = embedder
        self.target_embeddings = target_embeddings
        self.reference_embeddings = reference_embeddings

    def load_embeddings(self, target_path=None, reference_path=None):
        """
        Load embeddings from files using PyTorch's native load

        Args:
            target_path: Path to target embeddings file
            reference_path: Path to reference embeddings file
        """
        if target_path and os.path.exists(target_path):
            try:
                self.target_embeddings = torch.load(target_path)
                print(f"Loaded target embeddings from {target_path}")
            except Exception as e:
                print(f"Error loading target embeddings with torch: {e}")
                self.target_embeddings = None

        if reference_path and os.path.exists(reference_path):
            try:
                self.reference_embeddings = torch.load(reference_path)
                print(f"Loaded reference embeddings from {reference_path}")
            except Exception as e:
                print(f"Error loading reference embeddings with torch: {e}")
                self.reference_embeddings = None

    def compute_similarities(self, query_embedding):
        """
        Compute similarities between query embedding and reference embeddings.
        Ensures tensors are on the correct device before computation.

        Args:
            query_embedding: Query embedding tensor (can be on any device)

        Returns:
            Dictionary mapping image paths to similarity scores
        """
        if self.reference_embeddings is None:
            raise ValueError("Reference embeddings not loaded")
        if self.embedder is None:
            raise ValueError("Embedder instance is required for device information.")

        target_device = self.embedder.device
        similarities = {}

        # Ensure query is a tensor and move to target device
        if isinstance(query_embedding, np.ndarray):
            query_embedding = torch.from_numpy(query_embedding)
        query_embedding = query_embedding.to(target_device)

        # Simple dictionary of path -> tensor (currently on CPU due to load_embeddings)
        for path, ref_embedding in self.reference_embeddings.items():
            # Ensure reference is a tensor
            if isinstance(ref_embedding, np.ndarray):
                ref_embedding = torch.from_numpy(ref_embedding)

            # Move reference embedding to target device for computation
            ref_embedding_on_device = ref_embedding.to(target_device)

            # Compute similarity on the target device
            similarity = torch.nn.functional.cosine_similarity(
                query_embedding, ref_embedding_on_device.unsqueeze(0)
            ).item()

            similarities[path] = {"similarity": similarity, "path": path}

        return similarities

    def retrieve_similar_images(
        self, query_image_path=None, query_image_name=None, top_k=5
    ):
        """
        Retrieve top-k similar images to a query

        Args:
            query_image_path: Path to query image (optional)
            query_image_name: Name of query image in target embeddings (optional)
            top_k: Number of similar images to retrieve

        Returns:
            List of top-k similar images with similarity scores
        """
        if not query_image_path and not query_image_name:
            raise ValueError(
                "Either query_image_path or query_image_name must be provided"
            )

        if self.reference_embeddings is None:
            raise ValueError("Reference embeddings not loaded")

        # Get query embedding
        if query_image_path:
            if self.embedder is None:
                raise ValueError("Embedder must be provided to embed new images")

            # Use the embedder to get the embedding for the image path
            image = Image.open(query_image_path).convert("RGB")
            inputs = self.embedder.processor(images=image, return_tensors="pt").to(
                self.embedder.device
            )

            with torch.no_grad():
                outputs = self.embedder.model.get_image_features(**inputs)

            # Normalize the features (remains on self.embedder.device)
            query_embedding = outputs / outputs.norm(dim=-1, keepdim=True)
            # query_embedding = query_embedding.cpu() # Removed: Keep on embedder device
        else:
            if self.target_embeddings is None:
                raise ValueError("Target embeddings not loaded")

            if query_image_name not in self.target_embeddings:
                raise ValueError(
                    f"Query image {query_image_name} not found in target embeddings"
                )

            # Simple dictionary access (embedding is on CPU due to load_embeddings)
            query_embedding = self.target_embeddings[query_image_name]

        # Compute similarities (will handle moving query_embedding to device)
        similarities = self.compute_similarities(query_embedding)

        # Sort by similarity
        sorted_similarities = sorted(
            similarities.items(), key=lambda x: x[1]["similarity"], reverse=True
        )

        # Return top-k
        return sorted_similarities[:top_k]


# ImageFolderWithPaths for structured data
class PatternNetDataset(torch.utils.data.Dataset):
    def __init__(self, root, processor=None, device=None):
        """
        PatternNet dataset wrapper that yields (processed_tensor, label, index)
        
        Args:
            root: Path to PatternNet dataset root directory
            processor: CLIP processor for image preprocessing
            device: Torch device to map tensors to
        """
        if PatternNet is None:
            raise ImportError("torchgeo is required for PatternNet dataset. Install with: pip install torchgeo")
        
        self.dataset = PatternNet(root=root, download=True)
        self.processor = processor
        self.device = device
        if self.processor is None:
            raise ValueError("A CLIP processor must be provided.")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        """
        Fetches an item from the PatternNet dataset.
        
        Returns:
            Tuple: (processed_image_tensor, label, index_str)
        """
        try:
            sample = self.dataset[index]
            image_tensor = sample['image']  # Shape: (C, H, W)
            
            # Convert tensor to PIL Image for CLIP processing
            if isinstance(image_tensor, torch.Tensor):
                # Convert from (C, H, W) to (H, W, C) and ensure uint8
                image_np = image_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
                image = Image.fromarray(image_np)
            else:
                image = image_tensor
            
            # Process the image using the CLIP processor
            inputs = self.processor(
                images=image, return_tensors="pt", padding=True, truncation=True
            )
            
            # Extract the pixel values tensor (remove the batch dimension)
            pixel_values = inputs["pixel_values"].squeeze(0)
            
            # Use index as identifier - this allows us to retrieve the original image later
            index_str = str(index)
            
            return pixel_values, 0, index_str  # Return processed tensor, dummy label, and index
            
        except Exception as e:
            print(f"Error loading PatternNet sample {index}: {e}")
            # Return dummy tensor in case of error
            dummy_tensor = torch.zeros((3, 224, 224))
            return dummy_tensor, 0, f"{index}_error"
    
    def get_original_image(self, index):
        """
        Get the original image tensor for a given index
        
        Args:
            index: Dataset index
            
        Returns:
            Original image tensor (C, H, W) in [0, 1] range
        """
        try:
            sample = self.dataset[index]
            image_tensor = sample['image']  # Shape: (C, H, W)
            
            # Convert to [0, 1] range if needed
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor.float() / 255.0
            
            return image_tensor
        except Exception as e:
            print(f"Error getting original image for index {index}: {e}")
            return torch.zeros((3, 224, 224))


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __init__(self, root, processor=None, device=None):
        """
        Initializes the dataset.

        Args:
            root: Path to the root directory of the dataset.
            processor: CLIP processor for image preprocessing.
            device: Torch device to map tensors to.
        """
        super().__init__(root=root, transform=None)  # No default transform needed
        self.processor = processor
        self.device = device
        if self.processor is None:
            raise ValueError("A CLIP processor must be provided.")

    def __getitem__(self, index):
        """
        Fetches an item from the dataset.

        Returns:
            Tuple: (processed_image_tensor, dummy_label, path)
        """
        # Get the path and load the image
        path, _ = self.samples[index]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return None or a placeholder if an image fails to load
            # For simplicity, let's return None here and handle it in the DataLoader if needed
            # A better approach might be to return a dummy tensor and path
            # Or skip problematic files during dataset creation
            # Here we'll try returning a dummy tensor of the expected shape
            # Assuming the processor gives a standard size, e.g., (3, 224, 224)
            # Note: This requires knowing the processor's output shape or handling errors upstream
            dummy_tensor = torch.zeros((3, 224, 224))  # Example size
            return dummy_tensor, 0, path  # Return dummy tensor

        # Process the image using the CLIP processor
        inputs = self.processor(
            images=image, return_tensors="pt", padding=True, truncation=True
        )

        # Extract the pixel values tensor (remove the batch dimension added by processor)
        pixel_values = inputs["pixel_values"].squeeze(0)

        # Return the processed tensor, dummy label, and path
        return pixel_values, 0, path


def embed_dataset_batched(
    dataset_root, save_path, dataset_type="coco", batch_size=32, num_workers=4, device=None
):
    """
    Convenience function to embed all images in a dataset using batched processing.
    Loads embeddings from save_path if it exists, otherwise computes and saves them.

    Args:
        dataset_root: Path to dataset root
        save_path: Path to save/load the embeddings (.pt file)
        dataset_type: Type of dataset ("coco" or "patternnet")
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading
        device: Torch device to use

    Returns:
        Tuple: (Dictionary of embeddings, ImageEmbedder instance)
    """
    # Check if embeddings file already exists
    if os.path.exists(save_path):
        print(f"Loading existing embeddings from {save_path}")
        try:
            # Need an embedder instance to load embeddings correctly (for map_location)
            # Initialize a temporary embedder just for loading if needed
            temp_embedder = ImageEmbedder(device=device)
            embeddings = temp_embedder.load_embeddings(save_path)
            if embeddings:
                # Return the loaded embeddings and the temporary embedder
                # Or potentially return None for the embedder if loading?
                # Let's return the temp_embedder for consistency, the caller might need it.
                return embeddings, temp_embedder
            else:
                print(f"Failed to load embeddings from {save_path}. Recomputing...")
        except Exception as e:
            print(f"Error loading embeddings from {save_path}: {e}. Recomputing...")

    # If embeddings don't exist or loading failed, compute them
    # Create directory for embeddings if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Initialize embedder
    print("Initializing embedder...")
    embedder = ImageEmbedder(device=device)

    # Load dataset based on type
    print(f"Loading {dataset_type} dataset from {dataset_root}")
    try:
        if dataset_type.lower() == "patternnet":
            dataset = PatternNetDataset(
                root=dataset_root, processor=embedder.processor, device=device
            )
        else:  # Default to ImageFolderWithPaths for COCO and other structured datasets
            dataset = ImageFolderWithPaths(
                root=dataset_root, processor=embedder.processor, device=device
            )
        
        print(f"Found {len(dataset)} images in the {dataset_type} dataset")
        if len(dataset) == 0:
            print(
                f"Warning: No images found in {dataset_root}. Check the path and directory structure."
            )
            # Return empty embeddings and the embedder instance
            return {}, embedder
    except FileNotFoundError:
        print(f"Error: Dataset directory not found at {dataset_root}")
        # Return empty embeddings and the embedder instance
        return {}, embedder
    except Exception as e:
        print(f"Error loading {dataset_type} dataset from {dataset_root}: {e}")
        # Return empty embeddings and the embedder instance
        return {}, embedder

    # Process images
    print("Embedding images in batches...")
    embeddings = embedder.embed_dataset_batched(
        dataset=dataset, batch_size=batch_size, num_workers=num_workers
    )

    # Save embeddings
    embedder.save_embeddings(embeddings, save_path)
    print(f"Embeddings saved to {save_path}")

    return embeddings, embedder  # Return computed embeddings and the embedder


def embed_coco_dataset_batched(
    coco_root, save_path, batch_size=32, num_workers=4, device=None
):
    """
    Backward compatibility wrapper for embed_dataset_batched with COCO dataset.
    
    Args:
        coco_root: Path to COCO dataset root
        save_path: Path to save/load the embeddings (.pt file)
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading
        device: Torch device to use

    Returns:
        Tuple: (Dictionary of embeddings, ImageEmbedder instance)
    """
    return embed_dataset_batched(
        dataset_root=coco_root,
        save_path=save_path,
        dataset_type="coco",
        batch_size=batch_size,
        num_workers=num_workers,
        device=device
    )


def main(dataset_type="coco", target_images_dir=None, device="auto"):
    """
    Example usage of the retrieval system
    
    Args:
        dataset_type: Type of dataset to use for retrieval ("coco" or "patternnet")
        target_images_dir: Path to target images directory (overrides default)
        device: Runtime device ("auto", "cpu", "cuda", or "cuda:<index>")
    """
    # Configuration
    if dataset_type.lower() == "patternnet":
        # PatternNet configuration
        reference_root = "/data/spiderman/pattern_net"  # PatternNet dataset path
        reference_save_path = "resources/embeddings/patternnet_embeddings.pt"
        retrieved_embeddings_base_dir = "resources/retrieved_embeddings_pattern_net"
    else:
        # COCO configuration (default)
        reference_root = "resources/images/coco"
        reference_save_path = "resources/embeddings/coco_embeddings.pt"
        retrieved_embeddings_base_dir = "resources/retrieved_embeddings"
    
    # Target images are the same regardless of reference dataset
    if target_images_dir is None:
        target_images_dir = "resources/images/target_images_100"  # Updated to match config
    device = resolve_device(device)
    target_save_path = "resources/embeddings/target_embeddings.pt"

    print(f"Using {reference_root} as reference dataset root path.")
    print(f"Using retrieval device: {device}")
    if dataset_type.lower() == "patternnet":
        print("Using PatternNet dataset for retrieval.")
    else:
        print(f"Using COCO dataset. Images should be in {reference_root}/train2014/")

    # Embed reference dataset (COCO or PatternNet)
    reference_embeddings, reference_embedder = embed_dataset_batched(
        dataset_root=reference_root,
        save_path=reference_save_path,
        dataset_type=dataset_type,
        batch_size=32,
        num_workers=4,
        device=device,
    )
    
    # Store reference dataset for PatternNet image retrieval
    reference_dataset = None
    if dataset_type.lower() == "patternnet":
        reference_dataset = PatternNetDataset(
            root=reference_root, processor=reference_embedder.processor, device=device
        )

    # Embed target images (or load if exists)
    print(
        f"Using {target_images_dir} as target root path. Images should be in subdirectories like {target_images_dir}/class_name/"
    )

    # Embed target images using COCO-style structure (ImageFolderWithPaths)
    target_embeddings, target_embedder = embed_dataset_batched(
        dataset_root=target_images_dir,
        save_path=target_save_path,
        dataset_type="coco",  # Target images always use ImageFolder structure
        batch_size=32,
        num_workers=4,
        device=device,
    )

    # Initialize retriever and load embeddings
    retriever = ImageRetriever(embedder=target_embedder)
    retriever.target_embeddings = target_embeddings
    retriever.reference_embeddings = reference_embeddings

    # Example: Retrieve similar images and save their embeddings
    top_k = 5  # Number of similar images to retrieve and save

    if not target_embeddings:
        print("No target embeddings loaded or computed. Skipping retrieval.")
    elif not reference_embeddings:
        print(f"No reference ({dataset_type}) embeddings loaded or computed. Skipping retrieval.")
    elif retriever is None:
        print("Retriever not initialized. Skipping retrieval.")
    else:
        print(f"\nProcessing {len(target_embeddings)} target image(s)...")
        for query_image_path in target_embeddings.keys():
            query_image_filename = os.path.basename(query_image_path)
            query_image_name_no_ext = os.path.splitext(query_image_filename)[0]
            print(f"  Retrieving similar images for: {query_image_filename}")

            try:
                similar_images = retriever.retrieve_similar_images(
                    query_image_path=query_image_path, top_k=top_k
                )

                # Define and create the output directory for this target image's results
                output_dir = os.path.join(
                    retrieved_embeddings_base_dir, query_image_name_no_ext
                )
                os.makedirs(output_dir, exist_ok=True)

                print(
                    f"    Saving top {len(similar_images)} retrieved images to {output_dir}"
                )
                for i, (retrieved_path, _) in enumerate(similar_images):
                    dest_path = os.path.join(output_dir, f"{i + 1}.jpg")
                    
                    try:
                        if dataset_type.lower() == "patternnet" and reference_dataset is not None:
                            # For PatternNet, retrieved_path is the index string
                            try:
                                index = int(retrieved_path)
                                # Get original image tensor from dataset
                                image_tensor = reference_dataset.get_original_image(index)
                                # Save tensor as image
                                torchvision.utils.save_image(image_tensor, dest_path)
                            except (ValueError, TypeError) as e:
                                print(f"      Error: Invalid PatternNet index '{retrieved_path}': {e}")
                                continue
                        else:
                            # For COCO and other datasets, retrieved_path is a file path
                            source_path = retrieved_path
                            if os.path.exists(source_path):
                                # Copy the image file
                                shutil.copy(source_path, dest_path)
                            else:
                                print(f"      Warning: Could not find retrieved image file at path: {retrieved_path}")
                                continue
                                
                    except Exception as e:
                        print(f"      Error saving image for rank {i + 1}: {e}")

            except ValueError as e:
                print(f"  Skipping {query_image_filename} due to error: {e}")
            except Exception as e:
                print(
                    f"  An unexpected error occurred while processing {query_image_filename}: {e}"
                )

    print("\nProcessing complete. Retrieved images (if any) saved.")
    return reference_embeddings, target_embeddings, retriever


def run_retrieval_with_config(config):
    """
    Run retrieval system using configuration from hydra config
    
    Args:
        config: Hydra configuration object with data.retrieval_dataset parameter
    
    Returns:
        Tuple: (reference_embeddings, target_embeddings, retriever)
    """
    dataset_type = getattr(config.data, 'retrieval_dataset', 'coco')
    target_path = getattr(config.data, 'tgt_data_path', 'resources/images/target_images')
    retrieval_device = getattr(config.data, "retrieval_device", "auto")
    return main(
        dataset_type=dataset_type,
        target_images_dir=target_path,
        device=retrieval_device,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Image retrieval system")
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["coco", "patternnet"], 
        default="coco",
        help="Dataset type to use for retrieval (default: coco)"
    )
    parser.add_argument(
        "--target-images-dir",
        type=str,
        default=None,
        help="Target image root in ImageFolder format (e.g., resources/images/target_images)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Runtime device: "auto", "cpu", "cuda", or "cuda:<index>" (default: auto)',
    )
    
    args = parser.parse_args()
    main(
        dataset_type=args.dataset,
        target_images_dir=args.target_images_dir,
        device=args.device,
    )
