"""
Patch Embedding Module for Vision Transformer (ViT).

This module takes a raw image tensor and converts it into a sequence of
patch embeddings using a 2D convolution. The Conv2d layer simultaneously:
  1. Divides the image into non-overlapping patches
  2. Projects each patch into a high-dimensional embedding space

Example:
    Input:  [Batch, 3, 224, 224]  (RGB image)
    Output: [Batch, 196, 768]     (196 patch embeddings of dim 768)
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Turns an image into a sequence of learnable patch embeddings.

    Uses a Conv2d layer where kernel_size = stride = patch_size, so the
    convolution windows tile the image without overlap — each window
    becomes one patch embedding.

    Args:
        img_size (int): Height/Width of the input image (must be square).
        patch_size (int): Height/Width of each patch.
        in_channels (int): Number of input channels (3 for RGB).
        embed_dim (int): Dimension of each patch embedding vector.
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Conv2d does patching + linear projection in one step
        # kernel_size = patch_size → each kernel covers exactly one patch
        # stride = patch_size → kernels don't overlap
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        """
        Args:
            x: Image tensor of shape [B, C, H, W]

        Returns:
            Patch embeddings of shape [B, N_patches, Embed_dim]
        """
        # [B, 3, 224, 224] → [B, 768, 14, 14]
        x = self.projection(x)

        # [B, 768, 14, 14] → [B, 768, 196]
        x = x.flatten(2)

        # [B, 768, 196] → [B, 196, 768]
        x = x.transpose(1, 2)

        return x
