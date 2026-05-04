"""
Vision Transformer (ViT) Components.

This module builds the complete Vision Transformer architecture step-by-step:

  1. VisionInput      — Patch Embedding + [CLS] Token + Positional Encoding
  2. MultiHeadSelfAttention — The core attention mechanism
  3. MLP              — Feed-Forward Network inside each encoder block
  4. TransformerEncoderBlock — One complete encoder layer (MHSA + MLP + residuals)
  5. VisionTransformer — The full ViT (stack of encoder blocks)

Architecture Flow:
  Image → PatchEmbed → [CLS] + Patches → + PosEmbed → Encoder ×N → [CLS] output

Reference: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
"""

import torch
import torch.nn as nn
from .patch_embedding import PatchEmbedding


# =============================================================================
# Step 1: VisionInput — [CLS] Token + Positional Encoding
# =============================================================================

class VisionInput(nn.Module):
    """
    Prepares the full input sequence for the Transformer Encoder.

    This layer does three things:
      1. Converts the image into patch embeddings (via PatchEmbedding)
      2. Prepends a learnable [CLS] token to the sequence
      3. Adds learnable positional embeddings to every token

    The [CLS] token is a special learnable vector that doesn't correspond
    to any image patch. After passing through the Transformer, it acts as
    a "summary" of the entire image — this is what we'll later pass to the
    language model.

    Args:
        img_size (int): Input image size (square).
        patch_size (int): Size of each patch.
        in_channels (int): Number of image channels (3 for RGB).
        embed_dim (int): Embedding dimension.
        dropout (float): Dropout rate for regularization.

    Shape:
        Input:  [B, 3, 224, 224]
        Output: [B, 197, 768]   (196 patches + 1 [CLS] token)
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, dropout=0.1):
        super().__init__()

        # Patch embedding layer (from your notebook Cell 7)
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches  # 196 for 224/16

        # ── NEW: [CLS] Token ──
        # A single learnable vector prepended to the patch sequence.
        # Shape: [1, 1, embed_dim] — the 1s let us broadcast across batches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Positional Embeddings ──
        # Now num_patches + 1 to account for the [CLS] token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Dropout for regularization ──
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Image tensor [B, C, H, W]

        Returns:
            Token sequence [B, N_patches + 1, embed_dim]
        """
        B = x.shape[0]  # Batch size

        # 1. Get patch embeddings: [B, 196, 768]
        x = self.patch_embed(x)

        # 2. Expand [CLS] token for each image in the batch
        #    [1, 1, 768] → [B, 1, 768]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # 3. Prepend [CLS] token to the patch sequence
        #    [B, 1, 768] + [B, 196, 768] → [B, 197, 768]
        x = torch.cat([cls_tokens, x], dim=1)

        # 4. Add positional embeddings (broadcasts across batch)
        x = x + self.pos_embed

        # 5. Apply dropout
        x = self.dropout(x)

        return x


# =============================================================================
# Step 2: Multi-Head Self-Attention (MHSA)
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.

    Each patch token "looks at" every other patch token to understand
    spatial relationships. The attention is split across multiple "heads"
    so the model can learn different types of relationships simultaneously.

    How it works:
      1. Project input into Q (Query), K (Key), V (Value)
      2. Split into multiple heads
      3. Compute scaled dot-product attention: softmax(QK^T / √d) × V
      4. Concatenate heads and project back

    Intuition:
      - Head 1 might learn "nearby patches" relationships
      - Head 2 might learn "same-color patches" relationships
      - Head 3 might learn "edge/boundary" relationships
      - etc.

    Args:
        embed_dim (int): Total embedding dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout on attention weights.

    Shape:
        Input:  [B, N, embed_dim]    (N = num_patches + 1)
        Output: [B, N, embed_dim]
    """

    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads  # 768 / 12 = 64

        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        # Q, K, V projections — combined into one linear layer for efficiency
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)

        # Output projection — combines all heads back together
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout on attention weights (prevents overfitting to specific patches)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # Scaling factor: 1/√(head_dim) to prevent dot products from getting too large
        self.scale = self.head_dim ** -0.5

        # Store attention weights for visualization (set during forward pass)
        self.attn_weights = None

    def forward(self, x):
        """
        Args:
            x: Token sequence [B, N, D] where N = num_patches + 1

        Returns:
            Attended token sequence [B, N, D]
        """
        B, N, D = x.shape

        # 1. Project to Q, K, V and split into heads
        #    [B, N, D] → [B, N, 3*D] → [B, N, 3, num_heads, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)

        #    → [3, B, num_heads, N, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each: [B, num_heads, N, head_dim]

        # 2. Compute attention scores: Q × K^T / √d
        #    [B, H, N, d] × [B, H, d, N] → [B, H, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 3. Softmax → attention weights (each token's "attention distribution")
        attn = attn.softmax(dim=-1)
        self.attn_weights = attn.detach()  # Save for visualization

        # 4. Apply dropout to attention weights
        attn = self.attn_dropout(attn)

        # 5. Weighted sum of values
        #    [B, H, N, N] × [B, H, N, d] → [B, H, N, d]
        x = attn @ v

        # 6. Concatenate heads: [B, H, N, d] → [B, N, H*d] = [B, N, D]
        x = x.transpose(1, 2).reshape(B, N, D)

        # 7. Final output projection
        x = self.out_proj(x)
        x = self.out_dropout(x)

        return x


# =============================================================================
# Step 3: MLP (Feed-Forward Network)
# =============================================================================

class MLP(nn.Module):
    """
    Feed-Forward Network inside each Transformer block.

    Processes each token independently (no cross-token interaction).
    Uses a "bottleneck" expansion: embed_dim → 4×embed_dim → embed_dim.

    The GELU activation is smoother than ReLU and works better in
    Transformers — it allows small negative values through, which
    helps with gradient flow.

    Args:
        embed_dim (int): Input/output dimension.
        mlp_ratio (float): Expansion ratio (default 4.0 = standard ViT).
        dropout (float): Dropout rate.
    """

    def __init__(self, embed_dim=768, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)  # 768 × 4 = 3072

        self.fc1 = nn.Linear(embed_dim, hidden_dim)   # Expand
        self.act = nn.GELU()                            # Activation
        self.fc2 = nn.Linear(hidden_dim, embed_dim)    # Compress back
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [B, N, embed_dim]
        Returns:
            [B, N, embed_dim]
        """
        x = self.fc1(x)       # [B, N, 768] → [B, N, 3072]
        x = self.act(x)       # GELU activation
        x = self.dropout(x)
        x = self.fc2(x)       # [B, N, 3072] → [B, N, 768]
        x = self.dropout(x)
        return x


# =============================================================================
# Step 4: Transformer Encoder Block
# =============================================================================

class TransformerEncoderBlock(nn.Module):
    """
    One complete Transformer Encoder block.

    Architecture (Pre-Norm style, used in modern ViTs):
        x = x + MHSA(LayerNorm(x))    ← Attention + Residual
        x = x + MLP(LayerNorm(x))     ← Feed-Forward + Residual

    Key design choices:
      - Pre-Norm: LayerNorm BEFORE attention/MLP (more stable training)
      - Residual connections: The '+' operations (prevents vanishing gradients)
      - Each sub-layer has its own LayerNorm

    Args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): MLP expansion ratio.
        dropout (float): Dropout rate.
    """

    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        # Layer Normalization (normalizes across the embedding dimension)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Multi-Head Self-Attention
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        # MLP (Feed-Forward Network)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        """
        Args:
            x: [B, N, embed_dim]
        Returns:
            [B, N, embed_dim]
        """
        # Attention block with Pre-Norm and residual connection
        x = x + self.attn(self.norm1(x))

        # MLP block with Pre-Norm and residual connection
        x = x + self.mlp(self.norm2(x))

        return x


# =============================================================================
# Step 5: Full Vision Transformer (ViT)
# =============================================================================

class VisionTransformer(nn.Module):
    """
    Complete Vision Transformer (ViT-Base configuration).

    Full pipeline:
        Image → Patches → Embeddings → [CLS] + Pos → Encoder ×12 → [CLS] output

    The output is the [CLS] token's embedding after all encoder layers,
    which serves as a rich, contextual representation of the entire image.
    This vector is what will later be projected into the language model's
    embedding space for the VLM.

    Args:
        img_size (int): Input image size.
        patch_size (int): Patch size.
        in_channels (int): Number of image channels.
        embed_dim (int): Embedding dimension.
        num_heads (int): Attention heads per layer.
        num_layers (int): Number of Transformer encoder blocks.
        mlp_ratio (float): MLP expansion ratio.
        dropout (float): Dropout rate.

    ViT Configurations:
        ViT-Tiny:  embed_dim=192,  num_heads=3,  num_layers=12
        ViT-Small: embed_dim=384,  num_heads=6,  num_layers=12
        ViT-Base:  embed_dim=768,  num_heads=12, num_layers=12
        ViT-Large: embed_dim=1024, num_heads=16, num_layers=24
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, num_heads=12, num_layers=12,
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        # Input stage: patches + [CLS] + positional embeddings
        self.input_layer = VisionInput(
            img_size, patch_size, in_channels, embed_dim, dropout
        )

        # Stack of Transformer encoder blocks
        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # Final LayerNorm
        self.norm = nn.LayerNorm(embed_dim)

        # Store config for reference
        self.embed_dim = embed_dim
        self.num_patches = self.input_layer.patch_embed.n_patches

    def forward(self, x):
        """
        Args:
            x: Image tensor [B, 3, H, W]

        Returns:
            cls_output: [CLS] token embedding [B, embed_dim] — image summary
            all_tokens: All token embeddings [B, N+1, embed_dim] — for cross-attention
        """
        # 1. Prepare input sequence: [B, 197, 768]
        x = self.input_layer(x)

        # 2. Pass through all encoder blocks
        x = self.encoder(x)

        # 3. Final normalization
        x = self.norm(x)

        # 4. Extract the [CLS] token (index 0) as the image representation
        cls_output = x[:, 0]       # [B, 768]
        all_tokens = x             # [B, 197, 768] — useful for cross-attention later

        return cls_output, all_tokens
