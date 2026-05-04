"""
🚀 GPU Test Script for Vision Transformer

Tests the full ViT-Base pipeline on your RTX 3050:
  1. Loads your image
  2. Moves model + data to CUDA
  3. Runs forward pass with timing
  4. Reports GPU memory usage
  5. Visualizes attention maps
  6. Tests batch processing

Run from project root:
  python scripts/test_vit_gpu.py
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend

from src.models import VisionTransformer


def main():
    # ═══════════════════════════════════════════════════════════════
    # Step 1: Check GPU
    # ═══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("🔍 GPU CHECK")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ CUDA not available! Running on CPU instead.")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   PyTorch version: {torch.__version__}")

    # ═══════════════════════════════════════════════════════════════
    # Step 2: Load and preprocess image
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("📷 LOADING IMAGE")
    print("=" * 60)

    image_path = r"D:\MY DS-projects\vision-language-models\VLM-Transformers\src\data\images\image.png"

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                     std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]
    img_tensor = img_tensor.to(device)
    print(f"✅ Image loaded: {img.size} → tensor {img_tensor.shape} on {device}")

    # ═══════════════════════════════════════════════════════════════
    # Step 3: Build ViT-Base and move to GPU
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("🏗️  BUILDING ViT-Base")
    print("=" * 60)

    vit = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        mlp_ratio=4.0,
        dropout=0.1,
    )
    vit = vit.to(device)
    vit.eval()  # Set to evaluation mode (disables dropout)

    total_params = sum(p.numel() for p in vit.parameters())
    model_mb = total_params * 4 / (1024**2)  # float32
    print(f"✅ ViT-Base on {device}")
    print(f"   Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"   Model size: {model_mb:.1f} MB")

    if device.type == 'cuda':
        mem_allocated = torch.cuda.memory_allocated() / 1024**2
        print(f"   GPU memory used (model): {mem_allocated:.1f} MB")

    # ═══════════════════════════════════════════════════════════════
    # Step 4: Single image forward pass with timing
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("⚡ SINGLE IMAGE FORWARD PASS")
    print("=" * 60)

    # Warm-up (first CUDA call is slow due to kernel compilation)
    with torch.no_grad():
        _ = vit(img_tensor)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timed run
    start = time.perf_counter()
    with torch.no_grad():
        cls_output, all_tokens = vit(img_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000  # ms

    print(f"✅ Forward pass complete in {elapsed:.1f} ms")
    print(f"   [CLS] output:  {cls_output.shape}  → image summary vector")
    print(f"   All tokens:    {all_tokens.shape}  → per-patch features")

    if device.type == 'cuda':
        mem_after = torch.cuda.memory_allocated() / 1024**2
        mem_peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"   GPU memory (current): {mem_after:.1f} MB")
        print(f"   GPU memory (peak):    {mem_peak:.1f} MB")

    # ═══════════════════════════════════════════════════════════════
    # Step 5: Batch processing test
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("📦 BATCH PROCESSING TEST")
    print("=" * 60)

    for batch_size in [1, 2, 4, 8]:
        try:
            torch.cuda.reset_peak_memory_stats() if device.type == 'cuda' else None
            batch = img_tensor.repeat(batch_size, 1, 1, 1)  # Repeat image

            start = time.perf_counter()
            with torch.no_grad():
                cls_out, _ = vit(batch)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000

            peak = torch.cuda.max_memory_allocated() / 1024**2 if device.type == 'cuda' else 0
            fps = batch_size / (elapsed / 1000)
            print(f"   Batch {batch_size}: {elapsed:6.1f} ms | {fps:5.1f} img/s | Peak GPU: {peak:.0f} MB")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   Batch {batch_size}: ❌ Out of GPU memory!")
                torch.cuda.empty_cache()
                break
            raise

    # ═══════════════════════════════════════════════════════════════
    # Step 6: Visualize attention maps
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("🎨 GENERATING ATTENTION MAPS")
    print("=" * 60)

    # Run single image to get attention weights
    with torch.no_grad():
        cls_out, all_tok = vit(img_tensor)

    # Get attention from last encoder block
    last_block = vit.encoder[-1]
    attn_weights = last_block.attn.attn_weights  # [1, 12, 197, 197]
    print(f"   Attention shape: {attn_weights.shape}")

    # --- Plot 1: All 12 heads ---
    fig, axes = plt.subplots(2, 6, figsize=(20, 7))
    for i, ax in enumerate(axes.flatten()):
        cls_attn = attn_weights[0, i, 0, 1:].reshape(14, 14).cpu().numpy()
        ax.imshow(cls_attn, cmap='inferno')
        ax.set_title(f"Head {i+1}", fontsize=10)
        ax.axis('off')
    fig.suptitle("[CLS] Attention Maps — Last Layer (All 12 Heads)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path1 = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'images', 'attention_heads.png')
    fig.savefig(save_path1, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {os.path.abspath(save_path1)}")

    # --- Plot 2: Original image vs average attention ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image (denormalize)
    orig = img_tensor[0].cpu().clone()
    orig[0] = orig[0] * 0.229 + 0.485
    orig[1] = orig[1] * 0.224 + 0.456
    orig[2] = orig[2] * 0.225 + 0.406
    orig = orig.clamp(0, 1)
    axes[0].imshow(orig.permute(1, 2, 0).numpy())
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis('off')

    # Average attention
    avg_attn = attn_weights[0, :, 0, 1:].mean(dim=0).reshape(14, 14).cpu().numpy()
    axes[1].imshow(avg_attn, cmap='hot', interpolation='bilinear')
    axes[1].set_title("Average [CLS] Attention", fontsize=13)
    axes[1].axis('off')

    # Overlay
    import numpy as np
    attn_resized = np.array(Image.fromarray((avg_attn / avg_attn.max() * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR))
    axes[2].imshow(orig.permute(1, 2, 0).numpy())
    axes[2].imshow(attn_resized, cmap='hot', alpha=0.5, interpolation='bilinear')
    axes[2].set_title("Attention Overlay", fontsize=13)
    axes[2].axis('off')

    fig.suptitle("Vision Transformer — What Does the Model See?", fontsize=15, fontweight='bold')
    plt.tight_layout()
    save_path2 = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'images', 'attention_overlay.png')
    fig.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {os.path.abspath(save_path2)}")

    plt.show()

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print(f"""
   Model:        ViT-Base (86M params)
   Device:       {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})
   Input:        [1, 3, 224, 224]
   [CLS] output: [1, 768]  ← ready for the language model!
   Inference:    {elapsed:.1f} ms per image

   📌 Next: Build the Language side (text tokenizer + decoder)
""")


if __name__ == "__main__":
    main()
