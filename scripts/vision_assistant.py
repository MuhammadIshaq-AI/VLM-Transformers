"""
🚀 AI Vision Assistant — POC Demo

A web application powered by your custom Vision Transformer architecture
and pretrained BLIP model for real-world image understanding.

Features:
  📝 Image Captioning  — Describe any image in natural language
  ❓ Visual Q&A        — Ask questions about images
  🔍 Attention Maps    — Visualize WHERE your ViT looks (your custom code!)

Run:
  conda activate torch_env
  python scripts/vision_assistant.py

Opens at: http://localhost:7860
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import gradio as gr
from PIL import Image
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for web app
import matplotlib.pyplot as plt

# ─── Lazy-load models (only when first used) ────────────────────
_blip_processor = None
_blip_caption_model = None
_blip_vqa_model = None
_vit_model = None
_device = None


def get_device():
    global _device
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _device


def get_blip_caption():
    """Load BLIP captioning model (first call downloads ~1GB)."""
    global _blip_processor, _blip_caption_model
    if _blip_caption_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        print("📥 Loading BLIP captioning model...")
        model_name = "Salesforce/blip-image-captioning-base"
        _blip_processor = BlipProcessor.from_pretrained(model_name)
        _blip_caption_model = BlipForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(get_device())
        _blip_caption_model.eval()
        print("✅ BLIP captioning ready!")
    return _blip_processor, _blip_caption_model


def get_blip_vqa():
    """Load BLIP VQA model."""
    global _blip_vqa_model
    if _blip_vqa_model is None:
        from transformers import BlipForQuestionAnswering
        print("📥 Loading BLIP VQA model...")
        model_name = "Salesforce/blip-vqa-base"
        _blip_vqa_model = BlipForQuestionAnswering.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(get_device())
        _blip_vqa_model.eval()
        print("✅ BLIP VQA ready!")
    return _blip_vqa_model


def get_custom_vit():
    """Load YOUR custom Vision Transformer."""
    global _vit_model
    if _vit_model is None:
        from src.models import VisionTransformer
        print("🔧 Loading your custom ViT...")
        _vit_model = VisionTransformer(
            img_size=224, patch_size=16, embed_dim=768,
            num_heads=12, num_layers=12, dropout=0.0
        ).to(get_device())
        _vit_model.eval()
        print("✅ Custom ViT ready!")
    return _vit_model


# ═══════════════════════════════════════════════════════════════════
# Feature 1: Image Captioning
# ═══════════════════════════════════════════════════════════════════

def caption_image(image):
    """Generate a caption for the uploaded image."""
    if image is None:
        return "Please upload an image first!"

    try:
        processor, model = get_blip_caption()
        device = get_device()

        # Process image
        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

        # Generate caption
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=5,           # Beam search for better quality
                early_stopping=True,
            )

        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        return f"📝 {caption}"

    except Exception as e:
        return f"❌ Error: {str(e)}"


# ═══════════════════════════════════════════════════════════════════
# Feature 2: Visual Q&A
# ═══════════════════════════════════════════════════════════════════

def answer_question(image, question):
    """Answer a question about the image."""
    if image is None:
        return "Please upload an image first!"
    if not question or question.strip() == "":
        return "Please ask a question!"

    try:
        processor, _ = get_blip_caption()  # Reuse processor
        vqa_model = get_blip_vqa()
        device = get_device()

        inputs = processor(images=image, text=question, return_tensors="pt").to(device, torch.float16)

        with torch.no_grad():
            output_ids = vqa_model.generate(**inputs, max_new_tokens=30)

        answer = processor.decode(output_ids[0], skip_special_tokens=True)
        return f"💬 {answer}"

    except Exception as e:
        return f"❌ Error: {str(e)}"


# ═══════════════════════════════════════════════════════════════════
# Feature 3: Attention Map Visualization (YOUR Custom ViT!)
# ═══════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════
# Gradio Web App
# ═══════════════════════════════════════════════════════════════════

def build_app():
    """Build the Gradio web interface."""

    # Custom CSS for a polished look
    custom_css = """
    .gradio-container {
        max-width: 1000px !important;
        margin: auto !important;
    }
    h1 {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em !important;
    }
    """

    with gr.Blocks(css=custom_css, title="AI Vision Assistant") as app:

        gr.Markdown(
            """
            # 🧠 AI Vision Assistant
            **Powered by your custom Vision Transformer + BLIP**

            Upload any image and explore AI vision capabilities — captioning, Q&A, and attention visualization.
            All running locally on your GPU! 🚀
            """
        )

        with gr.Row():
            image_input = gr.Image(type="pil", label="📷 Upload an Image", height=350)

        with gr.Tabs():
            # ── Tab 1: Captioning ──
            with gr.TabItem("📝 Image Captioning"):
                gr.Markdown("*Generate a natural language description of the image*")
                caption_btn = gr.Button("Generate Caption", variant="primary", size="lg")
                caption_output = gr.Textbox(label="Caption", lines=2, interactive=False)
                caption_btn.click(fn=caption_image, inputs=image_input, outputs=caption_output)

            # ── Tab 2: Visual Q&A ──
            with gr.TabItem("❓ Visual Q&A"):
                gr.Markdown("*Ask any question about the image*")
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What color is the cat? / How many objects are there?",
                    lines=1
                )
                vqa_btn = gr.Button("Get Answer", variant="primary", size="lg")
                vqa_output = gr.Textbox(label="Answer", lines=2, interactive=False)
                vqa_btn.click(fn=answer_question, inputs=[image_input, question_input], outputs=vqa_output)

                gr.Examples(
                    examples=[
                        ["What is in the image?"],
                        ["What color is the main object?"],
                        ["How many animals are there?"],
                        ["What is the background?"],
                        ["Is this indoors or outdoors?"],
                    ],
                    inputs=question_input,
                    label="Example Questions"
                )

           

        gr.Markdown(
            """
            ---
            **Architecture:** Custom ViT-Base (86M params) + BLIP (pretrained VLM) |
            **GPU:** NVIDIA RTX 3050 6GB |
            **Built with:** PyTorch + HuggingFace Transformers + Gradio
            """
        )

    return app


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 AI Vision Assistant")
    print("=" * 60)
    print(f"   Device: {'CUDA — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    # ── Preload all models at startup (cached on disk after first download) ──
    print("\n📥 Preloading models (this is fast after the first download)...")
    get_blip_caption()
    get_blip_vqa()
    get_custom_vit()

    if get_device().type == 'cuda':
        mem = torch.cuda.memory_allocated() / 1024**2
        print(f"\n✅ All models loaded! GPU memory: {mem:.0f} MB")
    else:
        print("\n✅ All models loaded on CPU!")

    print(f"   Opening browser at http://localhost:7860\n")

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,       # Set True to get a public URL
        show_error=True,
    )
