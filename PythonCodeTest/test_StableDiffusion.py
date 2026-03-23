import torch
from diffusers import StableDiffusionPipeline

# ══════════════════════════════════════════════════════════════════════════════
# WHAT IS STABLE DIFFUSION?
# ──────────────────────────────────────────────────────────────────────────────
# Stable Diffusion generates images from TEXT descriptions.
# Example: "A serene sunset over a calm lake" → actual image of that scene.
#
# It combines THREE things you already know + one new concept:
#
#   1. VAE (you know this)
#      Compresses images into latent space and reconstructs them back.
#      Same concept as before, just much larger.
#
#   2. DIFFUSION (new concept)
#      Core idea:
#        - Take a real image
#        - Gradually add random noise over many steps until it's pure noise
#        - Train a network to REVERSE this — learn to denoise step by step
#        - At generation: start from pure noise, denoise step by step → image
#      Think of it like: crumpling a photo into a ball (adding noise),
#      then learning to uncrumple it back (denoising).
#
#   3. UNet (does the denoising)
#      Neural network that denoises at each step.
#      Shape: compresses down then expands back up with skip connections.
#
#   4. CLIP (text understanding)
#      Converts your text prompt into numbers the network understands.
#      "a cat on beach" → vector of numbers → guides denoising toward description.
#
# Full pipeline:
#   Text prompt → CLIP → text embeddings
#                                 ↓
#   Pure noise → UNet (guided by text) → denoise step by step → VAE decode → image
#
# Key differences from GAN and VAE:
#   VAE — no text, compresses/reconstructs, generates from random latent point
#   GAN — no text, generator vs discriminator competition
#   SD  — text guided, denoises step by step, uses all three concepts combined
#
# Libraries:
#   torch     — PyTorch, Facebook's ML framework. SD runs on PyTorch not TensorFlow.
#               Same concept as TensorFlow — builds and runs neural networks.
#   diffusers — Hugging Face's library that packages the ENTIRE Stable Diffusion
#               pipeline (CLIP + UNet + VAE + Scheduler) into one easy object.
#               Without it you'd have to wire all four components manually.
# ══════════════════════════════════════════════════════════════════════════════


# ── Model Selection ────────────────────────────────────────────────────────────
# Model ID points to a pretrained model hosted on huggingface.co.
# from_pretrained() downloads it first time, caches locally after.
#
# Why this tiny model?
#   Full SD models (runwayml/stable-diffusion-v1-5) = 4-5 GB download,
#   needs 4-8 GB RAM, takes 30-60 seconds per image on CPU.
#   This tiny test model = few MBs, runs in seconds, good for learning/testing.
#   Quality won't be great but the pipeline concept is identical.

model_id = "nota-ai/bk-sdm-tiny"
# Alternative full quality models (much heavier, not for laptop):
#   "runwayml/stable-diffusion-v1-5"          — standard SD, 4GB
#   "dreamlike-art/dreamlike-photoreal-2.0"   — photorealistic, 4GB


# ── Load the Pipeline ──────────────────────────────────────────────────────────
pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32,   # float16 requires GPU, float32 works on CPU
    safety_checker=None
)
# This single line loads all four components:
#   CLIP      — converts text prompt to numbers
#   UNet      — does the actual denoising at each step
#   VAE       — decodes final latent into a viewable image
#   Scheduler — controls how many denoising steps and how large each step is

pipeline = pipeline.to("cpu")
# Explicitly run on CPU (no GPU needed).
# On GPU you'd use .to("cuda") for much faster generation.


# ── Prompts ────────────────────────────────────────────────────────────────────
# Keeping to 1 image only — generating 3 would triple the time on laptop.

positive_prompts = [
    "A serene sunset over a calm lake"
]
# Positive prompt — what you WANT in the image.
# CLIP converts this text to numbers that guide the denoising toward this scene.
# This is unique to Stable Diffusion — VAE and GAN had no concept of text guidance.

negative_prompts = [
    "blurry, distorted, low quality"
]
# Negative prompt — what you DON'T want.
# Tells the model to steer AWAY from these qualities during denoising.
# Works by pulling the generation in the opposite direction of these descriptors.


# ── Generate Images ────────────────────────────────────────────────────────────
generated_images = []

for i, (prompt, neg_prompt) in enumerate(zip(positive_prompts, negative_prompts)):
    # Pairs each positive prompt with its negative prompt and loops through all.

    print(f"Generating image {i+1}/{len(positive_prompts)}: '{prompt}'")

    image = pipeline(
        prompt          = prompt,
        negative_prompt = neg_prompt,
        num_inference_steps = 20,
        # num_inference_steps = how many denoising steps to run.
        # Default is 50. Fewer steps = faster but lower quality.
        # 20 is a good balance for laptop speed vs visible output.
        # Think of it as: 50 steps of uncrumpling vs 20 steps — less refined.

        height = 256,
        width  = 256,
        # Default output size is 512×512. Reduced to 256×256.
        # Smaller image = less computation = much faster on CPU.
        # 4x fewer pixels = roughly 4x faster generation.

    ).images[0]
    # .images[0] grabs the first (only) generated image from the result.
    # Full pipeline runs here:
    #   text prompt → CLIP → UNet denoises 20 steps → VAE decodes → image

    image.save(f'image_{i}.jpg')
    # Saves directly to disk as JPEG.
    # Unlike GAN/VAE which used matplotlib, SD outputs PIL Image objects
    # that can be saved directly.

    generated_images.append(image)
    # Also keep in memory for further use.

print(f"\nDone. Generated {len(generated_images)} image(s).")
print("Saved as image_0.jpg in your current directory.")

# ── Display the Generated Image ────────────────────────────────────────────────
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
for i, image in enumerate(generated_images):
    plt.subplot(1, len(generated_images), i + 1)
    plt.imshow(image)
    plt.title(f"Image {i+1}", fontsize=10)
    plt.axis('off')

plt.suptitle('Stable Diffusion Generated Image', fontsize=12)
plt.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# RECAP — What just happened end to end:
# ──────────────────────────────────────────────────────────────────────────────
#
#   1. Text prompt fed into CLIP → converted to numerical embeddings
#   2. Pure random noise created (like GAN's random noise input)
#   3. UNet runs 20 denoising steps, guided by text embeddings each step
#      → Each step the noise becomes slightly more image-like
#      → Text embeddings steer each step toward the described scene
#   4. VAE decodes the final denoised latent → viewable image
#   5. Saved to disk and displayed
#
# Speed optimizations made for laptop:
#   - Tiny model         → MBs instead of GBs download
#   - float32            → compatible with CPU (float16 needs GPU)
#   - num_inference_steps=20 → half the default 50 steps
#   - 256×256 resolution → quarter the pixels of default 512×512
#   - 1 image only       → no repeat overhead
# ══════════════════════════════════════════════════════════════════════════════