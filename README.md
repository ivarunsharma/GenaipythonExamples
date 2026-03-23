# 🤖 GenAI Python Examples

> **A hands-on, deeply documented journey through the four pillars of modern Generative AI** — from compressing pixels to generating images from thin air, from adversarial competition to text-guided creation, and from raw text understanding to intelligent response generation.

Every file in this repository is written to be **read like a textbook** — not just executed. Inline comments explain not just *what* the code does, but *why* it does it, with analogies, visual diagrams, and concept explanations built directly into the source.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [The Learning Journey](#-the-learning-journey)
- [Repository Structure](#-repository-structure)
- [Detailed File Breakdown](#-detailed-file-breakdown)
  - [VAE — Variational Autoencoder](#1-test_vaepy--variational-autoencoder)
  - [GAN — Generative Adversarial Network](#2-test_ganpy--generative-adversarial-network)
  - [Stable Diffusion](#3-test_stablediffusionpy--stable-diffusion)
  - [Transformers](#4-test_transformerspy--transformer-nlp-applications)
- [Architecture Comparisons](#-architecture-comparisons)
- [Key Concepts Glossary](#-key-concepts-glossary)
- [Requirements & Installation](#-requirements--installation)
- [Running the Examples](#-running-the-examples)
- [Laptop-Friendly Optimizations](#-laptop-friendly-optimizations)
- [Model Downloads Reference](#-model-downloads-reference)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🌐 Overview

This repository is a **practical, educational GenAI reference** built for developers who want to understand how modern generative models work from the inside — not just call an API.

Each example:
- Is **self-contained** — runs independently with no cross-file dependencies
- Is **CPU-friendly** — optimized to run on a standard laptop without a GPU
- Is **heavily documented** — every design decision is explained inline
- Progresses **conceptually** — each file builds on ideas from the previous one

**Who this is for:**
- Developers learning Generative AI for the first time
- Engineers looking for clean, readable reference implementations
- Anyone who wants to understand what actually happens inside these models

---

## 🗺️ The Learning Journey

The four examples are intentionally ordered. Each one introduces new concepts that either build on, or contrast with, the previous:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. VAE ──────────► 2. GAN ──────────► 3. Stable Diffusion             │
│     │                   │                   │                           │
│     │                   │                   │                           │
│  Compress &          Compete &           Text-guided                    │
│  Reconstruct         Generate            Denoising                      │
│  (blurry but         (sharper via        (combines                      │
│   principled)         adversarial         VAE + new                     │
│                       training)           concepts)                     │
│                                                                         │
│  4. Transformers ──────────────────────────────────────────────────►   │
│     │                                                                   │
│  Understand &                                                           │
│  Generate Text                                                          │
│  (attention-based,                                                      │
│   pretrained models)                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

| Step | File | Domain | Core Concept | Output |
|------|------|--------|-------------|--------|
| 1 | `test_VAE.py` | Images | Compression + Reconstruction | Generated digit images |
| 2 | `test_GAN.py` | Images | Adversarial Competition | Sharper digit images |
| 3 | `test_StableDiffusion.py` | Images | Text-guided Denoising | Photo from text prompt |
| 4 | `test_Transformers.py` | Text | Attention Mechanism | Sentiment + Generated text |

---

## 📁 Repository Structure

```
GenaipythonExamples/
│
├── 📄 test_VAE.py                  # Variational Autoencoder
├── 📄 test_GAN.py                  # Generative Adversarial Network
├── 📄 test_StableDiffusion.py      # Stable Diffusion (text-to-image)
├── 📄 test_Transformers.py         # Transformer NLP (classify + generate)
│
├── 📄 requirements.txt             # All Python dependencies
├── 📄 LICENSE                      # MIT License
└── 📄 README.md                    # This file
```

---

## 📚 Detailed File Breakdown

---

### 1. `test_VAE.py` — Variational Autoencoder

#### What is a VAE?

A **Variational Autoencoder** is a neural network that learns to compress data into a small representation and reconstruct it back. The "Variational" part is the key differentiator: instead of compressing to **exact numbers**, it compresses to a **range** (defined by a mean and variance).

This seemingly small difference has a profound consequence: you can **generate brand new images** by sampling random points from that range — the network has learned the *concept* of what digits look like, not just memorized them.

#### The Counterfeiter Analogy

Think of it like describing a person's face using just 2 numbers:
- A basic autoencoder says: *"This face = (3.2, 7.1)"* — exact, fixed.
- A VAE says: *"This face is somewhere around (3.2 ± 0.5, 7.1 ± 0.3)"* — a range.

With a range, you can sample a random point within it and generate a **slightly different but still realistic face** — one the network has never seen.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ENCODER                                  │
│                                                                 │
│  Input Image        Flatten      Dense(128)     Outputs         │
│  (28×28)     ──►   (784)   ──►  (patterns) ──► z_mean (2)      │
│                                             └─► z_log_var (2)   │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   SAMPLING LAYER   │
                    │                   │
                    │  z = mean +        │
                    │  exp(0.5*logvar)   │
                    │  * ε (random)      │
                    └─────────┬──────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                        DECODER                                  │
│                                                                 │
│  Latent (2)  ──► Dense(128) ──► Dense(784) ──► Reshape(28×28)  │
│                                sigmoid            Output Image  │
└─────────────────────────────────────────────────────────────────┘
```

#### The Two Losses

The VAE uses two loss functions added together:

| Loss | Formula | Purpose |
|------|---------|---------|
| **Reconstruction Loss** | Binary crossentropy × 784 | How different is output from input? |
| **KL Loss** | −0.5 × (1 + log_var − mean² − exp(log_var)) | Is the latent space well organized? |

KL Loss is the guardian of generation quality — without it, the encoder could output wild, scattered ranges that reconstruct well but produce garbage for new random points.

#### Key Parameters (Optimized for Laptop)

| Parameter | Value | Why |
|-----------|-------|-----|
| `latent_dim` | 8 | Balance between detail and speed (2=blurry, 128=sharp but slow) |
| `Dense` size | 128 | Middle ground between 784 input and 2 latent |
| `epochs` | 10 | Sufficient to see meaningful output |
| Training samples | 5,000 | Fast training without sacrificing concept demonstration |

#### What the Output Signifies

A 4×4 grid of 16 generated digit images that **never existed in the training data**. The VAE decoded random latent points into coherent digit-like images — proof it learned the structural concept of handwritten digits, not just pixel patterns.

---

### 2. `test_GAN.py` — Generative Adversarial Network

#### What is a GAN?

A **Generative Adversarial Network** trains two networks in direct competition:

- **Generator** — creates fake images from random noise. Goal: fool the discriminator.
- **Discriminator** — examines images and decides real or fake. Goal: catch the generator.

They train simultaneously. The generator gets better at faking; the discriminator gets better at detecting. This arms race drives the generator to produce increasingly realistic images.

**The perfect analogy: Counterfeiter vs Detective.** Both improve by competing with each other.

#### Why GAN Produces Sharper Images Than VAE

| Model | Optimization Target | Result |
|-------|-------------------|--------|
| VAE | Average of all possible reconstructions | Blurry (mathematically averaging causes blur) |
| GAN | Fool a trained detector | Sharp (must look realistic enough to deceive) |

#### Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         GENERATOR                                  │
│                                                                    │
│  100 random     Dense    Dense    Dense    Dense(784)  Reshape     │
│  numbers   ──► (256) ──► (512) ──► (1024) ──► tanh  ──► (28,28,1) │
│  (noise)        ↕         ↕         ↕                              │
│              LeakyReLU LeakyReLU LeakyReLU                         │
│              BatchNorm BatchNorm BatchNorm                          │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                       DISCRIMINATOR                                │
│                                                                    │
│  Image      Flatten   Dense    Dense    Dense(1)                   │
│  (28,28,1) ──► (784) ──► (512) ──► (256) ──► sigmoid              │
│                           ↕         ↕         │                    │
│                        LeakyReLU LeakyReLU    ▼                    │
│                        Dropout   Dropout    0=Fake                 │
│                                             1=Real                 │
└────────────────────────────────────────────────────────────────────┘
```

#### The Training Loop

Each batch executes two distinct steps:

```
┌──────────────────────────────────────────────────────────────┐
│  STEP 1 — Train Discriminator (Generator frozen)             │
│                                                              │
│  Real images + noisy labels (~0.8-1.0) → train              │
│  Fake images + noisy labels (~0.0-0.2) → train              │
│  Discriminator learns to distinguish real from fake          │
└──────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│  STEP 2 — Train Generator (Discriminator frozen)             │
│                                                              │
│  Noise → Generator → Fake image → Discriminator             │
│  Tell GAN these should score ~1.0 (real)                     │
│  Only generator weights update                               │
│  Generator learns to produce images that fool discriminator  │
└──────────────────────────────────────────────────────────────┘
```

#### Stability Techniques Used

| Technique | What It Does | Why Needed |
|-----------|-------------|-----------|
| **Label Smoothing** | Real=0.8-1.0, Fake=0.0-0.2 instead of exact 0/1 | Prevents discriminator overconfidence |
| **Instance Noise** | Adds small noise to all images | Forces discriminator to learn structure, not pixels |
| **Label Flipping** | 5% of labels randomly flipped | Prevents generator exploiting patterns |
| **LeakyReLU** | Passes 20% of negative values | Prevents dead neurons in generator |
| **Dropout (30%)** | Randomly disables neurons | Keeps discriminator from overpowering generator |

#### What to Watch in Training Output

```
D acc  → ideally 0.5-0.8. At 1.0 = discriminator too powerful, generator stuck
D loss → should stay moderate, not collapse to 0
G loss → should gradually decrease as generator improves
```

---

### 3. `test_StableDiffusion.py` — Stable Diffusion

#### What is Stable Diffusion?

Stable Diffusion generates images from **text descriptions**. It is the most architecturally complex model in this repository, combining four separate components into one seamless pipeline.

#### The Four Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STABLE DIFFUSION PIPELINE                        │
│                                                                     │
│  Text Prompt                                                        │
│      │                                                              │
│      ▼                                                              │
│  ┌────────┐    text embeddings                                      │
│  │  CLIP  │ ──────────────────────────────────────►                │
│  └────────┘                                        │               │
│  Converts text to                                  │               │
│  numerical vectors                                 ▼               │
│                                              ┌──────────┐          │
│  Pure Random Noise ─────────────────────────►│  UNet    │          │
│                                              │          │          │
│                                              │ Denoises │          │
│                                              │ step by  │          │
│                                              │ step     │          │
│                                              │ (guided  │          │
│                                              │ by text) │          │
│                                              └────┬─────┘          │
│                                                   │                │
│                                                   ▼                │
│                                              ┌──────────┐          │
│                                              │   VAE    │          │
│                                              │ Decoder  │          │
│                                              └────┬─────┘          │
│                                                   │                │
│                                                   ▼                │
│                                           Generated Image          │
└─────────────────────────────────────────────────────────────────────┘
```

| Component | Role | Analogy |
|-----------|------|---------|
| **CLIP** | Converts text → numbers | Translator |
| **Scheduler** | Controls noise removal steps | Metronome |
| **UNet** | Does the actual denoising | Artist |
| **VAE** | Decodes final result to viewable image | Printer |

#### The Diffusion Concept

The core innovation of diffusion models:

1. **Forward process (training):** Take a real image → gradually add random noise over ~1000 steps → pure noise
2. **Reverse process (generation):** Start from pure noise → learn to denoise step by step → image emerges

Think of it as: **crumpling a photo into a ball** (adding noise), then **learning to uncrumple it back** (denoising). The text prompt guides *what* the image should look like as it uncrumples.

#### Positive vs Negative Prompts

```python
positive_prompt = "A serene sunset over a calm lake"
# → Steers generation TOWARD these qualities

negative_prompt = "blurry, distorted, low quality"
# → Steers generation AWAY from these qualities
```

This dual-prompt system is unique to Stable Diffusion — VAE and GAN had no concept of text guidance.

#### Key Parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `num_inference_steps` | 20 | Fewer steps = faster but less refined (default: 50) |
| `height/width` | 256×256 | Smaller = 4× faster (default: 512×512) |
| `safety_checker` | None | Disabled due to tiny model size mismatch |

---

### 4. `test_Transformers.py` — Transformer NLP Applications

#### What are Transformers?

The architecture behind virtually every modern language model — GPT, BERT, Claude, and others. The core innovation is the **attention mechanism**: instead of processing words one by one, transformers look at the **entire sequence at once** and learn which words are most relevant to each other.

**The attention insight:**
> In the sentence *"The cat sat on the mat because **it** was tired"* — what does "it" refer to? A transformer learns to **pay attention** to "cat" when processing "it", connecting related words regardless of their distance in the sentence.

#### Part 1 — Sentiment Analysis (Text Classification)

Uses **DistilBERT** — a smaller, faster version of BERT (Bidirectional Encoder Representations from Transformers).

```
Raw Review Text
      │
      ▼
  Tokenizer
  (text → numbers)
      │
      ▼
  DistilBERT Layers
  (attention mechanism)
      │
      ▼
  Classification Head
      │
      ▼
  POSITIVE (0.9997) or NEGATIVE (0.9998)
```

**Why high confidence on these reviews?**
Both test reviews are unambiguously clear. A mixed review like *"battery great but camera terrible"* would produce a lower confidence score — the model detects conflicting signals.

#### Part 2 — Text Generation (GPT-2)

Uses **GPT-2** — a generative transformer that predicts the next token repeatedly.

```
[Complaint text] + "Customer service response:" + [starter]
      │
      ▼
  GPT-2 predicts next token
      │
      ▼
  Appends token, predicts next
      │
      ▼
  Repeats until max_length=150
      │
      ▼
  Completed customer service response
```

**The technique used: Prompt Engineering** — structuring the input with the complaint + response starter to guide GPT-2 toward generating a relevant continuation. This is the same fundamental principle behind modern ChatGPT prompting.

#### Discriminative vs Generative Transformers

| Type | Model | Input | Output | Use Case |
|------|-------|-------|--------|----------|
| **Discriminative** | DistilBERT | Full text | Label + score | Classification, Q&A |
| **Generative** | GPT-2 | Partial text | Text continuation | Writing, completion |

#### Hugging Face Pipeline Tasks Reference

```python
pipeline("text-classification")      # Sentiment, topic classification
pipeline("text-generation")          # Continue/complete text
pipeline("summarization")            # Condense long documents
pipeline("translation_en_to_fr")     # Language translation
pipeline("question-answering")       # Extract answers from passages
pipeline("ner")                      # Find names, places, dates
pipeline("fill-mask")                # Predict masked words
pipeline("zero-shot-classification") # Classify without training
```

---

## ⚖️ Architecture Comparisons

### Image Generation: VAE vs GAN vs Stable Diffusion

| Aspect | VAE | GAN | Stable Diffusion |
|--------|-----|-----|-----------------|
| **Training** | Single network with two losses | Two competing networks | Pretrained, no training needed |
| **Input** | Random latent point | Random noise vector | Text prompt |
| **Output Quality** | Blurry (averages) | Sharp (adversarial) | Very sharp (denoising) |
| **Speed** | Fast | Medium | Slow (many steps) |
| **Text guided** | ❌ | ❌ | ✅ |
| **Interpretable latent** | ✅ (mean + variance) | ❌ | Partial |
| **Stability** | Very stable | Notoriously tricky | Stable (pretrained) |

### All Four Models: Common Threads

All four models share these fundamental building blocks:

| Concept | VAE | GAN | Stable Diffusion | Transformers |
|---------|-----|-----|-----------------|-------------|
| **Normalization** | ÷255 → [0,1] | −127.5÷127.5 → [−1,1] | Built-in | Tokenization |
| **Loss function** | Recon + KL | Binary crossentropy | Diffusion loss | Cross-entropy |
| **Latent space** | ✅ Explicit | ✅ Implicit (noise) | ✅ (VAE inside) | ✅ (embeddings) |
| **Decoder/Generator** | ✅ | ✅ | ✅ (UNet+VAE) | ✅ (GPT-2) |
| **Backpropagation** | ✅ | ✅ | (pretrained) | (pretrained) |

---

## 📖 Key Concepts Glossary

| Term | Definition |
|------|-----------|
| **Latent Space** | A compressed numerical representation of data. High-dimensional images become low-dimensional vectors. |
| **Encoder** | Neural network component that compresses input into latent space. |
| **Decoder / Generator** | Neural network component that reconstructs/creates output from latent space. |
| **Normalization** | Scaling input values to a small, consistent range (0–1 or −1 to 1) for stable training. |
| **Activation Function** | Non-linear transformation applied after each layer. `relu`, `sigmoid`, `tanh`, `LeakyReLU`. |
| **Loss Function** | Mathematical measure of how wrong the model is. Training minimizes this. |
| **Backpropagation** | The algorithm that adjusts network weights based on the gradient of the loss. |
| **Epoch** | One complete pass through the entire training dataset. |
| **Batch Size** | Number of samples processed per gradient update step. |
| **Learning Rate** | How large each weight adjustment step is. Too high = unstable. Too low = slow. |
| **Weights** | The learned numerical parameters inside a neural network. Stored in memory, saveable to disk. |
| **Dense Layer** | Fully connected layer — every neuron connects to every neuron in the next layer. |
| **Dropout** | Randomly disabling neurons during training to prevent overfitting. |
| **BatchNormalization** | Rescaling layer outputs to a stable range after each batch. |
| **LeakyReLU** | Activation that passes small negatives through (unlike relu which kills them). |
| **KL Divergence** | Measures how different one probability distribution is from another. Used in VAE loss. |
| **Attention** | Mechanism that learns which parts of a sequence are most relevant to each other. |
| **Tokenizer** | Splits raw text into chunks (tokens) and maps each to a number. |
| **Token** | The basic unit of text a transformer processes (can be a word, subword, or character). |
| **Embedding** | Dense numerical representation of a token or text in high-dimensional space. |
| **Prompt Engineering** | Carefully structuring input text to guide model output toward desired results. |
| **Inference Steps** | Number of denoising iterations in Stable Diffusion. More steps = higher quality but slower. |
| **Pretrained Model** | A model already trained on massive data, ready to use without further training. |
| **Fine-tuning** | Further training a pretrained model on a specific task or dataset. |
| **Hugging Face** | Platform hosting thousands of pretrained models, accessible via the `transformers` library. |
| **CLIP** | Model that connects text and images by mapping both to a shared numerical space. |
| **UNet** | Neural network architecture shaped like a U — compresses then expands with skip connections. |
| **Diffusion** | Process of gradually adding/removing noise from data for generation. |

---

## ⚙️ Requirements & Installation

### Python Version

```
Python 3.8 or higher
```

### Quick Install

```bash
pip install -r requirements.txt
```

### Manual Install

```bash
# Core ML frameworks
pip install tensorflow>=2.10
pip install torch>=2.0

# Image generation
pip install diffusers>=0.20
pip install accelerate>=0.20

# NLP
pip install transformers>=4.30

# Utilities
pip install numpy>=1.21
pip install matplotlib>=3.5
pip install pandas>=1.3
```

### Full Dependency Table

| Library | Min Version | Used In | Purpose |
|---------|------------|---------|---------|
| `tensorflow` | 2.10 | VAE, GAN | Model building and training |
| `numpy` | 1.21 | VAE, GAN | Array operations and data manipulation |
| `matplotlib` | 3.5 | VAE, GAN, SD | Visualizing generated images |
| `torch` | 2.0 | SD, Transformers | PyTorch backend for HuggingFace models |
| `diffusers` | 0.20 | Stable Diffusion | Stable Diffusion pipeline |
| `transformers` | 4.30 | SD, Transformers | Pretrained transformer models |
| `accelerate` | 0.20 | Stable Diffusion | Optimized model loading |
| `pandas` | 1.3 | Transformers | Results display as tables |

---

## 🚀 Running the Examples

### VAE — Variational Autoencoder

```bash
python test_VAE.py
```

**What happens:**
1. Downloads MNIST dataset (~11MB, cached after)
2. Trains for 10 epochs on 5,000 images (~1 minute on CPU)
3. Displays 4×4 grid of 16 generated digit images

**Expected output:** Slightly blurry but recognizable digit shapes (blurriness is mathematically expected in VAEs)

---

### GAN — Generative Adversarial Network

```bash
python test_GAN.py
```

**What happens:**
1. Loads MNIST (already cached)
2. Trains for 5 epochs on 10,000 images (~2 minutes on CPU)
3. Prints loss/accuracy every 25 batches
4. Displays 10×10 grid of 100 generated digit images

**Expected output:** Sharper digit-like shapes than VAE. Some may still be noisy at only 5 epochs.

---

### Stable Diffusion

```bash
python test_StableDiffusion.py
```

**What happens:**
1. Downloads tiny-stable-diffusion model (~10MB, first run only)
2. Runs 20 denoising steps on a text prompt
3. Saves `image_0.jpg` to current directory
4. Displays generated image via matplotlib

**Expected output:** Low quality image (tiny model is for pipeline testing). Swap model ID for a real model for production quality.

---

### Transformers

```bash
python test_Transformers.py
```

**What happens:**
1. Downloads DistilBERT (~250MB, first run only)
2. Classifies both reviews with confidence scores
3. Downloads GPT-2 (~500MB, first run only)
4. Generates customer service response continuation

**Expected output:**
```
Negative Review Result:
      label     score
0  NEGATIVE  0.999812

Positive Review Result:
      label     score
0  POSITIVE  0.999711

Generated Customer Service Response:
[full complaint + generated continuation...]
```

---

## 💻 Laptop-Friendly Optimizations

All examples are designed to run on a **standard CPU laptop** without any GPU. Here is what was reduced from typical production settings and why:

| File | Default (Production) | This Repo (Laptop) | Speedup |
|------|---------------------|-------------------|---------|
| `test_VAE.py` | 60,000 samples, Dense(512), 100 epochs | 5,000 samples, Dense(128), 10 epochs | ~50× faster |
| `test_GAN.py` | 60,000 samples, 50 epochs | 10,000 samples, 5 epochs | ~30× faster |
| `test_StableDiffusion.py` | Full SD v1.5 (4GB), 512×512, 50 steps | Tiny model (10MB), 256×256, 20 steps | ~100× faster |
| `test_Transformers.py` | Custom fine-tuned models | DistilBERT + GPT-2 (smallest capable) | Optimal |

### Saving Trained Models

By default, trained weights exist **only in memory** and are lost when the script ends. To save and reuse:

```python
# Save after training
generator.save('generator.h5')          # GAN generator
vae.encoder.save('vae_encoder.h5')      # VAE encoder
vae.decoder.save('vae_decoder.h5')      # VAE decoder

# Load next time (skip retraining)
from tensorflow.keras.models import load_model
generator = load_model('generator.h5')
```

**Approximate saved file sizes for this repo's models:**

| Model | File Size |
|-------|----------|
| VAE Encoder | ~1-2 MB |
| VAE Decoder | ~1-2 MB |
| GAN Generator | ~3-5 MB |
| GAN Discriminator | ~3-5 MB |

---

## 📦 Model Downloads Reference

All Hugging Face models are downloaded on first run and cached at `~/.cache/huggingface/`.

| File | Model ID | Size | Download Time (est.) |
|------|----------|------|---------------------|
| `test_StableDiffusion.py` | `hf-internal-testing/tiny-stable-diffusion-pipe` | ~10 MB | Seconds |
| `test_Transformers.py` | `distilbert-base-uncased-finetuned-sst-2-english` | ~250 MB | ~1 min |
| `test_Transformers.py` | `gpt2` | ~500 MB | ~2 min |

**To use full quality Stable Diffusion** (requires ~4GB disk and ~8GB RAM):
```python
# Replace model_id in test_StableDiffusion.py with:
model_id = "runwayml/stable-diffusion-v1-5"
```

---

## 🔧 Troubleshooting

### VAE / GAN: `ModuleNotFoundError: No module named 'tensorflow'`
```bash
pip install tensorflow
```

### Stable Diffusion: `ValueError: Input image size doesn't match model`
Add `safety_checker=None` to `from_pretrained()`:
```python
pipeline = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None)
```

### Stable Diffusion: Out of Memory
Reduce image size further:
```python
image = pipeline(prompt=prompt, height=128, width=128, num_inference_steps=10).images[0]
```

### Transformers: Slow first run
Normal — models are downloading. Check `~/.cache/huggingface/` to confirm download progress.

### GAN: Generated images completely black/white
The model needs more epochs. Increase:
```python
train(epochs=15, batch_size=64)
```

### General: Want to speed things up further
For all TensorFlow files, disable GPU warnings:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Varun Sharma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙌 Acknowledgements

| Resource | Contribution |
|----------|-------------|
| [TensorFlow / Keras](https://www.tensorflow.org/) | VAE and GAN model building and training |
| [PyTorch](https://pytorch.org/) | Backend for Stable Diffusion and Transformers |
| [Hugging Face](https://huggingface.co/) | `diffusers` and `transformers` libraries + model hosting |
| [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) | Handwritten digit benchmark dataset by Yann LeCun |
| [OpenAI GPT-2](https://openai.com/research/language-unsupervised) | Generative language model |
| [Google DistilBERT](https://huggingface.co/distilbert-base-uncased) | Efficient text classification model |
| [Stability AI](https://stability.ai/) | Stable Diffusion architecture and weights |

---

## 🔮 What's Next

Having completed these four foundations, natural next steps include:

| Topic | Builds On | What You'd Learn |
|-------|-----------|-----------------|
| **Convolutional VAE** | VAE | Use Conv layers instead of Dense for sharper images |
| **DCGAN** | GAN | Deep Convolutional GAN — industry standard for image generation |
| **BERT Fine-tuning** | Transformers | Train a pretrained model on your own classification task |
| **LangChain** | Transformers | Chain multiple LLM calls into complex workflows |
| **RAG (Retrieval Augmented Generation)** | Transformers | Give LLMs access to your own documents |
| **LoRA Fine-tuning** | Stable Diffusion | Fine-tune SD on custom image styles efficiently |

---

*Every file in this repository is written to be read like a textbook. The best way to use it is to open a file, run it, and read every comment from top to bottom.*