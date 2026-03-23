# GenAI Python Examples

A progressive, hands-on collection of Generative AI examples in Python — from calling a cloud AI API for the very first time, through building stateful chatbots and multimodal PDF tools, all the way to implementing Variational Autoencoders, GANs, Stable Diffusion pipelines, and Transformer models.

Every file is written to be **read like a textbook**. Inline comments explain not just *what* the code does, but *why* it does it — with analogies, visual diagrams, and concept explanations woven directly into the source.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Learning Path](#learning-path)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Modules](#modules)
  - [1. FirstExampleAPIKey — Gemini API Basics](#1-firstexampleapikey--gemini-api-basics)
  - [2. SummaryExampleAPI — Text Summarization](#2-summaryexampleapi--text-summarization)
  - [3. ChatBotExample — Stateless vs Stateful Chatbots](#3-chatbotexample--stateless-vs-stateful-chatbots)
  - [4. PDFSummaryExample — PDF Understanding and Q&A](#4-pdfsummaryexample--pdf-understanding-and-qa)
  - [5. PythonCodeTest — Core ML Model Implementations](#5-pythoncodetest--core-ml-model-implementations)
- [Datasets](#datasets)
- [Architecture Comparisons](#architecture-comparisons)
- [Key Concepts Glossary](#key-concepts-glossary)
- [Laptop-Friendly Optimizations](#laptop-friendly-optimizations)
- [Model Downloads Reference](#model-downloads-reference)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Project Overview

This repository is structured as a two-track learning path through Generative AI:

**Track 1 — Cloud AI APIs (Google Gemini)**
Work with Google's Gemini 2.5 Flash model via the `google-genai` SDK. No GPU or local training required. These examples teach how to call a production-grade language model, build prompt-engineered requests, manage multi-turn conversation history, and process multimodal inputs such as PDFs.

**Track 2 — Local ML Model Implementation (TensorFlow / PyTorch)**
Build and train generative models from scratch using real datasets, with no API key required. These examples teach the fundamental architectures that power modern AI — from VAEs and GANs to Stable Diffusion and Transformers.

Both tracks reinforce each other: the cloud API examples show you what modern AI can do; the local implementations show you how it works under the hood.

---

## Learning Path

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  TRACK 1 — Cloud AI APIs (Google Gemini)                                     │
│                                                                              │
│  1. FirstExampleAPIKey                                                       │
│     Send your first message → list models → interactive Q&A                 │
│           │                                                                  │
│           ▼                                                                  │
│  2. SummaryExampleAPI                                                        │
│     Hardcoded text → user-pasted text → file-based summarization            │
│           │                                                                  │
│           ▼                                                                  │
│  3. ChatBotExample                                                           │
│     Stateless chatbot → stateful chatbot with conversation memory           │
│           │                                                                  │
│           ▼                                                                  │
│  4. PDFSummaryExample                                                        │
│     Multimodal PDF summary → interactive PDF Q&A loop                       │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  TRACK 2 — Local Model Implementations                                       │
│                                                                              │
│  5a. test_VAE.py                                                             │
│      Compress images to 2D, reconstruct them, generate new ones             │
│            │                                                                 │
│            ▼  (sharper output via adversarial training)                      │
│  5b. test_GAN.py                                                             │
│      Generator vs Discriminator competition — counterfeiter vs detective     │
│            │                                                                 │
│            ▼  (adds text guidance and denoising)                             │
│  5c. test_StableDiffusion.py                                                 │
│      Text prompt → image via CLIP + UNet + VAE + Scheduler                  │
│            │                                                                 │
│            ▼  (shifts to language)                                           │
│  5d. test_Transformers.py                                                    │
│      Sentiment analysis (DistilBERT) + text generation (GPT-2)              │
│            │                                                                 │
│            ▼  (pure Python, no ML framework)                                 │
│  5e. test_encodeDecode.py                                                    │
│      Run-length encoding and decoding for text files                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
GenAiPythonExamples/
│
├── .env                              # Your API key (not committed to git)
│
├── datasets/                         # Shared data files used across all modules
│   ├── summarization_dataset.txt     # Long-form text for summarization examples
│   ├── sample.pdf                    # PDF for multimodal summary and Q&A
│   ├── sampleText.txt                # Plain text for encode/decode testing
│   ├── sampleText_encoded.txt        # Auto-generated encoded output
│   └── sampleText_decoded.txt        # Auto-generated decoded output
│
├── FirstExampleAPIKey/               # Track 1 — Getting started with Gemini
│   ├── first_ai.py                   # Validate API key with a simple message
│   ├── list_models.py                # List all models on your API key
│   └── ask_question.py              # Interactive single-turn Q&A
│
├── SummaryExampleAPI/                # Track 1 — Text summarization
│   ├── summarize.py                  # Summarize hardcoded text
│   ├── summarize_input.py            # Summarize user-pasted text (interactive)
│   └── summarize_file.py             # Summarize from a .txt file
│
├── ChatBotExample/                   # Track 1 — Conversational AI patterns
│   ├── chatbot.py                    # Stateless chatbot (no memory)
│   └── chatbot_memory.py             # Stateful chatbot (full conversation history)
│
├── PDFSummaryExample/                # Track 1 — Multimodal PDF understanding
│   ├── pdf_summary.py                # One-shot PDF summarizer
│   └── pdf_summary_advanced.py       # PDF summarizer with interactive Q&A loop
│
├── PythonCodeTest/                   # Track 2 — Core generative ML implementations
│   ├── test_VAE.py                   # Variational Autoencoder (TensorFlow/Keras)
│   ├── test_GAN.py                   # Generative Adversarial Network (TensorFlow/Keras)
│   ├── test_StableDiffusion.py       # Stable Diffusion text-to-image (PyTorch)
│   ├── test_Transformers.py          # Sentiment analysis + text generation (PyTorch)
│   └── test_encodeDecode.py          # Run-length encoding/decoding utility
│
├── requirements.txt                  # All Python dependencies
└── README.MD                         # This file
```

---

## Prerequisites

| Requirement | Detail |
|---|---|
| Python | 3.9 or higher |
| pip | Latest recommended |
| Google AI API Key | Required for Track 1 modules only |
| Internet connection | Required for first-time Hugging Face model downloads |
| GPU | Optional — all examples are tuned to run on CPU |

> **No GPU needed.** Every example in this repository is configured to complete in minutes on a standard laptop CPU. Training sizes, epochs, and model selections have been deliberately reduced for this purpose.

---

## Installation

**1. Clone the repository**

```bash
git clone <your-repo-url>
cd GenAiPythonExamples
```

**2. Create and activate a virtual environment**

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install all dependencies**

```bash
pip install -r requirements.txt
```

> **PyTorch note:** The default `torch` install via pip is CPU-only. If you have an NVIDIA GPU and want to use CUDA acceleration, follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) to install the GPU-enabled build.

---

## Configuration

All Track 1 scripts (Gemini API) require a Google AI API key stored in a `.env` file at the project root.

**1. Create your `.env` file**

```bash
touch .env
```

**2. Add your API key**

```
GOOGLE_API_KEY=your_api_key_here
```

**3. Get an API key**

Visit [Google AI Studio](https://aistudio.google.com/) to generate a free API key. The Gemini 2.5 Flash model used throughout Track 1 has a generous free tier.

All Track 1 scripts load this file automatically using `python-dotenv`, regardless of the subdirectory they are run from:

```python
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
```

Track 2 scripts (VAE, GAN, Stable Diffusion, Transformers) require no API key.

---

## Modules

---

### 1. FirstExampleAPIKey — Gemini API Basics

The entry point for Track 1. Three minimal scripts that introduce the `google-genai` SDK with as little code as possible.

| Script | What it does |
|---|---|
| `first_ai.py` | Sends `"Hello, world!"` to Gemini 2.5 Flash and prints the response. Validates that your API key and SDK are working correctly. |
| `list_models.py` | Lists every Gemini model available under your API key. Useful for discovering which models you have access to. |
| `ask_question.py` | Prompts you to type any question, sends it to Gemini, and prints the answer. Your first interactive AI script. |

**Run:**

```bash
python FirstExampleAPIKey/first_ai.py
python FirstExampleAPIKey/list_models.py
python FirstExampleAPIKey/ask_question.py
```

**Key concepts introduced:**

- Initializing `genai.Client` with an API key
- `client.models.generate_content()` — the core API call
- Specifying a model by name (`gemini-2.5-flash`)
- Accessing the response via `.text`

---

### 2. SummaryExampleAPI — Text Summarization

Three progressively more capable summarization tools, all using the same Gemini API call but with increasingly sophisticated input handling.

| Script | What it does |
|---|---|
| `summarize.py` | Summarizes a hardcoded paragraph about AI in 2 sentences. Demonstrates the most basic prompt pattern. |
| `summarize_input.py` | Multi-line interactive input mode. Paste any text, type `END` on a new line to finish, then select summary length (1 sentence / 3 sentences / 1 paragraph). |
| `summarize_file.py` | Automatically loads `datasets/summarization_dataset.txt` and summarizes the full document in one paragraph. |

**Run:**

```bash
python SummaryExampleAPI/summarize.py
python SummaryExampleAPI/summarize_input.py
python SummaryExampleAPI/summarize_file.py
```

**Key concepts introduced:**

- Prompt engineering — embedding instructions inside the `contents` string
- Dynamic prompts using f-strings: `f"Summarize the following text in {length}: {text}"`
- Reading from files and passing large text bodies to the API
- User-driven output control (length selection menu)

---

### 3. ChatBotExample — Stateless vs Stateful Chatbots

Two chatbots that appear identical from the outside but behave fundamentally differently because of how (or whether) they maintain conversation history.

| Script | What it does |
|---|---|
| `chatbot.py` | **Stateless.** Each message is sent in isolation. The model has no memory of previous turns. Every question is answered as if it is the first message of a new conversation. |
| `chatbot_memory.py` | **Stateful.** Every message and reply is appended to a `history` list. The full list is sent to the model on every turn, giving it complete context of the conversation so far. |

**Run:**

```bash
python ChatBotExample/chatbot.py
python ChatBotExample/chatbot_memory.py
```

Type `quit` to exit.

**Key concepts introduced:**

- Stateless vs stateful API design
- Conversation history as a list of `{"role": ..., "parts": [...]}` dictionaries
- The `role` field: `"user"` for human turns, `"model"` for AI replies
- How context window works: sending the full history on each API call
- The practical cost of memory: history grows with every turn, consuming tokens

**Demonstration:**

Run both chatbots. In the first message, say `"My name is Alex"`. In the second message, ask `"What is my name?"`. The stateless bot will not know. The memory bot will answer correctly.

**How the history list grows:**

```
Turn 1:  history = [{user: "My name is Alex"}, {model: "Nice to meet you, Alex!"}]
Turn 2:  history = [{user: "..."}, {model: "..."}, {user: "What is my name?"}, {model: "Your name is Alex."}]
Turn 3:  history = [...previous 4 entries..., {user: "..."}, {model: "..."}]
```

---

### 4. PDFSummaryExample — PDF Understanding and Q&A

Demonstrates Gemini's **multimodal** capability — the model reads and understands a PDF document's content directly, without any text extraction or preprocessing on your side.

| Script | What it does |
|---|---|
| `pdf_summary.py` | Loads `datasets/sample.pdf`, encodes it as base64, sends it to Gemini as a `Part` object alongside a text instruction, and prints a one-paragraph summary. |
| `pdf_summary_advanced.py` | Loads the PDF, lets you choose summary depth (short / medium / detailed), prints the summary, then enters an interactive Q&A loop where you can ask any question about the PDF's content. |

**Run:**

```bash
python PDFSummaryExample/pdf_summary.py
python PDFSummaryExample/pdf_summary_advanced.py
```

Ensure `datasets/sample.pdf` exists before running. Type `quit` to exit the Q&A loop.

**Key concepts introduced:**

- Sending binary data to a multimodal model using `types.Part.from_bytes()`
- `mime_type` specification (`"application/pdf"`)
- Combining a binary part and a text instruction in a single `contents` list
- Interactive Q&A over a fixed document — the PDF is sent on every question, keeping the model's context anchored to its content

**How Gemini handles PDFs:**

```
PDF bytes → base64 encoded → Part.from_bytes(data, mime_type="application/pdf")
                                         │
                                         ▼
         contents = [pdf_part, "Summarize this PDF in one paragraph."]
                                         │
                                         ▼
                              Gemini processes both inputs
                              and returns a text response
```

---

### 5. PythonCodeTest — Core ML Model Implementations

Five standalone scripts implementing foundational generative AI architectures. No API key required. Each script is extensively commented to explain every design decision, hyperparameter, and layer choice.

---

#### test_VAE.py — Variational Autoencoder

**Framework:** TensorFlow / Keras
**Dataset:** MNIST handwritten digits (auto-downloaded, ~11 MB)
**Training time:** ~2–3 minutes on CPU

**What is a VAE?**

A Variational Autoencoder learns to compress images into a compact numerical representation and reconstruct them. The key differentiator from a standard autoencoder: instead of compressing to exact numbers, it compresses to a *distribution* — defined by a mean and variance. This means you can sample a random point from that distribution and generate a brand new image.

**Architecture:**

```
ENCODER
  Input (28×28) → Flatten (784) → Dense(128, relu)
                                        ├─► z_mean    (2)
                                        └─► z_log_var (2)

SAMPLING LAYER
  z = z_mean + exp(0.5 × z_log_var) × ε   (ε = random noise)

DECODER
  z (2) → Dense(128, relu) → Dense(784, sigmoid) → Reshape (28×28)
```

**Why 2 dimensions?** A 2D latent space can be visualized directly on an X-Y plot. Real applications use 64–256 dimensions for sharper results.

**The two losses:**

| Loss | What it measures | Why it is needed |
|---|---|---|
| Reconstruction loss | Pixel-by-pixel difference between input and output | Forces the decoder to rebuild images accurately |
| KL divergence loss | How far the learned distribution deviates from a standard normal | Keeps the latent space organized so random sampling produces meaningful images |

Without KL loss, the encoder could output wild, scattered ranges that reconstruct well but produce garbage when you sample random points for generation.

**Run:**

```bash
python PythonCodeTest/test_VAE.py
```

**Output:** A 4×4 matplotlib grid of 16 generated digit images invented from random latent points — never seen during training.

**Expected appearance:** Slightly blurry. This is mathematically expected — VAE optimizes for pixel-level average similarity, which produces smooth but imprecise outputs. Increase `latent_dim` to 32 or 64 for sharper results at the cost of longer training.

---

#### test_GAN.py — Generative Adversarial Network

**Framework:** TensorFlow / Keras
**Dataset:** MNIST (30,000 samples)
**Training time:** ~5–10 minutes on CPU

**What is a GAN?**

A GAN trains two neural networks in direct competition:

- **Generator** — creates fake images from random noise. Goal: fool the discriminator.
- **Discriminator** — classifies images as real or fake. Goal: catch the generator.

They improve by competing. The generator gets better at faking; the discriminator gets better at detecting. Think of it as **Counterfeiter vs Detective** — both become more skilled through the adversarial relationship.

**Architecture:**

```
GENERATOR
  100 noise → Dense(256) → Dense(512) → Dense(1024) → Dense(784, tanh) → Reshape(28×28×1)
               LeakyReLU    LeakyReLU    LeakyReLU
               BatchNorm    BatchNorm    BatchNorm

DISCRIMINATOR
  Image(28×28×1) → Flatten(784) → Dense(512) → Dense(256) → Dense(1, sigmoid)
                                   LeakyReLU    LeakyReLU    0=Fake / 1=Real
                                   Dropout      Dropout
```

**Training loop (per batch):**

```
Step 1 — Train Discriminator (Generator frozen):
  Real images + smooth labels (~0.8–1.0) → discriminator.train_on_batch()
  Fake images + smooth labels (~0.0–0.2) → discriminator.train_on_batch()

Step 2 — Train Generator (Discriminator frozen):
  Noise → Generator → Fake image → Discriminator
  Target label ~1.0 — "these fakes should be called real"
  Only generator weights update
```

**Stability techniques used:**

| Technique | Effect |
|---|---|
| Label smoothing | Real labels ~0.9 (not 1.0), fake labels ~0.1 (not 0.0) — prevents discriminator overconfidence |
| Instance noise | Small Gaussian noise added to images — forces discriminator to learn structure, not memorize pixels |
| Label flipping | 5% of generator training labels randomly flipped — prevents generator exploiting discriminator patterns |
| LeakyReLU | Passes 20% of negative values through — prevents dead neurons |
| Dropout (30%) | Randomly disables neurons — keeps discriminator from becoming too powerful too fast |

**What to watch during training:**

```
D acc  → Ideally 0.5–0.8. At 1.0 = discriminator too powerful, generator gets no useful feedback.
D loss → Should stay moderate, not collapse to 0.
G loss → Should gradually decrease as generator improves.
```

**Run:**

```bash
python PythonCodeTest/test_GAN.py
```

**Output:** A 10×10 matplotlib grid of 100 generated digit images. Noticeably sharper than VAE output, because GAN optimizes for perceptual realism (fooling a discriminator) rather than pixel-level average.

---

#### test_StableDiffusion.py — Text-to-Image Generation

**Framework:** PyTorch + Hugging Face Diffusers
**Model:** `nota-ai/bk-sdm-tiny` (CPU-friendly, small download)
**Runtime:** ~1–3 minutes on CPU

**What is Stable Diffusion?**

Stable Diffusion generates images from text descriptions. It is the most architecturally complex model here — combining four components into one seamless pipeline.

**The four components:**

| Component | Role |
|---|---|
| **CLIP** | Converts your text prompt into numerical embeddings that guide generation |
| **Scheduler** | Controls how many denoising steps to run and how large each step is |
| **UNet** | Does the actual denoising — takes noisy latent + text embeddings, predicts denoised version |
| **VAE** | Decodes the final denoised latent into a viewable pixel image |

**The diffusion concept:**

```
FORWARD PROCESS (training, not run here):
  Real image → add noise (step 1) → add more noise (step 2) → ... → pure noise (step 1000)

REVERSE PROCESS (generation, what this script does):
  Pure noise → UNet denoise (step 1) → ... → UNet denoise (step 20) → VAE decode → image
                      ↑
              guided at every step by CLIP text embeddings
```

Think of it as: **crumpling a photograph into a ball** (forward: adding noise), then **learning to uncrumple it back** (reverse: denoising). The text prompt tells the model *what image* to uncrumple toward.

**Full generation pipeline:**

```
Text: "A serene sunset over a calm lake"
         │
         ▼
     CLIP encoder
         │ (text embeddings)
         ▼
     Pure random noise + text embeddings
         │
         ▼ (×20 denoising steps)
     UNet (guided by text at each step)
         │
         ▼
     VAE decoder
         │
         ▼
     Generated image (saved as image_0.jpg)
```

**Positive and negative prompts:**

```python
positive_prompt = "A serene sunset over a calm lake"
# Steers generation TOWARD these qualities

negative_prompt = "blurry, distorted, low quality"
# Steers generation AWAY from these qualities
```

This dual-prompt system is unique to Stable Diffusion — VAE and GAN have no concept of text guidance.

**CPU optimizations applied:**

| Setting | This repo | Production default | Why reduced |
|---|---|---|---|
| Model size | ~MB (tiny) | ~4 GB (v1.5) | Download and RAM constraints |
| dtype | `float32` | `float16` | `float16` requires GPU |
| Inference steps | 20 | 50 | 2.5× faster |
| Resolution | 256×256 | 512×512 | 4× fewer pixels = ~4× faster |

**Run:**

```bash
python PythonCodeTest/test_StableDiffusion.py
```

**Output:** A JPEG image saved as `image_0.jpg` and displayed via matplotlib. Quality will be limited by the tiny model — swap in `runwayml/stable-diffusion-v1-5` for production quality (requires ~4 GB download and 8 GB RAM).

---

#### test_Transformers.py — Sentiment Analysis and Text Generation

**Framework:** PyTorch + Hugging Face Transformers
**Models:** DistilBERT (~250 MB), GPT-2 (~500 MB) — downloaded once, cached
**Runtime:** ~1–2 minutes on CPU

**What are Transformers?**

The architecture behind virtually every modern language model — GPT, BERT, Claude, and others. The core innovation is the **attention mechanism**: instead of processing words one by one (as older RNNs did), transformers look at the entire sequence at once and learn which words are most relevant to each other.

**The attention insight:**

> In `"The cat sat on the mat because it was tired"` — the transformer learns that `"it"` refers to `"cat"` by attending to the most contextually relevant word, regardless of distance in the sentence.

This script demonstrates two types of transformer, back to back, using the same `pipeline()` interface.

---

**Part 1 — Text Classification (DistilBERT)**

*Task: determine whether an iPhone review is positive or negative.*

```
Raw review text
      │
      ▼ Tokenizer: "I love cats" → ["I", "love", "cats"] → [101, 2293, 8870]
      │
      ▼ DistilBERT attention layers
        (bidirectional — reads the full sentence both ways)
      │
      ▼ Classification head
      │
      ▼ POSITIVE (0.9997) or NEGATIVE (0.9998)
```

DistilBERT is a *discriminative* transformer — it reads and understands text to output a label. It was pretrained on Wikipedia and BooksCorpus, then fine-tuned on the SST-2 movie review sentiment dataset.

**Expected output:**

```
Negative Review Result:
      label     score
0  NEGATIVE  0.999812

Positive Review Result:
      label     score
0  POSITIVE  0.999711
```

Both are high confidence because the test reviews are unambiguous. A mixed review like `"battery great but camera terrible"` would score lower — conflicting signals produce less certain output.

---

**Part 2 — Text Generation (GPT-2)**

*Task: continue a customer service response given the complaint that triggered it.*

```
[Complaint text] + "\n\nCustomer service response:\n" + [starter sentence]
      │
      ▼ GPT-2: predict next token → append → predict next → append → ...
                (repeats until max_length=150 tokens reached)
      │
      ▼ Completed response continuation
```

GPT-2 is a *generative* transformer — it predicts the next token given all previous tokens, one token at a time. It reads left to right (unidirectional), unlike DistilBERT's bidirectional reading.

The technique of structuring input carefully to guide generation is called **prompt engineering** — the same principle behind modern ChatGPT and Claude prompting.

**Discriminative vs Generative — side by side:**

| Property | DistilBERT (discriminative) | GPT-2 (generative) |
|---|---|---|
| Direction | Bidirectional (reads both ways) | Left-to-right only |
| Task | Understand → classify | Predict → generate |
| Input | Full text at once | Partial text (prompt) |
| Output | Label + score | Text continuation |

**Run:**

```bash
python PythonCodeTest/test_Transformers.py
```

**Hugging Face pipeline tasks reference:**

```python
pipeline("text-classification")       # Sentiment, topic classification
pipeline("text-generation")           # Continue/complete text (GPT-style)
pipeline("summarization")             # Condense long documents
pipeline("translation_en_to_fr")      # Translate between languages
pipeline("question-answering")        # Extract answers from a passage
pipeline("ner")                       # Named entity recognition (names, places, dates)
pipeline("fill-mask")                 # Predict a masked word in a sentence
pipeline("zero-shot-classification")  # Classify without task-specific training
```

---

#### test_encodeDecode.py — Run-Length Encoding

A pure Python utility implementing run-length encoding (RLE) — a simple lossless compression algorithm that replaces repeated characters with a count and character pair.

**How RLE works:**

```
Input text:    "AAABBBCCDDDDEE"
Encoded pairs: [(A,3), (B,3), (C,2), (D,4), (E,2)]
File format:   3,65\n3,66\n2,67\n4,68\n2,69
                     ↑
               ASCII ordinal (A=65, B=66, ...)
```

The script reads `datasets/sampleText.txt`, encodes it to `datasets/sampleText_encoded.txt`, then decodes the encoded file back to `datasets/sampleText_decoded.txt`, printing file sizes at each step.

**Run:**

```bash
python PythonCodeTest/test_encodeDecode.py
```

**Key concepts introduced:**

- Run-length encoding as a general compression principle
- Using `ord()` and `chr()` to convert between characters and ASCII codes
- File I/O with context managers
- Measuring compression ratio via `os.path.getsize()`

---

## Datasets

All datasets live in the `datasets/` folder at the project root — shared across all modules.

| File | Used by | Description |
|---|---|---|
| `summarization_dataset.txt` | `SummaryExampleAPI/summarize_file.py` | Long-form text document for file-based summarization |
| `sample.pdf` | `PDFSummaryExample/*.py` | PDF document for multimodal summarization and Q&A |
| `sampleText.txt` | `PythonCodeTest/test_encodeDecode.py` | Plain text for run-length encoding testing |
| `sampleText_encoded.txt` | Auto-generated | Run-length encoded output (created on first run) |
| `sampleText_decoded.txt` | Auto-generated | Decoded reconstruction (created on first run) |

Scripts reference the datasets folder using a root-relative path:

```python
os.path.join(os.path.dirname(__file__), '..', 'datasets', 'filename.txt')
```

This resolves correctly regardless of which subdirectory you invoke Python from.

---

## Architecture Comparisons

### Image Generation: VAE vs GAN vs Stable Diffusion

| Aspect | VAE | GAN | Stable Diffusion |
|---|---|---|---|
| Training | Single model, dual loss | Two competing models | Pretrained — no training here |
| Input at generation | Random latent point (2D) | Random noise vector (100D) | Text prompt |
| Output quality | Blurry (pixel averaging) | Sharp (adversarial pressure) | Very sharp (iterative denoising) |
| Text guidance | No | No | Yes |
| Interpretable latent | Yes (mean + variance) | No | Partial |
| Training stability | Very stable | Notoriously tricky | Stable (pretrained) |
| GPU requirement | No | No | No (slow on CPU) |

### VAE vs GAN — Why the output quality differs

| Model | Optimization target | Result |
|---|---|---|
| VAE | Minimize average pixel difference between input and reconstruction | Blurry — averaging many plausible outputs produces a blurry mean |
| GAN | Generate images realistic enough to fool a trained classifier | Sharp — must look convincingly real, not just statistically average |

### All Four Models — Common Building Blocks

| Concept | VAE | GAN | Stable Diffusion | Transformers |
|---|---|---|---|---|
| Normalization | ÷255 → [0,1] | −127.5÷127.5 → [−1,1] | Built-in | Tokenization |
| Loss function | Reconstruction + KL | Binary crossentropy | Diffusion loss | Cross-entropy |
| Latent space | Explicit (2D) | Implicit (100D noise) | VAE inside | Token embeddings |
| Generation | Decoder | Generator | UNet + VAE | GPT-2 |
| Framework | TensorFlow | TensorFlow | PyTorch | PyTorch |

---

## Key Concepts Glossary

| Term | Definition |
|---|---|
| **Latent space** | A compressed numerical representation of data. VAE maps a 784-pixel image to 2 numbers. |
| **Encoder** | Neural network that compresses input into latent space. |
| **Decoder / Generator** | Neural network that reconstructs or creates output from latent space. |
| **Normalization** | Scaling input values to a small, consistent range (0–1 or −1 to 1) for stable training. |
| **Activation function** | Non-linear transformation after each layer. `relu`, `sigmoid`, `tanh`, `LeakyReLU`. |
| **Loss function** | Mathematical measure of how wrong the model is. Training minimizes this. |
| **Backpropagation** | Algorithm that adjusts weights based on the gradient of the loss. |
| **Epoch** | One complete pass through the entire training dataset. |
| **Batch size** | Number of samples processed per gradient update step. |
| **Learning rate** | How large each weight update step is. Too high = unstable. Too low = slow. |
| **Dense layer** | Fully connected layer — every input neuron connects to every output neuron. |
| **Dropout** | Randomly disabling neurons during training to prevent overfitting and overconfidence. |
| **BatchNormalization** | Rescaling layer outputs to a stable range after each batch. |
| **LeakyReLU** | Activation that lets small negatives pass through (20%) unlike relu which kills them entirely. |
| **KL divergence** | Measures how different one probability distribution is from another. Used in VAE loss. |
| **Attention** | Mechanism that learns which parts of a sequence are most relevant to each other. Core of transformers. |
| **Tokenizer** | Splits raw text into chunks (tokens) and maps each to a number. |
| **Token** | The basic unit of text a transformer processes — can be a word, subword, or character. |
| **Embedding** | Dense numerical vector representing a token in high-dimensional space. |
| **Prompt engineering** | Carefully structuring input text to guide model output toward a desired result. |
| **Inference steps** | Number of denoising iterations in Stable Diffusion. More steps = higher quality, slower. |
| **Pretrained model** | A model already trained on massive data, downloadable and usable without further training. |
| **Fine-tuning** | Further training a pretrained model on a specific task or dataset. |
| **CLIP** | Model trained to understand the relationship between images and text. |
| **UNet** | Neural network shaped like a U — compresses down then expands back up with skip connections. |
| **Diffusion** | Process of gradually adding noise to data, then learning to reverse it for generation. |
| **Label smoothing** | Using soft labels (0.9 instead of 1.0) to prevent model overconfidence during training. |
| **Instance noise** | Adding small Gaussian noise to training images to prevent memorization of exact pixel patterns. |
| **Stateless** | Each API call has no knowledge of previous calls — no conversation memory. |
| **Stateful** | Conversation history is maintained and sent to the model on every turn. |
| **Multimodal** | A model that can process more than one type of input — e.g., text and images or PDFs together. |
| **Hugging Face** | Platform hosting thousands of pretrained models, downloadable via the `transformers` library. |

---

## Laptop-Friendly Optimizations

All examples are designed to run on a standard CPU laptop. Here is what was reduced from typical production settings and why:

| File | Production setting | This repo | Approximate speedup |
|---|---|---|---|
| `test_VAE.py` | 60,000 samples, Dense(512), latent_dim=128, 100 epochs | 5,000 samples, Dense(128), latent_dim=2, 10 epochs | ~50× faster |
| `test_GAN.py` | 60,000 samples, 50 epochs | 30,000 samples, 15 epochs | ~6× faster |
| `test_StableDiffusion.py` | SD v1.5 (4 GB), 512×512, 50 steps, float16 | Tiny model (MB), 256×256, 20 steps, float32 | ~100× faster |
| `test_Transformers.py` | Large fine-tuned models | DistilBERT + GPT-2 (smallest capable public models) | Optimal for CPU |

### Saving Trained Models (Optional)

By default, trained weights exist only in memory and are lost when the script ends. To save and reload without retraining:

```python
# After training — TensorFlow / Keras
generator.save('generator.keras')
vae.encoder.save('vae_encoder.keras')
vae.decoder.save('vae_decoder.keras')

# To reload next time
from tensorflow.keras.models import load_model
generator = load_model('generator.keras')
```

---

## Model Downloads Reference

Hugging Face models are downloaded on first run and cached at `~/.cache/huggingface/`. Subsequent runs use the local cache.

| Script | Model | Size | Notes |
|---|---|---|---|
| `test_StableDiffusion.py` | `nota-ai/bk-sdm-tiny` | ~few MB | Fast, low quality — good for pipeline testing |
| `test_Transformers.py` | `distilbert-base-uncased-finetuned-sst-2-english` | ~250 MB | Default text-classification model |
| `test_Transformers.py` | `gpt2` | ~500 MB | Lightest real generative model |

**For production quality Stable Diffusion** (requires ~4 GB disk, ~8 GB RAM):

```python
# In test_StableDiffusion.py, replace model_id with:
model_id = "runwayml/stable-diffusion-v1-5"
```

---

## Troubleshooting

**VAE / GAN: `ModuleNotFoundError: No module named 'tensorflow'`**

```bash
pip install tensorflow
```

**Stable Diffusion: `ValueError: Input image size doesn't match model`**

Ensure `safety_checker=None` is passed to `from_pretrained()`:

```python
pipeline = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None)
```

**Stable Diffusion: Out of memory on CPU**

Reduce resolution and steps further:

```python
image = pipeline(prompt=prompt, height=128, width=128, num_inference_steps=10).images[0]
```

**Transformers: Very slow first run**

Normal — models are downloading (~750 MB total). Check `~/.cache/huggingface/` to confirm progress.

**GAN: Generated images are all black or all white**

The model needs more training. Increase epochs:

```python
train(epochs=30, batch_size=64)
```

**Track 1: `GOOGLE_API_KEY` not found**

Ensure your `.env` file exists at the project root (not inside a subdirectory) and contains:

```
GOOGLE_API_KEY=your_actual_key_here
```

**General: Suppress TensorFlow startup warnings**

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*The best way to use this repository is to open a script, run it, and read every comment from top to bottom while the output appears. The comments are the tutorial.*
