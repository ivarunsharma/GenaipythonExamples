import torch
from transformers import pipeline, set_seed
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# WHAT ARE TRANSFORMERS?
# ──────────────────────────────────────────────────────────────────────────────
# The architecture behind almost every modern AI model — GPT, BERT, Claude, etc.
#
# Core idea: instead of processing words one by one (like older models),
# transformers look at the ENTIRE sequence at once and figure out which
# words are most relevant to each other.
#
# Key mechanism — ATTENTION:
#   Given: "The cat sat on the mat because it was tired"
#   What does "it" refer to? The transformer learns to PAY ATTENTION to "cat"
#   when processing "it" — connecting related words regardless of distance.
#
# How it connects to what you've learned:
#   VAE/GAN/SD  → you built model architecture yourself from scratch
#   Transformers → pretrained models already built and trained, you just use them
#   Instead of training for days/weeks, you use models trained on billions
#   of examples and just apply them to your specific text.
#
# Libraries:
#   torch        — PyTorch, runs transformers under the hood
#   transformers — Hugging Face's toolbox giving access to thousands of
#                  pretrained models with minimal code
#   pandas       — displays results as clean tables instead of raw Python lists
#
# What the transformers library gives you (one consistent interface):
#   pipeline("text-classification") — sentiment analysis, positive/negative
#   pipeline("text-generation")     — generate text continuations (like GPT)
#   pipeline("summarization")       — summarize long text
#   pipeline("translation_en_to_fr")— translate between languages
#   pipeline("question-answering")  — answer questions from a passage
#   pipeline("ner")                 — find names, places, dates in text
#   pipeline("fill-mask")           — fill in the blank
#   Same pipeline() function, just different task string.
#
# What is Hugging Face?
#   A company that hosts pretrained models and makes them easy to use.
#   Think of it like GitHub but for AI models.
#   Anyone can upload and download models from there.
#
# What happens when you call pipeline():
#   1. Downloads pretrained model weights from huggingface.co (first time only)
#   2. Caches locally (~/.cache/huggingface/)
#   3. Loads tokenizer — converts raw text to numbers
#   4. Loads model — the actual transformer network
#   5. Returns ready-to-use function
#
# What is a Tokenizer?
#   Transformers can't read raw text — everything must be numbers.
#   Tokenizer splits text into chunks called TOKENS and maps each to a number:
#     "I love cats" → ["I", "love", "cats"] → [101, 2293, 8870]
#   Tokens are not always whole words:
#     "unbelievable" → ["un", "##believe", "##able"] → [2102, 7882, 3085]
#   The ## means it's a continuation of the previous token.
# ══════════════════════════════════════════════════════════════════════════════


# ── Sample Texts ───────────────────────────────────────────────────────────────
# Two opposite iPhone reviews — one clearly negative, one clearly positive.
# Classic setup for testing a sentiment classifier.
# If the model gets these two wrong, something is broken.

text1 = '''Extremely disappointed with my recent iPhone purchase from Apple. The device constantly lags, and the battery life is abysmal,
barely lasting through the day. Despite the hefty price tag, the performance is far from satisfactory. Customer support has been unhelpful,
providing no viable solutions to address these persistant issues. This experience has left me regretting my decision to choose Apple,
and I expected much better from a company known for its premium products.'''
# Key negative signals the transformer will focus on:
#   "Extremely disappointed", "constantly lags", "abysmal", "regretting"

text2 = '''I recently purchased an iPhone from Apple, and it has been an absolute delight! The device runs smoothly, and the battery life is impressive, easily lasting throughout the day.
The price, though high, is justified by the excellent performance and top-notch customer support. I am thoroughly satisfied with my decision to choose Apple, and it reaffirms their reputation
for delivering premium products. Highly recommended for anyone seeking a reliable and high-performance smartphone'''
# Key positive signals the transformer will focus on:
#   "absolute delight", "runs smoothly", "impressive", "Highly recommended"


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: TEXT CLASSIFICATION — Sentiment Analysis
# ──────────────────────────────────────────────────────────────────────────────
# Task: read a review and decide POSITIVE or NEGATIVE with a confidence score.
#
# Model used: distilbert-base-uncased-finetuned-sst-2-english (default)
#   DistilBERT = smaller, faster version of BERT. Good for laptops.
#   Already trained on millions of reviews — no training needed.
#   Downloads ~250MB first time, cached after.
#
# DISCRIMINATIVE transformer — reads and understands text → outputs a label.
# Different from generative transformers that write/continue text.
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PART 1: TEXT CLASSIFICATION (Sentiment Analysis)")
print("=" * 60)

classifier = pipeline("text-classification")
# Creates text classification pipeline with one line.
# Hugging Face picks best default model automatically.
# Same pipeline() function used for all NLP tasks — just different task string.

# ── Analyze text1 (negative review) ───────────────────────────────────────────
outputs1 = classifier(text1)
# Runs text1 through entire transformer pipeline:
#   raw text → tokenize → attention layers → classification → label + score
#
# Attention mechanism latches onto strongly negative words:
#   "disappointed", "abysmal", "regretting" → pushes output toward NEGATIVE
#
# Returns: [{'label': 'NEGATIVE', 'score': 0.9998}]
#   label = predicted sentiment
#   score = confidence (0 to 1). 0.9998 = 99.98% confident

print("\nNegative Review Result:")
print(pd.DataFrame(outputs1))
# pandas DataFrame displays result as neat table instead of raw Python list.

# ── Analyze text2 (positive review) ───────────────────────────────────────────
outputs2 = classifier(text2)
# Same pipeline, same process — different text, different attention focus.
# Attention latches onto: "delight", "smoothly", "impressive", "Highly recommended"
# These dominate and push output toward POSITIVE.

print("\nPositive Review Result:")
print(pd.DataFrame(outputs2))

# Expected combined results:
# ┌──────┬──────────┬────────┐
# │ Text │ Label    │ Score  │
# ├──────┼──────────┼────────┤
# │ text1│ NEGATIVE │ ~0.9998│
# │ text2│ POSITIVE │ ~0.9997│
# └──────┴──────────┴────────┘
# Both high confidence because reviews are unambiguously clear in sentiment.
# Ambiguous text like "battery great but camera terrible" would score lower.


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: TEXT GENERATION — Customer Service Response
# ──────────────────────────────────────────────────────────────────────────────
# Task: given a complaint + start of a response, continue writing the response.
#
# Model used: GPT-2
#   OpenAI's older but lightweight language model. Good for laptops.
#   GENERATIVE transformer — predicts what comes next → generates text.
#   Downloads ~500MB first time, cached after.
#
# GPT-2 vs DistilBERT:
#   DistilBERT — reads and understands text → classification (discriminative)
#   GPT-2      — predicts next word repeatedly → generation (generative)
#
# Realistic expectation:
#   GPT-2 is old (2019) and small. Output makes some sense but may go
#   off-topic or repeat itself. GPT-4 does this dramatically better
#   but GPT-2 runs on your laptop in seconds.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 2: TEXT GENERATION (Customer Service Response)")
print("=" * 60)

set_seed(42)
# Transformers use randomness during text generation.
# Without a fixed seed, every run produces different output.
# Setting seed to 42 (common convention, any number works) ensures
# you get the SAME output every run — useful for testing and reproducibility.

generator = pipeline("text-generation", model="gpt2")
# Creates text generation pipeline using GPT-2.
# Unlike classifier which picked default model, here we explicitly choose gpt2
# because it's the lightest real generative model available.

response = "Dear Patron, Thanks for writing in! I am sorry to hear your experience with us."
# Starter customer service response — beginning of what a support agent might write.

prompt = text1 + "\n\nCustomer service response:\n" + response
# Builds full prompt by combining:
#   [negative review text1]
#   + "Customer service response:"
#   + [starter response]
#
# Gives GPT-2 full context:
#   - Here's the complaint
#   - Here's how the response starts
#   - Now continue writing it...
#
# This technique is called PROMPT ENGINEERING — carefully structuring input
# to guide the model's output toward what you want.
# Same concept used in modern ChatGPT/Claude prompting.

outputs = generator(prompt, max_length=150)
# Runs GPT-2 on the prompt.
# Generates tokens one by one — each time predicting most likely next word
# given everything before it.
#
# max_length=150 — stop after 150 tokens total (prompt + generated text).
# Keeps output short and fast on laptop.
#
# How generation works step by step:
#   "Dear Patron, Thanks" → predict next word → "for"
#   "Dear Patron, Thanks for" → predict next word → "writing"
#   "Dear Patron, Thanks for writing" → predict next word → "in"
#   ... repeats until max_length reached

print("\nGenerated Customer Service Response:")
print(outputs[0]['generated_text'])
# outputs is a list (GPT-2 can generate multiple responses if asked).
# [0] grabs the first response.
# ['generated_text'] extracts full text — original prompt + generated continuation.


# ══════════════════════════════════════════════════════════════════════════════
# FULL RECAP
# ──────────────────────────────────────────────────────────────────────────────
# Two transformer applications in one file:
#
#   PART 1 — Text Classification (DistilBERT)
#     Input:  raw review text
#     Output: POSITIVE/NEGATIVE label + confidence score
#     Type:   Discriminative — reads and understands
#
#   PART 2 — Text Generation (GPT-2)
#     Input:  complaint + start of response
#     Output: completed customer service response
#     Type:   Generative — predicts and writes
#
# Both use the same pipeline() interface from Hugging Face transformers.
# Both use pretrained models — no training required.
# This is the power of transformers: apply state-of-the-art NLP in 3 lines.
#
# Journey so far:
#   VAE             → compress and reconstruct images
#   GAN             → generate sharp images via competition
#   Stable Diffusion → generate images from text via denoising
#   Transformers    → understand and generate text via attention
# ══════════════════════════════════════════════════════════════════════════════