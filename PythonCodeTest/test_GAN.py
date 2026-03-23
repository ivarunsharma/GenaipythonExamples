import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# WHAT IS A GAN (Generative Adversarial Network)?
# ──────────────────────────────────────────────────────────────────────────────
# A GAN has two neural networks competing against each other:
#
#   GENERATOR     — creates fake images from random noise.
#                   Tries to fool the discriminator.
#   DISCRIMINATOR — looks at images and decides real or fake.
#                   Tries to catch the generator.
#
# They train together. Generator gets better at faking,
# discriminator gets better at detecting.
# Over time the generator produces very realistic images.
#
# Think of it as: COUNTERFEITER vs DETECTIVE.
# Both get better by competing with each other.
#
# GAN vs VAE:
#   VAE  — compresses images to a range, generates by sampling that range. Blurry.
#   GAN  — generator learns from direct feedback of discriminator. Much sharper.
#
# New layers introduced in GAN (not seen in VAE):
#   LeakyReLU       — like relu but lets tiny negatives through (prevents dead neurons)
#   BatchNormalization — rescales values to stable range between layers
#   Dropout         — randomly turns off neurons to prevent overconfidence
#   Sequential      — simpler way to stack layers one after another
# ══════════════════════════════════════════════════════════════════════════════


# ── Load and Prepare Data ──────────────────────────────────────────────────────
# MNIST: 70,000 handwritten digit images (0–9), each 28×28 pixels.

(X_train, _), (_, _) = mnist.load_data()
# Both _ and (_, _) throw away labels and test set entirely.
# GAN only needs training images — no labels needed (unsupervised).

# Use only 30,000 samples to keep it fast on a laptop.
X_train = X_train[:30000]

X_train = (X_train.astype(np.float32) - 127.5) / 127.5
# Normalize to -1 to 1 (different from VAE which used 0 to 1).
# Why -1 to 1? Generator's final layer uses 'tanh' which outputs -1 to 1.
# Data range must match output activation range.
# Why 127.5? It's exactly the midpoint of 0–255:
#   0   → (0 - 127.5) / 127.5   = -1.0  (black)
#   255 → (255 - 127.5) / 127.5 =  1.0  (white)
#   127 → (127 - 127.5) / 127.5 ≈  0.0  (mid gray)

X_train = np.expand_dims(X_train, axis=3)
# MNIST shape is (10000, 28, 28) — no channel info.
# Adding channel dimension: (10000, 28, 28, 1)
# The 1 = grayscale (1 channel). Color images would be 3 (RGB).
# Convolutional-style layers expect this 4D shape.


# ── Label Smoothing Functions ──────────────────────────────────────────────────
# Instead of labeling real=1 and fake=0 exactly, we use fuzzy ranges.
# Why? Perfect labels make discriminator overconfident → stops giving
# generator useful feedback → training collapses.
# Fuzzy labels keep discriminator humble and training stable.

def smooth_positive_labels(size):
    return np.random.uniform(low=0.8, high=1.0, size=(size, 1))
    # Real images labeled as random value between 0.8 and 1.0 (not exactly 1)

def smooth_negative_labels(size):
    return np.random.uniform(low=0.0, high=0.2, size=(size, 1))
    # Fake images labeled as random value between 0.0 and 0.2 (not exactly 0)


# ══════════════════════════════════════════════════════════════════════════════
# GENERATOR — Creates fake images from random noise
# ──────────────────────────────────────────────────────────────────────────────
# Flow: 100 random numbers → 256 → 512 → 1024 → 784 → 28×28×1 image
#
# Expanding pattern: noise → more neurons → more neurons → pixel output
# Think of it as brainstorming (expand) then writing final draft (compress to 784)
# 784 is fixed — it's always 28×28 pixels = 784 numbers
# ══════════════════════════════════════════════════════════════════════════════

def create_generator():
    model = Sequential()

    model.add(Dense(256, input_dim=100))
    # Takes 100 random numbers (noise) as input, expands to 256 neurons.
    # This 100-number noise vector is the seed for the entire fake image.

    model.add(LeakyReLU(alpha=0.2))
    # alpha=0.2 means negative values pass through at 20% strength.
    # Regular relu: -5 → 0 (killed)
    # LeakyReLU:    -5 → -1 (20% passes through)
    # GANs prefer this — killing negatives causes neurons to "die" and stop learning.

    model.add(BatchNormalization(momentum=0.8))
    # Rescales values to a stable range after each layer.
    # momentum=0.8 controls how much it remembers from previous batches.
    # Higher = smoother stabilization.

    # Pattern Dense → LeakyReLU → BatchNorm repeats, expanding capacity:
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    # By here the network has 1024 neurons worth of learned patterns.
    # Now compress to actual pixel output:

    model.add(Dense(784, activation='tanh'))
    # 784 = 28×28 pixels. One value per pixel.
    # 'tanh' outputs values between -1 and 1 — matches our normalized data range.
    # This is why we normalized data to -1 to 1, not 0 to 1 like VAE.

    model.add(Reshape((28, 28, 1)))
    # 784 flat numbers → 28×28×1 image shape. Ready to be judged by discriminator.

    return model


# ══════════════════════════════════════════════════════════════════════════════
# DISCRIMINATOR — Judges images as real or fake
# ──────────────────────────────────────────────────────────────────────────────
# Flow: 28×28×1 image → Flatten → 784 → 512 → 256 → 1 number (real/fake score)
#
# Opposite direction to generator — compresses image down to a single decision.
# Uses Dropout instead of BatchNorm to prevent becoming too confident too fast.
# ══════════════════════════════════════════════════════════════════════════════

def create_discriminator(dropout_rate=0.3):
    model = Sequential()

    model.add(Flatten(input_shape=(28, 28, 1)))
    # Takes 28×28×1 image, flattens to 784 numbers.
    # Opposite of generator's Reshape. Full flow: 784 → 512 → 256 → 1

    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout_rate))
    # Dropout randomly turns off 30% of neurons each training step.
    # Prevents discriminator from becoming too powerful too fast.
    # If discriminator gets too good, generator never learns.
    # Note: pattern here is Dense → LeakyReLU → Dropout
    #       (different from generator which uses Dense → LeakyReLU → BatchNorm)

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))
    # Final output: 1 single number between 0 and 1.
    # Close to 1 = real image
    # Close to 0 = fake image
    # sigmoid is perfect for binary real/fake decisions.

    return model


# ── Optimizers ─────────────────────────────────────────────────────────────────
# Two separate optimizers — discriminator and generator train at different
# times with different data, so independent optimizers keep their learning
# states separate.

disc_opt = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
gan_opt  = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
# learning_rate=0.0002 — much smaller than usual (0.001). GANs are fragile,
#                        smaller steps = more stable training.
# beta_1=0.5           — normally 0.9. Lowered because too much gradient memory
#                        causes instability in GANs.
# beta_2=0.999         — left at default. Long-term gradient memory.
# These are standard DCGAN settings proven through research.


# ── Create and Compile Models ──────────────────────────────────────────────────

discriminator = create_discriminator()
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=disc_opt,
    metrics=['accuracy']
)
# Discriminator compiled standalone — it trains on its own each step.
# binary_crossentropy = perfect for yes/no (real/fake) decisions.

generator = create_generator()
# Generator NOT compiled standalone — it never trains alone.
# It only trains through the combined GAN model below.

# ── Build Combined GAN Model ───────────────────────────────────────────────────

discriminator.trainable = False
# Freeze discriminator weights for generator training.
# When GAN trains, only generator weights update.
# Discriminator just provides feedback, doesn't learn during this phase.

gan_input  = Input(shape=(100,))          # 100 random numbers go in
synthetic  = generator(gan_input)          # generator creates fake image
gan_output = discriminator(synthetic)      # discriminator judges it
gan = Model(inputs=gan_input, outputs=gan_output)
gan.compile(loss='binary_crossentropy', optimizer=gan_opt)
# Combined pipeline:
# 100 random numbers → Generator → fake image → Discriminator → real/fake score
#
# GAN's job: train generator to make discriminator score close to 1 (fooled).


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────
# Each batch has TWO steps:
#   Step 1 — Train Discriminator (generator frozen)
#             Real images → should output ~1
#             Fake images → should output ~0
#   Step 2 — Train Generator (discriminator frozen)
#             Noise → fake images → discriminator → should output ~1
#             Generator learns to fool the discriminator
# ══════════════════════════════════════════════════════════════════════════════

def train(epochs=5, batch_size=128, noise_dim=100, instance_noise_std=0.05):
    # epochs=5       — reduced from 50 to keep it fast on laptop
    # batch_size=128 — larger than VAE's 64, GANs benefit from larger batches
    # noise_dim=100  — each fake image starts from 100 random numbers
    # instance_noise_std=0.05 — tiny noise added to images for stability

    batches_per_epoch = X_train.shape[0] // batch_size
    # 10,000 images ÷ 128 = ~78 batches per epoch

    for epoch in range(epochs):
        for batch in range(batches_per_epoch):

            # ── Step 1: Train Discriminator ────────────────────────────────────
            discriminator.trainable = True
            # Unfreeze discriminator — its turn to learn.

            # Train on REAL images:
            real_images = X_train[batch * batch_size:(batch + 1) * batch_size]
            # Grab next 128 real images by slicing training data.

            if instance_noise_std:
                real_images = real_images + np.random.normal(0, instance_noise_std, real_images.shape)
                real_images = np.clip(real_images, -1.0, 1.0)
            # Add tiny random noise to real images — called INSTANCE NOISE.
            # Makes discrimination harder, forces discriminator to focus on
            # overall structure rather than memorizing exact pixel patterns.
            # clip keeps values within -1 to 1 range.

            real_labels = smooth_positive_labels(batch_size)
            # Real images get fuzzy labels ~0.8-1.0 (not exactly 1).
            d_loss_real, d_acc_real = discriminator.train_on_batch(real_images, real_labels)
            # train_on_batch = one weight update step on this batch only.

            # Train on FAKE images:
            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            # Generate 128 random noise vectors, each 100 numbers long.

            fake_images = generator.predict_on_batch(noise)
            # Feed noise through generator → 128 fake images.
            # predict_on_batch = just generate, no training here.

            if instance_noise_std:
                fake_images = fake_images + np.random.normal(0, instance_noise_std, fake_images.shape)
                fake_images = np.clip(fake_images, -1.0, 1.0)
            # Same instance noise applied to fake images too.

            fake_labels = smooth_negative_labels(batch_size)
            # Fake images get fuzzy labels ~0.0-0.2 (not exactly 0).
            d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_images, fake_labels)

            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_acc  = 0.5 * (d_acc_real + d_acc_fake)
            # Average both losses and accuracies for reporting.

            # ── Step 2: Train Generator ────────────────────────────────────────
            discriminator.trainable = False
            # Freeze discriminator again — generator's turn now.

            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            # Fresh batch of 128 noise vectors.

            valid_y = smooth_positive_labels(batch_size)
            # Labels set to ~1.0 — telling GAN "these should be real".
            # Generator learns: produce images that score close to 1.

            flip_mask = np.random.rand(batch_size, 1) < 0.05
            if np.any(flip_mask):
                valid_y[flip_mask] = smooth_negative_labels(np.sum(flip_mask)).ravel()
            # LABEL FLIPPING — randomly flips 5% of labels from real to fake.
            # Another stabilization trick. Prevents generator from exploiting
            # any consistent pattern it finds in the discriminator.

            g_loss = gan.train_on_batch(noise, valid_y)
            # Trains the full GAN pipeline but only generator weights update
            # (discriminator is frozen). Generator learns to fool discriminator.

            if batch % 25 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | Batch {batch}/{batches_per_epoch} | "
                    f"D Loss: {d_loss:.4f} (acc {d_acc:.3f}) | G Loss: {g_loss:.4f}"
                )
                # Print every 25 batches to monitor without flooding console.
                # What to watch:
                #   D acc  → ideally 0.5-0.8. At 1.0 = discriminator too powerful.
                #   D loss → should stay moderate, not collapse to 0.
                #   G loss → should gradually decrease as generator improves.


# ── Run Training ───────────────────────────────────────────────────────────────
train(epochs=15, batch_size=64, noise_dim=100, instance_noise_std=0.05)
# epochs=15 — enough to see meaningful results without taking too long on laptop.
# Full training loop:
#   30,000 images ÷ 64 batch = ~468 batches per epoch
#   468 batches × 15 epochs = ~7,020 total training steps
#   Each step = discriminator update + generator update


# ══════════════════════════════════════════════════════════════════════════════
# GENERATE AND DISPLAY IMAGES
# ──────────────────────────────────────────────────────────────────────────────
# The PAYOFF — generate brand new digit images from pure random noise.
# Encoder is not involved at all (that was VAE).
# Here: random noise → Generator → new image. That's it.
# ══════════════════════════════════════════════════════════════════════════════

random_noise = np.random.normal(0, 1, [100, 100])
# 100 random noise vectors, each 100 numbers long. Shape = (100, 100).
# First 100  = number of images to generate
# Second 100 = noise_dim (what generator was built to expect)

generated_images = generator.predict(random_noise)
# Feed all 100 noise vectors through generator at once.
# Output shape = (100, 28, 28, 1) — 100 fake images.
# predict = just generate, no training happening here.

# ── Plot 10×10 grid of 100 generated images ────────────────────────────────────
plt.figure(figsize=(10, 10))

for i in range(generated_images.shape[0]):
    plt.subplot(10, 10, i + 1)                          # 10×10 grid = 100 slots
    plt.imshow(generated_images[i, :, :, 0], cmap='gray')
    # generated_images[i, :, :, 0] indexing:
    #   i    = which image (0 to 99)
    #   :, : = all rows and columns (full 28×28)
    #   0    = first (only) channel — drops channel dim for plotting
    plt.axis('off')                                      # hide axis numbers

plt.suptitle('GAN Generated Digits', fontsize=14)
plt.tight_layout()
plt.show()
# Output = 10×10 grid of 100 brand new digit images invented from pure noise.
# Should look SHARPER than VAE output because:
#   VAE optimizes for average → blurry
#   GAN optimizes to fool discriminator → sharper, more realistic
#
# Full recap:
#   Training:   Real image ←→ Discriminator ←→ Generator ← noise
#               (counterfeiter vs detective, both improving)
#   Generation: Random noise → Generator → Brand new image