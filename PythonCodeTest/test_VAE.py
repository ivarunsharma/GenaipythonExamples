import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

learning_rate = 0.001  # How big each learning step is. Too big = overshoots,
                       # too small = learns very slowly. 0.001 is a safe default.

num_steps  = 10        # Number of epochs (full passes through training data).
                       # Kept at 10 (instead of 100) for fast run on laptop.

batch_size = 64        # Instead of feeding all images at once (too slow) or
                       # one at a time (too noisy), we process 64 at a time.
                       # Powers of 2 (32, 64, 128) run efficiently on hardware.

latent_dim = 2         # Each image gets compressed to just 2 numbers.
                       # Why 2? Easy to visualize on an X-Y plot.
                       # Real projects use 64, 128, 256 for sharper results.

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING AND PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

# MNIST is a classic dataset of 70,000 handwritten digit images (0-9).
# Each image is 28x28 pixels in grayscale.
mnist = tf.keras.datasets.mnist

# mnist.load_data() returns two tuples:
#   (images, labels) for training -> 60k images
#   (images, labels) for test    -> 10k images
# The _ throws away the labels — we don't need them.
# VAEs are unsupervised: they learn from images alone, no labels required.
(x_train, _), (x_test, _) = mnist.load_data()

# Use only 5000 samples so it runs fast on a laptop.
# Remove the slice [:5000] if you want to train on the full dataset.
x_train = x_train[:5000]
x_test  = x_test[:1000]

# NORMALIZATION: Each pixel is originally 0-255 (brightness).
# Dividing by 255 maps all values to 0.0-1.0.
# Neural networks train faster and more stably with small numbers.
# The images are the same — just rescaled.
x_train = x_train / 255.0
x_test  = x_test  / 255.0

# ══════════════════════════════════════════════════════════════════════════════
# WHAT IS A VAE?
# ══════════════════════════════════════════════════════════════════════════════
#
# A Variational Autoencoder (VAE) has two parts:
#
#   ENCODER: Compresses a 28x28 image down to just 2 numbers (latent space).
#            But instead of exact numbers, it outputs a RANGE:
#              - z_mean    -> the center of the range
#              - z_log_var -> how wide the range is (variance)
#
#   DECODER: Takes a point from that range and reconstructs a 28x28 image.
#
# The "Variational" part means we sample a random point within the range
# instead of using a fixed point. This randomness is what allows the VAE
# to GENERATE new images — not just reconstruct seen ones.
#
# Full pipeline:
#   Image -> Encoder -> (z_mean, z_log_var) -> Sample point -> Decoder -> Reconstructed Image
#
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# ENCODER
# Compresses image (28x28 = 784 numbers) -> 2 numbers
# ══════════════════════════════════════════════════════════════════════════════

# Entry point of the encoder. Declares the expected input shape.
# Nothing is processed yet — just defining what comes in.
encoder_inputs = tf.keras.Input(shape=(28, 28))

# A 28x28 image is a 2D grid. Neural networks need a flat 1D list.
# Flatten unrolls it: 28x28 grid -> 784 numbers in a single row.
# Same data, just reshaped.
x = tf.keras.layers.Flatten()(encoder_inputs)

# Dense layer: every one of the 784 numbers connects to 128 neurons.
# These neurons learn patterns (edges, curves, shapes in digits).
# 'relu' activation: kills negative values (sets to 0), keeps positives.
# This adds non-linearity so the network learns complex patterns.
# Note: reduced from 512 -> 128 for faster training on laptop.
x = tf.keras.layers.Dense(128, activation='relu')(x)

# Two separate Dense layers, both taking the same 128 numbers as input,
# both outputting 2 numbers (latent_dim=2).
# They look identical in code but have SEPARATE internal weights.
# During training they learn different things because the loss function
# holds each responsible for a different job:

# z_mean: the CENTER of the latent range for this image.
z_mean    = tf.keras.layers.Dense(latent_dim)(x)

# z_log_var: the SPREAD/WIDTH of the latent range for this image.
# Uses log of variance (instead of variance directly) for training stability.
z_log_var = tf.keras.layers.Dense(latent_dim)(x)

# ══════════════════════════════════════════════════════════════════════════════
# SAMPLING LAYER — The "Variational" magic
# ══════════════════════════════════════════════════════════════════════════════
#
# The encoder gave us a range (mean + variance).
# Now we need to PICK A RANDOM POINT within that range.
# That sampled point is what gets passed to the decoder.
#
# Think of it like a dartboard:
#   z_mean    = the bullseye (center)
#   z_log_var = the size of the board (how spread out throws can be)
#   epsilon   = the random throw
#   result    = where the dart actually lands
#
# Formula: z = z_mean + exp(0.5 * z_log_var) * epsilon
#   exp(0.5 * z_log_var) -> converts log_variance back to standard deviation
#   * epsilon            -> scale random noise by that standard deviation
#   + z_mean             -> shift to be centered around the mean
#
# Every time this runs, you get a slightly different point —
# that's what makes generation possible later.
#
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs

        # Read dimensions to know how much random noise to generate.
        batch = tf.shape(z_mean)[0]  # number of images in this batch (64)
        dim   = tf.shape(z_mean)[1]  # latent dimensions (2)

        # Pure random noise — this is the randomness that makes VAE generative.
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        # Sample a point near the mean, scaled by the variance.
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Run the Sampling layer — feed z_mean and z_log_var, get back one sampled point.
encoder_outputs = Sampling()([z_mean, z_log_var])

# Package the encoder as an official Keras Model.
# Input : one 28x28 image
# Output: THREE things — mean, log_var, and sampled point.
# We return all three because the loss function needs mean and log_var
# to calculate how good the range is (KL loss — explained below).
encoder = tf.keras.Model(
    inputs=encoder_inputs,
    outputs=[z_mean, z_log_var, encoder_outputs]
)

# ══════════════════════════════════════════════════════════════════════════════
# DECODER
# Expands 2 numbers back -> reconstructed 28x28 image
# Exact reverse of the encoder.
# ══════════════════════════════════════════════════════════════════════════════

# Entry point of the decoder. Expects 2 numbers (the sampled latent point).
# The comma in (latent_dim,) means it's a 1D shape of size 2, not a 2D grid.
latent_inputs = tf.keras.Input(shape=(latent_dim,))

# Expand 2 numbers -> 128 neurons.
# The network learns to stretch 2 numbers into meaningful patterns
# that can eventually become a full image.
x = tf.keras.layers.Dense(128, activation='relu')(latent_inputs)

# Expand 128 -> 784 pixel values.
# 'sigmoid' outputs values between 0.0 and 1.0 — perfect for normalized pixels.
# (encoder used 'relu' for hidden layers, decoder uses 'sigmoid' for output)
x = tf.keras.layers.Dense(784, activation='sigmoid')(x)

# Reshape flat 784 numbers back into a 28x28 image grid.
decoder_outputs = tf.keras.layers.Reshape((28, 28))(x)

# Package the decoder as an official Keras Model.
# Input : 2 numbers (latent point)
# Output: reconstructed 28x28 image
decoder = tf.keras.Model(inputs=latent_inputs, outputs=decoder_outputs)

# ══════════════════════════════════════════════════════════════════════════════
# VAE MODEL WITH LOSS
# ══════════════════════════════════════════════════════════════════════════════
#
# The VAE has TWO losses that work together:
#
#   1. RECONSTRUCTION LOSS: How different is the output from the input?
#      -> Forces the decoder to rebuild images accurately.
#
#   2. KL LOSS (Kullback-Leibler): Is the latent space well organized?
#      -> Forces the encoder to keep ranges close to mean~0, variance~1.
#      -> Without this, the encoder could cheat — output wild ranges that
#         reconstruct well but generate garbage for new random points.
#
#   Total Loss = Reconstruction Loss + KL Loss
#
# ══════════════════════════════════════════════════════════════════════════════

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        # Store encoder and decoder inside the VAE so they can be used later.
        self.encoder = encoder
        self.decoder = decoder

    def compute_loss(self, x):
        # Step 1: Run the image through encoder -> get mean, log_var, sampled point.
        z_mean, z_log_var, z = self.encoder(x)

        # Step 2: Run the sampled point through decoder -> get reconstructed image.
        reconstructed = self.decoder(z)

        # LOSS 1 — RECONSTRUCTION LOSS
        # Measures pixel-by-pixel how different the reconstruction is from original.
        # binary_crossentropy works well here since pixels are between 0 and 1.
        # Multiplied by 28*28 to scale up to pixel level (otherwise averaged too small).
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(x, reconstructed)
        )
        reconstruction_loss *= 28 * 28

        # LOSS 2 — KL LOSS
        # Measures how far the learned distribution (z_mean, z_log_var) is
        # from a standard normal distribution (mean=0, variance=1).
        # Keeps the latent space tidy and organized so random sampling works.
        # The -0.5 and the formula come from the KL divergence math.
        kl_loss  = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss  = tf.reduce_mean(kl_loss) * -0.5

        # Total loss — minimize both at the same time.
        return reconstruction_loss + kl_loss

    def train_step(self, data):
        # MNIST returns (images, labels) — we only need images, grab index 0.
        if isinstance(data, tuple):
            data = data[0]

        # GradientTape watches all the math and records it.
        # This is how TensorFlow knows which weights to adjust and by how much.
        with tf.GradientTape() as tape:
            loss = self.compute_loss(data)

        # grads = how much each weight contributed to the loss.
        grads = tape.gradient(loss, self.trainable_variables)

        # Nudge every weight slightly in the direction that reduces loss.
        # This is BACKPROPAGATION — the fundamental learning mechanism.
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {'loss': loss}

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

vae = VAE(encoder, decoder)

# Adam optimizer: a smart version of gradient descent that adjusts
# the learning rate automatically. Most popular choice in deep learning.
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

# vae.fit() runs the training loop automatically.
# Notice: x_train is passed TWICE (as both input and output).
# This is unique to autoencoders — the goal is to reconstruct the input itself.
# Unlike classification where: input=image, output=label
# Here:                         input=image, output=same image
#
# Each epoch = one full pass through all 5000 training images.
# Each epoch has 5000 / 64 = ~78 steps internally.
vae.fit(x_train, x_train, epochs=num_steps, batch_size=batch_size)

# ══════════════════════════════════════════════════════════════════════════════
# IMAGE GENERATION
# ══════════════════════════════════════════════════════════════════════════════
#
# This is the payoff. We generate brand new digit images that never existed.
#
# Key insight: the ENCODER is not used here at all.
# We skip straight to the decoder with random latent points.
# This works because KL loss during training forced the latent space to be
# smooth and organized — any random point maps to a meaningful digit-like image.
#
# Training pipeline:   Image -> Encoder -> Sample -> Decoder -> Reconstructed Image
# Generation pipeline:          Random point       -> Decoder -> Brand New Image
#
def generate_images(model, n_images=16):
    # Generate n_images random points in latent space.
    # Shape = (16, 2) — sixteen pairs of random numbers.
    # No real image needed — pure random input.
    random_latent_vectors = tf.random.normal(shape=(n_images, latent_dim))

    # Feed random points directly into the decoder.
    # Each 2-number point -> one 28x28 generated image.
    generated_images = model.decoder(random_latent_vectors)

    # Convert TensorFlow tensor -> numpy array.
    # Needed because matplotlib works with numpy, not TensorFlow tensors.
    generated_images = generated_images.numpy()

    # Plot all generated images in a grid (4 images per row).
    n_rows = int(np.ceil(n_images / 4))
    plt.figure(figsize=(8, 8))
    for i in range(n_images):
        plt.subplot(n_rows, 4, i + 1)
        plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')  # Hide axis numbers for cleaner look.

    plt.suptitle('VAE Generated Digits — Brand new, never seen before', fontsize=12)
    plt.tight_layout()
    plt.show()

# Generate and display 16 brand new handwritten digit images.
# They will look slightly blurry because latent_dim=2 is very compressed.
# Increase latent_dim to 32 or 64 for sharper results (but slower training).
generate_images(vae, 16)