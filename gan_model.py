# -*- coding: utf-8 -*-

from tensorflow.keras.models import Model

class GAN(Model):
    """
    A custom implementation of a Generative Adversarial Network (GAN) that extends
    the Keras Model class.

    Attributes:
        generator (Model): The generator model responsible for generating fake images.
        discriminator (Model): The discriminator model responsible for distinguishing
        between real and fake images.
    """

    def __init__(self, generator, discriminator, *args, **kwargs):
        """
        Initializes the GAN with generator and discriminator models.

        Args:
            generator (Model): The generator model.
            discriminator (Model): The discriminator model.
            *args: Additional positional arguments for the base Model class.
            **kwargs: Additional keyword arguments for the base Model class.
        """
        super().__init__(*args, **kwargs)
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):
        """
        Configures the GAN for training by setting up optimizers and loss functions.

        Args:
            g_opt: Optimizer for the generator.
            d_opt: Optimizer for the discriminator.
            g_loss: Loss function for the generator.
            d_loss: Loss function for the discriminator.
            *args: Additional positional arguments for the base Model class.
            **kwargs: Additional keyword arguments for the base Model class.
        """
        super().compile(*args, **kwargs)
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss

    def train_step(self, batch):
        """
        Executes one training step for both the generator and the discriminator.

        Args:
            batch: The batch of real images used for training.

        Returns:
            dict: A dictionary containing the discriminator loss (`d_loss`)
            and generator loss (`g_loss`).
        """
        # Real images from the current batch
        real_images = batch

        # Generate fake images using the generator
        fake_images = self.generator(tf.random.normal((128, 128, 1)), training=False)

        # Train the discriminator
        with tf.GradientTape() as d_tape:
            # Discriminator outputs for real and fake images
            yhat_real = self.discriminator(real_images, training=True)
            yhat_fake = self.discriminator(fake_images, training=True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)

            # Labels for real (0) and fake (1) images
            y_realfake = tf.concat(
                [tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0
            )

            # Add random noise to real and fake labels for improved training stability
            noise_real = 0.15 * tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15 * tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real, noise_fake], axis=0)

            # Compute the discriminator loss
            total_d_loss = self.d_loss(y_realfake, yhat_realfake)

        # Apply gradients to the discriminator's trainable variables
        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as g_tape:
            # Generate new fake images
            gen_images = self.generator(tf.random.normal((128, 128, 1)), training=True)

            # Discriminator predictions for generated images
            predicted_labels = self.discriminator(gen_images, training=False)

            # Compute the generator loss (aims to "fool" the discriminator)
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels)

        # Apply gradients to the generator's trainable variables
        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        # Return a dictionary of the losses
        return {"d_loss": total_d_loss, "g_loss": total_g_loss}