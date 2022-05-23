import flwr as fl
import tensorflow as tf
import argparse
from keras.datasets.mnist import load_data
from numpy import expand_dims
from layers import create_model, generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss, seed, generate_and_save_images

BATCH_SIZE = 256
noise_dim = 100
def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, choices=range(0, 2), required=True)
    args = parser.parse_args()

    x_train, x_test = load_partition(args.partition)
    x_train = expand_dims(x_train, axis=-1)
    x_train = x_train.astype('float32')
    x_train = x_train / 255.0

    x_test = expand_dims(x_test, axis=-1)
    x_test = x_test.astype('float32')
    x_test = x_test / 255.0

    model = create_model()

    client = GanClient(model, x_train, x_test)
    fl.client.start_numpy_client("[::]:8080", client=client)


def load_partition(idx: int):
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = expand_dims(x_train, axis=-1)
    x_train = x_train.astype('float32')
    x_train = x_train / 255.0

    x_test = expand_dims(x_test, axis=-1)
    x_test = x_test.astype('float32')
    x_test = x_test / 255.0
    return x_train[idx * 30000 : (idx + 1) * 30000], x_test[idx * 5000 : (idx + 1) * 5000]

class GanClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, x_test):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        x_train_ds = tf.data.Dataset.from_tensor_slices((x_train))
        self.x_train_ds = x_train_ds.batch(BATCH_SIZE)  # batch_size can be 1
        x_test_ds = tf.data.Dataset.from_tensor_slices((x_test))
        self.x_test_ds = x_test_ds.batch(BATCH_SIZE)  # batch_size can be 1

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        generator = self.model.layers[0]
        discriminator = self.model.layers[1]
        for i, images in enumerate(self.x_train_ds):
            noise = tf.random.normal([BATCH_SIZE, noise_dim])
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(noise, training=True)

                real_output = discriminator(images, training=True)
                fake_output = discriminator(generated_images, training=True)

                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
            print('%d d=%.3f, g=%.3f' % (i + 1, disc_loss, gen_loss))
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        generator = self.model.layers[0]
        discriminator = self.model.layers[1]
        noise = tf.random.normal([BATCH_SIZE, noise_dim])
        generated_images = generator(noise, training=True)
        real_output = discriminator(self.x_test, training=True)
        fake_output = discriminator(generated_images, training=True)
        loss = discriminator_loss(real_output, fake_output)
        generate_and_save_images(self.model.layers[0], seed)
        np = float(loss.numpy())
        return np, len(self.x_test), {}


if __name__ == "__main__":
    main()