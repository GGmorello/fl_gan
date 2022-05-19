# Install dependencies if not already installed
#!pip install tensorflow==2.3.1

# Create a federation
from openfl.interface.interactive_api.federation import Federation
import time
import tensorflow as tf
# from layers import train_acc_metric, val_acc_metric, loss_fn
from openfl.interface.interactive_api.experiment import TaskInterface, DataInterface, ModelInterface, FLExperiment
from layers import create_model, define_discriminator, define_generator, optimizer, cross_entropy, seed
from layers import generator_loss, discriminator_loss, generator_optimizer, discriminator_optimizer, generate_and_save_images
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import plot_model


client_id = 'api'
cert_dir = 'cert'
director_node_fqdn = 'localhost'
director_port = 50051

federation = Federation(
    client_id=client_id,
    director_node_fqdn=director_node_fqdn,
    director_port=director_port, 
    tls=False
)


shard_registry = federation.get_shard_registry()

# First, request a dummy_shard_desc that holds information about the federated dataset 
dummy_shard_desc = federation.get_dummy_shard_descriptor(size=10)
dummy_shard_dataset = dummy_shard_desc.get_dataset('train')
sample, target = dummy_shard_dataset[0]
f"Sample shape: {sample.shape}, target shape: {target.shape}"


framework_adapter = 'openfl.plugins.frameworks_adapters.keras_adapter.FrameworkAdapterPlugin'
gen_model = define_generator()
dis_model = define_discriminator()
gan_model = create_model(gen_model, dis_model)

MI = ModelInterface(model=gan_model, optimizer=optimizer, framework_plugin=framework_adapter)


class DataGenerator(Sequence):

    def __init__(self, shard_descriptor, batch_size):
        self.shard_descriptor = shard_descriptor
        self.batch_size = batch_size
        self.indices = np.arange(len(shard_descriptor))
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        index = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]

        X = self.shard_descriptor[batch]
        return X

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


class MnistFedDataset(DataInterface):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def shard_descriptor(self):
        return self._shard_descriptor

    @shard_descriptor.setter
    def shard_descriptor(self, shard_descriptor):
        """
        Describe per-collaborator procedures or sharding.

        This method will be called during a collaborator initialization.
        Local shard_descriptor will be set by Envoy.
        """
        self._shard_descriptor = shard_descriptor
        self.train_set = shard_descriptor.get_dataset('data')

    def __getitem__(self, index):
        return self.shard_descriptor[index]

    def __len__(self):
        return len(self.shard_descriptor)

    def get_train_loader(self):
        """
        Output of this method will be provided to tasks with optimizer in contract
        """
        if self.kwargs['train_bs']:
            batch_size = self.kwargs['train_bs']
        else:
            batch_size = 32
        return DataGenerator(self.train_set, batch_size=batch_size)

    def get_valid_loader(self):
        """
        Output of this method will be provided to tasks without optimizer in contract
        """
        if self.kwargs['valid_bs']:
            batch_size = self.kwargs['valid_bs']
        else:
            batch_size = 32

        return DataGenerator(self.train_set, batch_size=batch_size)

    def get_train_data_size(self):
        """
        Information for aggregation
        """
        
        return len(self.train_set)


    def get_valid_data_size(self):
        """
        Information for aggregation
        """
        return len(self.train_set)

fed_dataset = MnistFedDataset(train_bs=256, valid_bs=512)


TI = TaskInterface()


@TI.register_fl_task(model='model', data_loader='train_dataset', device='device', optimizer='optimizer')
def train(model, train_dataset, device, optimizer, BATCH_SIZE=256, noise_dim=100):
    generator = model.layers[0]
    discriminator = model.layers[1]
    for i, images in enumerate(train_dataset):
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
        # summarize loss on this batch
        print('%d d=%.3f, g=%.3f' % (i+1, disc_loss, gen_loss))
    return {'gen_loss': gen_loss, 'disc_loss': disc_loss}


@TI.register_fl_task(model='model', data_loader='train_dataset', device='device')
def validate(model, train_dataset, device, seed=seed):
    generate_and_save_images(model.layers[0], seed)
    return {' ': 0}

# create an experimnet in federation
experiment_name = 'mnist_experiment'
fl_experiment = FLExperiment(federation=federation, experiment_name=experiment_name)

# The following command zips the workspace and python requirements to be transfered to collaborator nodes
fl_experiment.start(model_provider=MI, 
                   task_keeper=TI,
                   data_loader=fed_dataset,
                   rounds_to_train=50,
                   opt_treatment='CONTINUE_GLOBAL')

fl_experiment.stream_metrics()

a = fl_experiment.get_last_model().layers[0]

print(type(a))

plot_model(a, 'best_model.png', show_shapes=True)

generate_and_save_images(a, seed)
