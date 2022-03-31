
# MNIST Generative Adversial Network in OpenFL

###

This workspace trains a Generative Adversial Network using Federated Learning, models are a variant of keras models 
found [here](https://www.tensorflow.org/tutorials/generative/dcgan), federated learning is achieved using OpenFL's interactive API.

![mnist digits](http://i.ytimg.com/vi/0QI3xgXuB-Q/hqdefault.jpg "MNIST Digits")


### 1. About dataset
It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9. More info at [wiki](https://en.wikipedia.org/wiki/MNIST_database).

### 2. About models
Generator and discriminator are defined in
[layers.py](./workspace/layers.py) file.


### 3. How to run this tutorial (without TLS and locally as a simulation):

1. Run director:
```sh
cd director_folder
./start_director.sh
```

2. Run envoy:
```sh
cd envoy_1
./start_envoy.sh env_one envoy_config_one.yaml
```

Optional: start second envoy:
```sh
cd envoy_2
./start_envoy.sh env_two envoy_config_two.yaml
```

3. Run `Tensorflow_gan_MNIST.py` jupyter notebook:
```sh
cd workspace
python Tensorflow_gan_MNIST.py
```


## Acknowledgements

* https://www.tensorflow.org/tutorials/generative/dcgan
* https://github.com/intel/openfl