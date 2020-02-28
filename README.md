# Classifying Images using Regional CNN

This program classifies images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) using a Convolutional Neural Network (CNN). The images are preprocessed and normalized then used to train a CNN consisting of convolutional, max pooling, dropout, fully connected, and output layers.


In-depth analysis and examples can be found in [dlnd_image_classification.ipynb](https://github.com/BananuhBeatDown/image_recognition/blob/master/dlnd_image_classification.ipynb).

## Specifications

- Python 3.7
- [TensorFlow 2.0](https://www.tensorflow.org/install/?nav=true)
- [tqdm](https://github.com/noamraph/tqdm)
- [numpy](https://numpy.org/)
- [sklearn](https://scikit-learn.org/)

## Dataset

The CIFAR-10 dataset consists of 5 batches, named `data_batch_1`, `data_batch_2`, etc.. Each batch contains the labels and images that are one of the following:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Parameters

#### image_recognition.py

You can view different example of the CIFAR-10 dataset by changing the values of the `batch_id` and the `sample_id`:

- `batch_id` - id for a batch (1-5)
- `sample_id` - id for a image and label pair in the batch

```python
batch_id = 1
sample_id = 10
helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)
```
<img src="https://user-images.githubusercontent.com/10539813/27656181-52f23142-5c48-11e7-8f39-7a204c6d11eb.png" width="256">


#### train_image_recognition.py

You can experiment with the CNN by altering:
- `depth` - Alter the depths of the CNN layers using common memory sizes
    + 64
    + 128
    + 256
    + ...
- `epochs` - number of training iterations
- `batch_size` - the number of training examples utilized in one iteration. Make sure to change this according to your system's memory and GPU capacity
- `keep_probability` - the probability of keeping node using dropout

## Example Output

**Command Line**   

`python image_recognition.py`   

**You must press [enter] to continue after example image appears.*  

<img src="https://user-images.githubusercontent.com/29889429/75466272-f4d57480-59af-11ea-91ec-f1a3824450ed.png">

`python train_image_recognition.py`  

<img src="https://user-images.githubusercontent.com/29889429/75466362-13d40680-59b0-11ea-99fb-f89ee07c7f3c.png">

## License
The image_classification program is a public domain work, dedicated to using [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).