# Semantic Segmentation Fully Convolutional Neural Network
Semantic Segmentation - It label the pixels of a road in images using a Fully Convolutional Network (FCN).

# Index

- [Neural network archtecture](#Neural-network-archtecture)
- [Some results](#Some-results)
- [Setup](#Setup)
    - [Frameworks and Packages](#Frameworks-and-Packages)
    - [Dataset](#Dataset)
- [Start](#Start)
    - [Usage](#Usage)
    - [google colabotory - GPU 4free](#Running-at-google-colabotory---GPU-4free)

## Neural network archtecture
The project start loading a vgg model so implements an FCN - Fully Convolutional Network 

!['./images/vgg-FCN.png'](./images/vgg-FCN.png)

More details about how VGG16 is used in this archtechture to converte it to a fully convolutional network, you can read ['this paper']('https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf').

## Some results
 ![](./images/1.png)

 ![](./images/2.png)

 ![](./images/3.png)


## Setup
#### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

#### Dataset
The project will download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) dataset from [here](http://www.cvlibs.net/download.php?file=data_road.zip). It will extract the files in `data` folder.  This will create the folder `data_road` with all the training a test images.

## Start

#### Usage
Run the following command to run the project:
```
python main.py
```
it will run the Neural network with default parameter values, to know more, type `python main.py -h`

```
Usage: main.py [options]

Options:
  -h, --help            show this help message and exit
  -i GLOB_TRAINIG_IMAGES_PATH, --glob_trainig_images_path=GLOB_TRAINIG_IMAGES_PATH
                        Path where is yours images to train the model. eg:
                        ./data/data_road/training/image_2/*.png'
  -l GLOB_LABELS_TRAINIG_IMAGE_PATH, --glob_labels_trainig_image_path=GLOB_LABELS_TRAINIG_IMAGE_PATH
                        Path where is yours label images to train the model.
                        eg: ./data/data_road/training/gt_image_2/*_road_*.png
  -r LEARN_RATE, --learn_rate=LEARN_RATE
                        The model learn rate | Default=9e-5
  -n NUM_CLASSES, --num_classes=NUM_CLASSES
                        Number of classes in your dataset | Default value = 2
  -e EPOCHS, --epochs=EPOCHS
                        Number of epochs that FCN will train | Default=25
  -b BATCH_SIZE, --batch_size=BATCH_SIZE
                        Number of batch size for each epoch. | Default=4
  -t DATA_PATH, --data_path=DATA_PATH
                        Training data path. | Default='data_road/training'
  -p LOG_PATH, --log_path=LOG_PATH
                        Path to save the tensorflow logs to TensorBoard |
                        Default='.'
  -v VGG_DIR, --vgg_dir=VGG_DIR
                        Path to dowloand vgg pre trained weigths. |
                        Default='./data/vgg'
  -g GRAPH_VISUALIZE, --graph_visualize=GRAPH_VISUALIZE
                        create a graph image of the FCN archtecture. |
                        Default=False
```

#### Running at google colabotory - GPU 4free
Open this [link](https://colab.research.google.com/drive/1bwYKqgZIKlMRuTqD0HdU3DeHJzH9xwe4)
