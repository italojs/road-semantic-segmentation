import re
import time
import shutil
import random
import zipfile
import os.path
import scipy.misc
import numpy as np
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from optparse import OptionParser
from urllib.request import urlretrieve


class DLProgress(tqdm):
  last_block = 0

  def hook(self, block_num=1, block_size=1, total_size=None):
    self.total = total_size
    self.update((block_num - self.last_block) * block_size)
    self.last_block = block_num


def maybe_download_pretrained_vgg(vgg_path):
  """
  Download and extract pretrained vgg model if it doesn't exist
  :param data_dir: Directory to download the model to
  """
  vgg_filename = 'vgg.zip'
  vgg_files = [
      os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
      os.path.join(vgg_path, 'variables/variables.index'),
      os.path.join(vgg_path, 'saved_model.pb')]

  missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
  if missing_vgg_files:
    # Clean vgg dir
    if os.path.exists(vgg_path):
      shutil.rmtree(vgg_path)
    os.makedirs(vgg_path)

    # Download vgg
    print('Downloading pre-trained vgg model...')
    with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
      urlretrieve(
          'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
          os.path.join(vgg_path, vgg_filename),
          pbar.hook)

    # Extract vgg
    print('Extracting model...')
    zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
    zip_ref.extractall(os.path.join(vgg_path,'..'))
    zip_ref.close()

    # Remove zip file to save space
    os.remove(os.path.join(vgg_path, vgg_filename))

def get_args():
    parser = OptionParser()

    parser.add_option("-i", "--glob_trainig_images_path", dest="glob_trainig_images_path", 
      help="Path where is yours images to train the model. eg: ./data/data_road/training/image_2/*.png'")
    # TODO: verify if the name of labels_images is really labels_images
    parser.add_option("-l", "--glob_labels_trainig_image_path", dest="glob_labels_trainig_image_path", 
      help="Path where is yours label images to train the model. eg: ./data/data_road/training/gt_image_2/*_road_*.png")
    parser.add_option("-r", "--learn_rate", dest="learn_rate", help="The model learn rate | Default=9e-5")
    parser.add_option("-n", "--num_classes", dest="num_classes", help="Number of classes in your dataset | Default value = 2")
    parser.add_option("-e", "--epochs", dest="epochs", help="Number of epochs that FCN will train | Default=25")
    parser.add_option("-b", "--batch_size", dest="batch_size", help="Number of batch size for each epoch. | Default=4")
    parser.add_option("-t", "--data_path", dest="data_path", help="Training data path. | Default='data_road/training'")
    parser.add_option("-p", "--log_path", dest="log_path", help="Path to save the tensorflow logs to TensorBoard | Default='.'")
    parser.add_option("-v", "--vgg_dir", dest="vgg_dir", help="Path to dowloand vgg pre trained weigths. | Default='./data/vgg'")
    parser.add_option("-g", "--graph_visualize", dest="graph_visualize", help="create a graph image of the FCN archtecture. | Default=False")
    
    (options, args) = parser.parse_args()

    log_path = options.log_path if options.log_path is None else '.'
    epochs = int(options.epochs) if options.epochs is not None else 25
    batch_size = options.batch_size if options.batch_size is not None else 4
    vgg_dir = options.vgg_dir if options.vgg_dir is not None else './data/vgg'
    learn_rate = float(options.learn_rate) if options.learn_rate is not None else 9e-5
    data_path = options.data_path if options.data_path is not None else './data/data_road'
    graph_visualize = options.graph_visualize if options.graph_visualize is not None else False
    num_classes = options.num_classes if options.num_classes is None else 2
    glob_trainig_images_path = options.glob_trainig_images_path if options.glob_trainig_images_path \
     is None else './data/data_road/training/image_2/*.png'
    glob_labels_trainig_image_path = options.glob_labels_trainig_image_path if options.glob_labels_trainig_image_path \
     is None else './data/data_road/training/gt_image_2/*_road_*.png'
     
    # if not options.num_classes:
    #   raise  Exception('-n/--num_classes is required')
    # if not options.glob_trainig_images_path:
    #   raise  Exception('-i/--glob_trainig_images_path is required')
    # if not options.glob_labels_trainig_image_path:
    #   raise  Exception('-l/--glob_labels_trainig_image_path is required')

    return (int(options.num_classes),
      epochs, 
      batch_size, 
      vgg_dir, 
      learn_rate,
      log_path, 
      data_path, 
      graph_visualize, 
      options.glob_trainig_images_path, 
      options.glob_labels_trainig_image_path)

def gen_batch_function(glob_trainig_images_path, glob_labels_trainig_image_path, image_shape):
  """
  Generate function to create batches of training data
  :param data_folder: Path to folder that contains all the datasets
  :param image_shape: Tuple - Shape of image
  :return:
  """
  def get_batches_fn(batch_size):
    """
    Create batches of training data
    :param batch_size: Batch Size
    :return: Batches of training data
    """
  
    image_paths = glob(glob_trainig_images_path)
    # TODO: verify a generic way to construct this batch dataset
    label_paths = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(glob_labels_trainig_image_path)}
    background_color = np.array([255, 0, 0])
    random.shuffle(image_paths)
    for batch_i in range(0, len(image_paths), batch_size):
      images = []
      gt_images = []
      for image_file in image_paths[batch_i:batch_i+batch_size]:
        gt_image_file = label_paths[os.path.basename(image_file)]

        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

        gt_bg = np.all(gt_image == background_color, axis=2)
        gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
        gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

        images.append(image)
        gt_images.append(gt_image)

      yield np.array(images), np.array(gt_images)
  return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
  """
  Generate test output using the test images
  :param sess: TF session
  :param logits: TF Tensor for the logits
  :param keep_prob: TF Placeholder for the dropout keep robability
  :param image_pl: TF Placeholder for the image placeholder
  :param data_folder: Path to the folder that contains the datasets
  :param image_shape: Tuple - Shape of image
  :return: Output for for each test image
  """
  for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
    image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, image_pl: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)

    yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
  # Make folder for current run
  output_dir = os.path.join(runs_dir, str(int(time.time())))
  if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
  os.makedirs(output_dir)

  # Run NN on test images and save them to HD
  print('Training Finished. Saving test images to: {}'.format(output_dir))
  image_outputs = gen_test_output(
      sess, logits, keep_prob, input_image, os.path.join(data_dir, 'testing'), image_shape)
  for name, image in image_outputs:
    scipy.misc.imsave(os.path.join(output_dir, name), image)
