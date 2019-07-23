import helper
import os
import warnings
import tensorflow as tf
import tests as tests
import scipy.misc

import progressbar
from tensorflow.python.util import compat
from distutils.version import LooseVersion
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2

image_shape = (160, 576)

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name('image_input:0')
    keep = graph.get_tensor_by_name('keep_prob:0')
    layer3 = graph.get_tensor_by_name('layer3_out:0')
    layer4 = graph.get_tensor_by_name('layer4_out:0')
    layer7 = graph.get_tensor_by_name('layer7_out:0')

    return w1, keep, layer3, layer4, layer7

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # KK Hyperparameters: Regularizer, Initializer, etc.
    l2_value = 1e-3
    kernel_reg = tf.contrib.layers.l2_regularizer(l2_value)
    stddev = 1e-3
    kernel_init = tf.random_normal_initializer(stddev=stddev)

    # KK 1x1 convolution to preserve spatial information
    conv_1x1_7 = tf.layers.conv2d(vgg_layer7_out, num_classes,
                                  kernel_size=1,
                                  strides=(1, 1),
                                  padding='same',
                                  kernel_regularizer=kernel_reg,
                                  kernel_initializer=kernel_init)

    # KK Print the shape of the 1x1
    tf.Print(conv_1x1_7, [tf.shape(conv_1x1_7)[1:3]])

    # KK Upsample by 2x so we can add it with layer4 in the skip layer to follow
    conv7_2x = tf.layers.conv2d_transpose(conv_1x1_7, num_classes,
                                          kernel_size=4,
                                          strides=(2, 2),
                                          padding='same',
                                          kernel_regularizer=kernel_reg,
                                          kernel_initializer=kernel_init)

    # KK Print the shape of the upsample
    print( '\n\nUpsampled layer 7 = ', tf.Print(conv7_2x, [tf.shape(conv7_2x)[1:3]]) )

    # KK 1x1 convolution to preserve spatial information
    conv_1x1_4 = tf.layers.conv2d(vgg_layer4_out, num_classes,
                                  kernel_size=1,
                                  strides=(1, 1),
                                  padding='same',
                                  kernel_regularizer=kernel_reg,
                                  kernel_initializer=kernel_init)

    # KK Add the layer4 with the upsampled 1x1 convolution as a skip layer
    skip_4_to_7 = tf.add(conv7_2x, conv_1x1_4)

    # KK Upsample the combined layer4 and 1x1 by 2x
    upsample2x_skip_4_to_7 = tf.layers.conv2d_transpose(skip_4_to_7, num_classes,
                                                        kernel_size=4,
                                                        strides=(2, 2),
                                                        padding='same',
                                                        kernel_regularizer=kernel_reg,
                                                        kernel_initializer=kernel_init)

    # KK Print the resulting shape
    print( '\n\nUpsampled 4 and 7 = ', tf.Print(upsample2x_skip_4_to_7, [tf.shape(upsample2x_skip_4_to_7)[1:3]]))

    # KK 1x1 convolution to preserve spatial information
    conv_1x1_3 = tf.layers.conv2d(vgg_layer3_out, num_classes,
                                  kernel_size=1,
                                  strides=(1, 1),
                                  padding='same',
                                  kernel_regularizer=kernel_reg,
                                  kernel_initializer=kernel_init)

    # KK Add layer 3 with the upsampled skip1 layer
    skip_3 = tf.add(upsample2x_skip_4_to_7, conv_1x1_3)

    # KK Upsample by 8x to get to original image size
    output = tf.layers.conv2d_transpose(skip_3, num_classes,
                                        kernel_size=16,
                                        strides=(8, 8),
                                        padding='same',
                                        kernel_regularizer=kernel_reg,
                                        kernel_initializer=kernel_init)

    # KK Print the resulting shape which should be the original image size
    print('\n\nShape of output image = ', tf.Print(output, [tf.shape(output)[1:3]]))

    return output

def get_logits(last_layer, num_classes):
    return tf.reshape(last_layer, (-1, num_classes))

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function KK-DONE

    # KK Get the logits of the network
    logits = get_logits(nn_last_layer, num_classes)

    # KK Get the loss of the network
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    #KK Regularization loss collector....Don't really understand this
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 0.01  # Choose an appropriate one.
    loss = cross_entropy_loss + reg_constant * sum(reg_losses)

    # KK Minimize the loss using Adam Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, loss

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    # KK loop through epochs
    for epoch in range(epochs):
        print('##############################################################')
        print('........................Training Epoch # {}/{}...................'.format(epoch, epochs))
        print('##############################################################')

        bar = progressbar.ProgressBar()
        # KK loop through images and labels
        loss = None
        for image, label in bar(get_batches_fn(batch_size)):
            # Training
            feed_dict = {
                input_image: image,
                correct_label: label, 
                keep_prob: 0.5, 
                learning_rate: learn_rate
            }
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
        print('\nTraining Loss = {:.3f}'.format(loss))

    pass

#KK Visualize the VGG16 model from Udacity reviewer
def graph_visualize():
    with tf.Session() as sess:
        model_filename = os.path.join(vgg_dir, 'saved_model.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
            data = compat.as_bytes(f.read())
            sm = saved_model_pb2.SavedModel()
            sm.ParseFromString(data)
            g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)
    train_writer = tf.summary.FileWriter(log_dir)
    train_writer.add_graph(sess.graph)

def run():
    runs_dir = './runs'

    print("\n\nTesting for datatset presence......")
    tests.test_looking_for_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(vgg_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(
            glob_trainig_images_path, 
            glob_labels_trainig_image_path,
            image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function KK-DONE

        # TF placeholders
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_dir)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        # tfImShape = tf.get_variable("image_shape")
        # tfLogits = tf.get_variable("logits")
        # tfKeepProb = tf.get_variable("keep_prob") TEM NO TF

        print(100*'*')
        print(image_shape)
        #(160, 576)
        print(100*'*')
        print(logits)
        #Tensor("Reshape:0", shape=(?, 2), dtype=float32)
        print(100*'*')
        print(keep_prob)
        #Tensor("keep_prob:0", dtype=float32)
        print(100*'*')
        print(input_image)
        #Tensor("image_input:0", shape=(?, ?, ?, 3), dtype=float32)
        print(100*'*')

        init_op = tf.global_variables_initializer()

        sess.run(init_op)
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)


        folderToSaveModel = "model"

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        for i, var in enumerate(saver._var_list):
            print('Var {}: {}'.format(i, var))

        if not os.path.exists(folderToSaveModel):
            os.makedirs(path)

        pathSaveModel = os.path.join(folderToSaveModel, "model.ckpt")
        pathSaveModel = saver.save(sess, pathSaveModel)
        print("Model saved in path: {}".format(pathSaveModel))
        
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video

def all_is_ok():
    # Check TensorFlow Version
    assert LooseVersion(tf.__version__) >= LooseVersion(
        '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
    print('TensorFlow Version: {}'.format(tf.__version__))

    # Check for a GPU
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

    print("\n\nTesting load_vgg function......")
    tests.test_load_vgg(load_vgg, tf)
    
    print("\n\nTesting layers function......")
    tests.test_layers(layers)

    print("\n\nTesting optimize function......")
    tests.test_optimize(optimize)

    print("\n\nTesting train_nn function......")
    tests.test_train_nn(train_nn)

def predict_by_model():
    if path_data is False:
        exit("Path video not set, pass the properly argument")

    """
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param num_classes: Number of classes to classify
    :param input_image: TF Placeholder for input images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    """
    
    # Path to vgg model
    vgg_path = os.path.join('./data', 'vgg')

    #IF EXCEED GPU MEMORY, USE THE CONFIG BELOW
    useCPU = tf.ConfigProto(device_count={'GPU': 0})
    with tf.Session(config=useCPU) as sess:
        # Predict the logits
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits = get_logits(nn_last_layer, num_classes)

        # Restore the saved model
        saver = tf.train.Saver()
        saver.restore(sess, path_model)
        
        if pred_data_from == 'video':
            # Predict a video
            helper.predict_video(path_data, sess, image_shape, logits, keep_prob, input_image)
        elif pred_data_from == 'image':
            # Predict a image
            image = scipy.misc.imresize(scipy.misc.imread(path_data), image_shape)
            street_im = helper.predict(sess, image, input_image, keep_prob, logits, image_shape)
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            imagePath = os.path.join(current_dir, "image_predicted.png")
            
            scipy.misc.imsave(imagePath, street_im)
            print("Image save in {}".format(imagePath))
        elif pred_data_from == 'zed':
            helper.read_zed(sess, image_shape, logits, keep_prob, input_image)

if __name__ == '__main__':
    (pred_data_from,
     path_model,
     path_data,
     num_classes,
     epochs,
     batch_size, 
     vgg_dir, 
     learn_rate,
     log_dir, 
     data_dir, 
     graph_visualize,
     glob_trainig_images_path,
     glob_labels_trainig_image_path) = helper.get_args()

    if not path_model:
        all_is_ok()
        run()
    else:
        predict_by_model()

    if graph_visualize:
        print("\n\nConverting .pb file to TF Summary and Saving Visualization of VGG16 graph..............")
        graph_visualize()
