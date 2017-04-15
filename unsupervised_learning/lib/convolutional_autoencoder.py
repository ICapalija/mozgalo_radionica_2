from lib.architectures import ema_ae
from lib.config import cfg
from lib.image_loader import ImageLoader
from lib.visualization import create_summary_embeddings

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

import Queue
import math
import os

def _create_autoencoder_graph(input_shape=[None, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, cfg.CHANNELS], corruption=True):
    """
    Create autoencoder graph.
    :input_shape: Input tensor shape.
    :return: Tensor dict.
    """
    # input to the network graph
    input_image_tensor = tf.placeholder(tf.float32, input_shape, name='input_image')

    current_input = input_image_tensor
    # Optionally apply denoising autoencoder
    if corruption:
        noise = tf.cast(tf.random_uniform(shape=tf.shape(current_input), minval=0, maxval=2, dtype=tf.int32), tf.float32)
        current_input = tf.mul(current_input, noise)
        shape = tf.shape(current_input)
        current_input = tf.reshape(current_input, [-1, shape[1], shape[2], input_shape[3]])

    # create autoencoder
    output_image_tensor, embedding_tensor = ema_ae(current_input)

    tf.add_to_collection("input_image_tensor", input_image_tensor)
    tf.add_to_collection("output_image_tensor", output_image_tensor)
    tf.add_to_collection("embedding_tensor", embedding_tensor)

    return {'input_image_tensor': input_image_tensor,
            'output_image_tensor': output_image_tensor,
            'embedding_tensor': embedding_tensor}


def train(model_path=None):
    """
    Train autoencoder.
    :model_path: Model path.
    :return:
    """

    # define graph
    ae = _create_autoencoder_graph(corruption=True)

    # create loss
    loss = tf.reduce_mean(tf.square(ae["output_image_tensor"] - ae["input_image_tensor"]))

    # init optimizer
    optimizer = tf.train.AdamOptimizer(cfg.LEARNING_RATE).minimize(loss)

    # create session
    sess = tf.Session()

    # create saver
    saver = tf.train.Saver()
    try:
        print("Loading saved model")
        saver.restore(sess, model_path)
        print("Model loaded.")
    except:
        print('Initializing vars')
        sess.run(tf.global_variables_initializer())

    # image loader
    train_loader = ImageLoader(cfg.TRAIN_FOLDER)

    # fit all training data
    iter = 1
    while True:

        # create batch
        batch_xs = []
        for _ in range(cfg.BATCH_SIZE):
            image, img_name, img_path = train_loader.next_image()
            batch_xs.append(image)

        # run optimization step
        sess.run(optimizer, feed_dict={ae['input_image_tensor']: batch_xs})

        # display
        if iter % cfg.DISPLAY_ITER == 0:
            loss_value = sess.run(loss, feed_dict={ae['input_image_tensor']: batch_xs})
            print('Iteration: %d, loss: %f' % (iter, loss_value))

        # save model
        if iter % cfg.SAVE_ITER == 0:
            model_path = os.path.join(cfg.MODEL_PATH, "autoencoder_" + str(iter))
            saver.save(sess, model_path)
            print('Model saved to : %s' % model_path)

        iter += 1

def extract(model_path, vis=True):
    """
    Extract embedding vector for every image.
    :model_path: Model path.
    :return:
    """

    # create session
    sess = tf.Session()

    # create saver
    try:
        print("Import meta graph")
        saver = tf.train.import_meta_graph(model_path + ".meta")
        print("Loading saved model.")
        saver.restore(sess, model_path)
        print("Model loaded.")
    except:
        print('Error while loading model.')
        exit()

    # get input image tensor
    input_image_tensor = tf.get_collection("input_image_tensor")[0]
    # get embedding tensor
    embedding_tensor = tf.get_collection("embedding_tensor")[0]
    # get output image tensor
    output_image_tensor = tf.get_collection("output_image_tensor")[0]

    # create summary writer
    test_writer = tf.summary.FileWriter(cfg.TENSORBOARD_PATH, sess.graph)

    # image loader
    test_loader = ImageLoader(cfg.TEST_FOLDER)

    # image info vars
    images = np.zeros(shape=(test_loader.size, cfg.EMB_IMAGE_HEIGHT, cfg.EMB_IMAGE_WIDTH, 3))
    image_names = []

    # embeddings array
    EMB = None

    # iterate images
    for image_index in range(test_loader.size):
        # load image
        image, img_name, img_path = test_loader.next_image()

        print('Image name: %s' % img_name)

        # forward pass
        output_tensor_value, embedding_tensor_value = sess.run([output_image_tensor, embedding_tensor], feed_dict={input_image_tensor: [image]})

        # flatten 4D tensor
        embedding = embedding_tensor_value.flatten()
        # save embedding
        if EMB == None:
            EMB = np.zeros((test_loader.size, len(embedding)), dtype='float32')
        EMB[image_index] = embedding

        # save image info for visualization
        image = cv2.resize(image, (cfg.EMB_IMAGE_HEIGHT, cfg.EMB_IMAGE_WIDTH), interpolation=cv2.INTER_CUBIC)
        images[image_index] = image
        image_names.append(img_path)

        # vis output image
        if vis:
            print('     Tensor shape = %s' % (str(output_tensor_value.shape)))
            print('     Embedding shape = %s' % (str(embedding.shape)))

            # plot reconstructions
            fig, axs = plt.subplots(2, 1, figsize=(10, 2))
            axs[0].imshow(image[:, :, 0])
            axs[1].imshow(np.array(output_tensor_value[0][:, :, 0]))
            fig.show()
            plt.draw()
            plt.waitforbuttonpress()
            plt.close()

    # save embeddings for tensorboard visualization
    create_summary_embeddings(sess, images, image_names, EMB, cfg.TENSORBOARD_PATH)