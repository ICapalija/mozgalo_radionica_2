from lib.image_loader import ImageLoader
from lib.visualization import create_summary_embeddings
from lib.config import cfg

import tensorflow as tf
import numpy as np
import cv2

def _get_graph(model_path):
    """
    Create and return graph.
    :param model_path:
    :return:
    """
    # read model file
    with tf.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import a graph_def into the current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None
        )

        # print layers in graph
        for op in graph.get_operations():
            print("------------------------")
            print(op.name)

    return graph

def extract(model_path, vis=False):
    """
    Extract embedding vector for every image.
    :param model_path: Model path.
    :return:
    """

    # create graph
    try:
        print("Import graph")
        graph = _get_graph(model_path)
        print("Graph loaded.")
    except:
        print('Error while loading model.')
        exit()

    with tf.Session(graph = graph) as sess:

        # set input tensor name
        input_tensor_name = cfg.INPUT_TENSOR_NAME
        # set output tensor name
        output_tensor_name = cfg.OUTPUT_TENSOR_NAME

        # image loader
        test_loader = ImageLoader(cfg.TEST_FOLDER)

        # create summary writer
        test_writer = tf.summary.FileWriter(cfg.TENSORBOARD_PATH, sess.graph)

        # image info vars
        images = np.zeros(shape=(test_loader.size, cfg.EMB_IMAGE_HEIGHT, cfg.EMB_IMAGE_WIDTH, 3))
        image_names = []

        # embeddings array
        EMB = None

        # iterate images
        for image_index in range(test_loader.size):

            # get next image
            image, img_name, img_path = test_loader.next_image()

            print('Image name: %s' % img_name)

            # read image
            image_data = tf.gfile.FastGFile(img_path, 'rb').read()

            # run forward pass
            output_tensor_value = sess.run(output_tensor_name, {input_tensor_name: image_data})

            # flatten 4D tensor
            embedding = output_tensor_value.flatten()
            # save embedding
            if EMB == None:
                EMB = np.zeros((test_loader.size, len(embedding)), dtype='float32')
            EMB[image_index] = embedding

            # save image info for visualization
            image_names.append(img_name)
            image = cv2.resize(image, (cfg.EMB_IMAGE_HEIGHT, cfg.EMB_IMAGE_WIDTH), interpolation = cv2.INTER_CUBIC)
            images[image_index] = image

            if vis:
                print('     Tensor shape = %s' % (str(output_tensor_value.shape)))
                print('     Embedding shape = %s' % (str(embedding.shape)))
                cv2.imshow(img_name, image)
                cv2.waitKey()

        # save embeddings for tensorboard visualization
        create_summary_embeddings(sess, images, image_names, EMB, cfg.TENSORBOARD_PATH)