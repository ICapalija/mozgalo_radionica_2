from easydict import EasyDict as edict

import os
import os.path as osp

project_dir = os.path.join(os.getcwd())

__C = edict()
cfg = __C

#################
## data config ##
#################
__C.TRAIN_FOLDER = project_dir + '/dataset/data/'
__C.TEST_FOLDER = project_dir + '/dataset/data/'
__C.MODEL_PATH = 'models/'

##################
## train config ##
##################
__C.BATCH_SIZE = 10
__C.DISPLAY_ITER = 100
__C.SAVE_ITER = 1000
__C.LEARNING_RATE = 0.0001
__C.IMAGE_WIDTH = 64
__C.IMAGE_HEIGHT = 64
__C.CHANNELS = 3

####################
## extract config ##
####################
__C.TENSORBOARD_PATH = 'tensorboard/test'
__C.EMB_IMAGE_WIDTH = 300
__C.EMB_IMAGE_HEIGHT = 300
